"""
索引构建脚本（在全量数据导入后运行）：
1. 创建 metadata GIN 索引
2. 创建向量索引（HNSW / IVFFlat，支持 fp16 / fp32 / bit 量化）
3. 创建 BM25 索引（pg_search，content_tokenized 列由 ingest_to_postgres.py 直接写入）
4. ANALYZE 更新统计信息

用法:
  python scripts/build_index.py                           # 默认 HNSW fp16
  python scripts/build_index.py --method ivfflat
  python scripts/build_index.py --method hnsw --m 32 --ef-construction 128
  python scripts/build_index.py --quantize bit            # 二值量化
  python scripts/build_index.py --dim 512                 # 降维到 512 维
"""
import os
import time
import argparse
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")
DB_USER = os.getenv("POSTGRES_USER", "rag_user")

ORIGINAL_DIM = 1024


def timed(desc):
    """简易计时装饰器"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            print(f"\n[开始] {desc}...")
            t0 = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            m, s = divmod(int(elapsed), 60)
            print(f"[完成] {desc}  ({m}分{s}秒)")
            return result
        return wrapper
    return decorator


def parse_args():
    p = argparse.ArgumentParser(description="WikiRAG 索引构建")

    p.add_argument("--method", choices=["ivfflat", "hnsw"], default="hnsw",
                   help="向量索引类型 (默认: hnsw)")

    # IVFFlat 参数
    p.add_argument("--lists", type=int, default=1000,
                   help="IVFFlat 聚类数 (默认: 1000，建议 sqrt(行数))")

    # HNSW 参数
    p.add_argument("--m", type=int, default=16,
                   help="HNSW 每层连接数 (默认: 16，越大越精准但越占内存)")
    p.add_argument("--ef-construction", type=int, default=32,
                   help="HNSW 构建时搜索宽度 (默认: 32，必须 >= 2*m，越大构建越慢但质量越高)")

    # 量化
    p.add_argument("--quantize", choices=["fp32", "fp16", "bit"], default="fp16",
                   help="索引精度: fp32=原始存储精度, fp16=半精度(默认,与嵌入原始精度一致), bit=二值量化")

    # 降维
    p.add_argument("--dim", type=int, default=ORIGINAL_DIM,
                   help=f"向量维度 (默认: {ORIGINAL_DIM}，降维时指定目标维度如 512)")

    return p.parse_args()


@timed("创建 metadata GIN 索引")
def create_metadata_index(cursor, conn):
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_wiki_documents_metadata
        ON wiki_documents USING gin (metadata);
    """)
    conn.commit()



def create_vector_index(cursor, conn, args):
    """根据参数创建向量索引，索引名包含配置信息以支持多索引共存"""
    method = args.method
    quantize = args.quantize
    dim = args.dim

    # 参数校验
    if method == "hnsw" and args.ef_construction < 2 * args.m:
        print(f"❌ ef_construction ({args.ef_construction}) 必须 >= 2 * m ({2 * args.m})")
        raise SystemExit(1)

    # 索引名编码配置
    if method == "hnsw":
        index_name = f"idx_emb_{method}_{quantize}_{dim}_m{args.m}_ef{args.ef_construction}"
    else:
        index_name = f"idx_emb_{method}_{quantize}_{dim}_l{args.lists}"

    # 检查是否已存在
    cursor.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s;", (index_name,))
    if cursor.fetchone():
        print(f"\n[跳过] 索引 {index_name} 已存在")
        return

    # 构建表达式和 ops（存储已经是 halfvec）
    if quantize == "fp16":
        if dim != ORIGINAL_DIM:
            expression = f"(embedding::halfvec({dim}))"
        else:
            expression = "embedding"
        ops = "halfvec_cosine_ops"
    elif quantize == "fp32":
        expression = f"(embedding::vector({dim}))"
        ops = "vector_cosine_ops"
    elif quantize == "bit":
        expression = f"(binary_quantize(embedding::vector({dim}))::bit({dim}))"
        ops = "bit_hamming_ops"
    else:
        expression = "embedding"
        ops = "halfvec_cosine_ops"

    # 构建索引参数
    if method == "hnsw":
        with_clause = f"WITH (m = {args.m}, ef_construction = {args.ef_construction})"
        desc = f"HNSW (m={args.m}, ef_construction={args.ef_construction})"
    else:
        with_clause = f"WITH (lists = {args.lists})"
        desc = f"IVFFlat (lists={args.lists})"

    if quantize != "fp32":
        desc += f", quantize={quantize}"
    if dim != ORIGINAL_DIM:
        desc += f", dim={dim}"

    sql = f"""
        CREATE INDEX {index_name}
        ON wiki_documents USING {method} ({expression} {ops})
        {with_clause};
    """

    print(f"\n[开始] 创建向量索引: {index_name}")
    print(f"  配置: {desc}")
    print(f"  SQL: {sql.strip()}")
    t0 = time.time()
    cursor.execute(sql)
    conn.commit()
    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"[完成] {index_name}  ({m}分{s}秒)")



@timed("创建 BM25 索引 (pg_search / Tantivy)")
def create_bm25_index(cursor, conn):
    """使用 pg_search 扩展构建真正的 BM25 索引（有 IDF、TF 饱和、长度归一化）。
    索引基于 content_tokenized 列（jieba 空格分词），使用 whitespace tokenizer。
    若 pg_search 未安装则跳过。
    """
    cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_search';")
    if not cursor.fetchone():
        print("  ⚠ pg_search 扩展未安装，跳过 BM25 索引（请切换到 paradedb/paradedb:latest-pg17 镜像）")
        return

    # 确认 content_tokenized 已填充
    cursor.execute("SELECT COUNT(*) FROM wiki_documents WHERE content_tokenized = '';")
    empty = cursor.fetchone()[0]
    if empty > 0:
        print(f"  ⚠ {empty:,} 行 content_tokenized 为空，请先运行 migrate_to_bm25.py，跳过 BM25 索引")
        return

    cursor.execute("SELECT 1 FROM pg_indexes WHERE indexname = 'idx_wiki_bm25';")
    if cursor.fetchone():
        print("  [跳过] idx_wiki_bm25 已存在")
        return

    cursor.execute("""
        CREATE INDEX idx_wiki_bm25
        ON wiki_documents
        USING bm25 (id, content_tokenized)
        WITH (
            key_field = 'id',
            text_fields = '{"content_tokenized": {"tokenizer": {"type": "whitespace"}}}'
        );
    """)
    conn.commit()


def main():
    args = parse_args()

    print(f"连接 PostgreSQL {DB_HOST}:{DB_PORT}/{DB_NAME}...")
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER)
    conn.autocommit = True
    cursor = conn.cursor()

    cursor.execute("SET maintenance_work_mem = '256MB';")

    # 确认数据存在
    cursor.execute("SELECT COUNT(*) FROM wiki_documents;")
    count = cursor.fetchone()[0]
    if count == 0:
        print("❌ wiki_documents 表为空，请先运行 ingest_to_postgres.py")
        return
    print(f"数据行数: {count}")

    # 打印配置
    print(f"\n向量索引配置:")
    print(f"  方法: {args.method}")
    if args.method == "hnsw":
        print(f"  m: {args.m}, ef_construction: {args.ef_construction}")
    else:
        print(f"  lists: {args.lists}")
    print(f"  量化: {args.quantize}")
    print(f"  维度: {args.dim}")

    create_metadata_index(cursor, conn)
    create_vector_index(cursor, conn, args)
    create_bm25_index(cursor, conn)

    print("\n[开始] ANALYZE wiki_documents（更新 planner 统计信息）...")
    t0 = time.time()
    cursor.execute("ANALYZE wiki_documents;")
    print(f"[完成] ANALYZE  ({time.time() - t0:.1f}秒)")

    cursor.close()
    conn.close()
    print("\n✅ 全部索引构建完成！")


if __name__ == "__main__":
    main()

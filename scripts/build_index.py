"""
索引构建脚本（在全量数据导入后运行）：
1. 从临时分词列构建 tsvector (title=A, content=B)
2. 清理临时列
3. 创建 GIN / 向量索引
4. 将 UNLOGGED 表转为 LOGGED

用法:
  python scripts/build_index.py                           # 默认 HNSW
  python scripts/build_index.py --method ivfflat          # IVFFlat default:1000
  python scripts/build_index.py --method hnsw --m 32 --ef-construction 128
  python scripts/build_index.py --method ivfflat --lists 2000
  python scripts/build_index.py --method hnsw --quantize halfvec    # FP16 量化
  python scripts/build_index.py --method hnsw --quantize bit        # 二值量化
  python scripts/build_index.py --dim 512                           # PCA 降维到 512 维
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
    p.add_argument("--ef-construction", type=int, default=64,
                   help="HNSW 构建时搜索宽度 (默认: 64，越大构建越慢但质量越高)")

    # 量化
    p.add_argument("--quantize", choices=["none", "halfvec", "bit"], default="halfvec",
                   help="量化方式: none=原始FP32, halfvec=FP16半精度, bit=二值量化, 注意本身就是FP16嵌入")

    # 降维
    p.add_argument("--dim", type=int, default=ORIGINAL_DIM,
                   help=f"向量维度 (默认: {ORIGINAL_DIM}，降维时指定目标维度如 512)")

    return p.parse_args()


@timed("构建 tsvector (title=A, content=B)")
def build_tsvector(cursor, conn):
    cursor.execute("""
        UPDATE wiki_documents SET tsv =
            setweight(to_tsvector('simple', title_seg), 'A') ||
            setweight(to_tsvector('simple', content_seg), 'B');
    """)
    conn.commit()


@timed("清理临时分词列")
def drop_seg_columns(cursor, conn):
    cursor.execute("ALTER TABLE wiki_documents DROP COLUMN title_seg;")
    cursor.execute("ALTER TABLE wiki_documents DROP COLUMN content_seg;")
    conn.commit()


@timed("创建 metadata GIN 索引")
def create_metadata_index(cursor, conn):
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_wiki_documents_metadata
        ON wiki_documents USING gin (metadata);
    """)
    conn.commit()


@timed("创建 tsv GIN 索引")
def create_tsv_index(cursor, conn):
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_wiki_documents_tsv
        ON wiki_documents USING gin (tsv);
    """)
    conn.commit()


def create_vector_index(cursor, conn, args):
    """根据参数创建向量索引，索引名包含配置信息以支持多索引共存"""
    method = args.method
    quantize = args.quantize
    dim = args.dim

    # 索引名编码配置：idx_emb_hnsw_halfvec_512_m16_ef64 / idx_emb_ivfflat_none_1024_l1000
    if method == "hnsw":
        index_name = f"idx_emb_{method}_{quantize}_{dim}_m{args.m}_ef{args.ef_construction}"
    else:
        index_name = f"idx_emb_{method}_{quantize}_{dim}_l{args.lists}"

    # 检查是否已存在
    cursor.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s;", (index_name,))
    if cursor.fetchone():
        print(f"\n[跳过] 索引 {index_name} 已存在")
        return

    # 构建表达式和 ops
    if quantize == "halfvec":
        expression = f"(embedding::halfvec({dim}))"
        ops = "halfvec_cosine_ops"
    elif quantize == "bit":
        expression = f"(binary_quantize(embedding)::bit({dim}))"
        ops = "bit_hamming_ops"
    elif dim != ORIGINAL_DIM:
        # 降维但不量化：用 halfvec 做降维容器（pgvector 不支持原生 PCA，
        # 这里截取前 N 维作为简易降维，实际生产中应用 PCA 矩阵）
        expression = f"(embedding::halfvec({dim}))"
        ops = "halfvec_cosine_ops"
    else:
        expression = "embedding"
        ops = "vector_cosine_ops"

    # 构建索引参数
    if method == "hnsw":
        with_clause = f"WITH (m = {args.m}, ef_construction = {args.ef_construction})"
        desc = f"HNSW (m={args.m}, ef_construction={args.ef_construction})"
    else:
        with_clause = f"WITH (lists = {args.lists})"
        desc = f"IVFFlat (lists={args.lists})"

    if quantize != "none":
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

    # 检查临时列是否还在
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'wiki_documents' AND column_name = 'title_seg';
    """)
    has_seg_columns = cursor.fetchone() is not None

    if has_seg_columns:
        build_tsvector(cursor, conn)
        drop_seg_columns(cursor, conn)
    else:
        print("\n临时分词列已清理，跳过 tsvector 构建")

    create_metadata_index(cursor, conn)
    create_tsv_index(cursor, conn)
    create_vector_index(cursor, conn, args)

    cursor.close()
    conn.close()
    print("\n✅ 全部索引构建完成！")


if __name__ == "__main__":
    main()

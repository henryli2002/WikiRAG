"""
全量数据导入：Parquet → PostgreSQL
jieba 分词走多进程，充分利用多核 CPU。
"""
import os
import re
import json
import glob
import logging
import psycopg2
import pandas as pd
import jieba
from io import StringIO
from multiprocessing import Pool, cpu_count
from dotenv import load_dotenv

# 静默 jieba 的日志输出
jieba.setLogLevel(logging.CRITICAL)

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")
DB_USER = os.getenv("POSTGRES_USER", "rag_user")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET_DIR = os.path.join(PROJECT_ROOT, "data", "parquet_output")

LOCK_FILE = os.path.join(PROJECT_ROOT, "data", ".ingest.lock")
WORKERS = max(1, cpu_count() - 1)


def _check_lock():
    """检查文件锁，防止误操作重复导入"""
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, "r") as f:
            info = f.read().strip()
        print(f"❌ 检测到文件锁: {LOCK_FILE}")
        print(f"   {info}")
        print(f"\n   如需重新导入，请先手动删除锁文件:")
        print(f"   rm {LOCK_FILE}")
        raise SystemExit(1)


def _create_lock(row_count: int):
    """导入成功后创建文件锁"""
    from datetime import datetime
    with open(LOCK_FILE, "w") as f:
        f.write(f"上次导入: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 共 {row_count} 行\n")
        f.write("删除此文件以允许重新导入\n")


def _init_jieba():
    """子进程初始化 jieba（静默加载，每个进程只加载一次词典）"""
    jieba.setLogLevel(logging.CRITICAL)
    jieba.initialize()


def _segment(text: str) -> str:
    tokens = jieba.cut(text)
    return " ".join(t for t in tokens if t.strip() and not re.match(r'^[\s\W]+$', t))


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def _process_row(args):
    """处理单行：序列化 + 分词，返回一行 TSV 字符串"""
    content, title, pageid, embedding_list = args

    content_esc = _escape(str(content))
    metadata = json.dumps({"pageid": pageid, "title": title}, ensure_ascii=False)
    metadata_esc = _escape(metadata)
    embedding = "[" + ",".join(str(float(x)) for x in embedding_list) + "]"

    content_tokenized = _escape(_segment(str(content)))

    return f"{content_esc}\t{metadata_esc}\t{embedding}\t{content_tokenized}\n"


def main():
    _check_lock()

    print(f"连接 PostgreSQL {DB_HOST}:{DB_PORT}/{DB_NAME}...")
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER)
    conn.autocommit = True
    cursor = conn.cursor()

    # 重建裸表（无索引、无约束，最大化写入速度）
    print("重建 wiki_documents 表...")
    cursor.execute("DROP TABLE IF EXISTS wiki_documents;")
    cursor.execute("""
        CREATE TABLE wiki_documents (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}',
            embedding HALFVEC(1024),
            content_tokenized TEXT NOT NULL DEFAULT ''
        );
    """)

    cursor.execute("ALTER TABLE wiki_documents DISABLE TRIGGER ALL;")
    cursor.execute("SET synchronous_commit = off;")
    cursor.execute("SET maintenance_work_mem = '2GB';")
    conn.autocommit = False

    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    print(f"找到 {len(parquet_files)} 个 parquet 文件，使用 {WORKERS} 个分词进程\n")

    pool = Pool(processes=WORKERS, initializer=_init_jieba)

    total_rows = 0
    for file_path in parquet_files:
        fname = os.path.basename(file_path)
        df = pd.read_parquet(file_path)

        # 准备多进程参数
        args_list = [
            (row["content"], row["title"], row["pageid"], row["embedding"].tolist())
            for _, row in df.iterrows()
        ]

        # 多进程分词 + 序列化
        lines = pool.map(_process_row, args_list, chunksize=256)

        # COPY 写入
        buf = StringIO("".join(lines))
        cursor.copy_from(
            buf,
            table="wiki_documents",
            columns=("content", "metadata", "embedding", "content_tokenized"),
            sep="\t",
            null="\\N",
        )
        conn.commit()
        total_rows += len(df)
        print(f"  {fname} → {len(df)} 行 (累计 {total_rows})")

    pool.close()
    pool.join()

    cursor.execute("ALTER TABLE wiki_documents ENABLE TRIGGER ALL;")
    cursor.execute("SET synchronous_commit = on;")
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM wiki_documents;")
    final_count = cursor.fetchone()[0]

    cursor.close()
    conn.close()
    _create_lock(final_count)

    print(f"\n✅ 数据导入完成！共 {final_count} 行")
    print("下一步请运行: python scripts/build_index.py")


if __name__ == "__main__":
    main()

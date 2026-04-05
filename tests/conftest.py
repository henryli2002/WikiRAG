import os
import psycopg2
import pytest

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")
DB_USER = os.getenv("POSTGRES_USER", "rag_user")


@pytest.fixture(scope="session")
def db_conn():
    """整个测试会话复用一个数据库连接"""
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER)
    yield conn
    conn.close()


@pytest.fixture
def cursor(db_conn):
    """每个测试用例独立 cursor"""
    cur = db_conn.cursor()
    yield cur
    cur.close()

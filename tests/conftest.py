import os

# Must run before ragmcp.config is imported: tests only ever touch the test database.
TEST_URL = "postgresql://ragmcp:ragmcp_local_dev@localhost/ragmcp_test"
os.environ["RAGMCP_STORE_URL"] = os.environ.get("RAGMCP_TEST_STORE_URL", TEST_URL)

import psycopg  # noqa: E402
import pytest  # noqa: E402


def _guard(conn) -> None:
    db = conn.execute("SELECT current_database()").fetchone()[0]
    if db != "ragmcp_test":
        raise RuntimeError(f"refusing to wipe database {db!r}; tests require ragmcp_test")


@pytest.fixture(scope="session", autouse=True)
def _fresh_test_schema():
    """Drop tables once so the test schema is built with the tests' dimension (3)."""
    with psycopg.connect(os.environ["RAGMCP_STORE_URL"]) as conn:
        _guard(conn)
        conn.execute("DROP TABLE IF EXISTS chunks, documents CASCADE")
    yield


@pytest.fixture(autouse=True)
def _clean_tables(_fresh_test_schema):
    """Each test starts with empty tables (when they exist)."""
    with psycopg.connect(os.environ["RAGMCP_STORE_URL"]) as conn:
        _guard(conn)
        if conn.execute("SELECT to_regclass('documents')").fetchone()[0]:
            conn.execute("TRUNCATE documents CASCADE")
    yield

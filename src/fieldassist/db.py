from __future__ import annotations

from importlib import resources

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def make_pool(url: str, min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    pool = ConnectionPool(url, min_size=min_size, max_size=max_size, open=False,
                          kwargs={"autocommit": True, "row_factory": dict_row})
    pool.open(wait=True, timeout=30)
    return pool


def apply_schema(pool: ConnectionPool) -> None:
    sql = resources.files("fieldassist").joinpath("schema.sql").read_text(encoding="utf-8")
    with pool.connection() as conn:
        conn.execute(sql)


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"

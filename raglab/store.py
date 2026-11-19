import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def data_dir():
    path = Path(os.getenv("RAGLAB_DATA", "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def connection():
    db = sqlite3.connect(data_dir() / "raglab.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    finally:
        db.close()


def initialize():
    with connection() as db:
        db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, text TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS datasets (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, cases_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS experiments (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
          config_json TEXT NOT NULL, report_json TEXT, error TEXT,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        """)
        # Interrupted jobs cannot silently remain in progress after restart.
        db.execute("UPDATE experiments SET status='failed', error='服务重启，实验已中断，请重新运行。' WHERE status IN ('queued','running')")


def documents():
    with connection() as db:
        return [dict(row) for row in db.execute("SELECT * FROM documents ORDER BY created_at, id")]


def experiment(row):
    value = dict(row)
    value["config"] = json.loads(value.pop("config_json"))
    report = value.pop("report_json")
    value["report"] = json.loads(report) if report else None
    return value

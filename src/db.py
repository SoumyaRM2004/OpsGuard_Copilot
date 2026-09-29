"""SQLite audit trail — stores every Q&A for compliance logging."""

import json
import sqlite3
from datetime import datetime, timezone
from src.config import get_settings


def init_db() -> None:
    path = get_settings().database_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'public',
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            route TEXT,
            used_web INTEGER DEFAULT 0,
            support_status TEXT,
            usefulness TEXT,
            trace_json TEXT,
            sources_json TEXT
        )
        """)
        # Auto-migration: check if user_id column is present on existing database
        cursor = conn.execute("PRAGMA table_info(rag_audit)")
        columns = [row[1] for row in cursor.fetchall()]
        if "user_id" not in columns:
            conn.execute("ALTER TABLE rag_audit ADD COLUMN user_id TEXT NOT NULL DEFAULT 'public'")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_id ON rag_audit(user_id)")
        conn.commit()


def save_audit(user_id: str, question: str, result: dict) -> None:
    path = get_settings().database_file
    uid = (user_id or "public").strip()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """INSERT INTO rag_audit
            (user_id, created_at, question, answer, route, used_web, support_status, usefulness, trace_json, sources_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uid,
                datetime.now(timezone.utc).isoformat(),
                question,
                result.get("answer", ""),
                result.get("route", ""),
                int(bool(result.get("used_web_search"))),
                result.get("support_status", ""),
                result.get("usefulness", ""),
                json.dumps(result.get("trace", []), ensure_ascii=False),
                json.dumps(result.get("sources", []), ensure_ascii=False),
            ),
        )
        conn.commit()


def latest_audits(user_id: str = "", limit: int = 50):
    path = get_settings().database_file
    uid = (user_id or "public").strip()
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM rag_audit WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (uid, limit),
        ).fetchall()
    return [dict(r) for r in rows]
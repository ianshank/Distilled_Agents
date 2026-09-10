"""SQLite-backed error learning."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from enhanced_system.core.errors.types import ErrorRecord

logger = logging.getLogger(__name__)


class ErrorLearner:
    """Track error patterns in a local SQLite database."""

    def __init__(
        self,
        db_path: str = "./data/errors.db",
        enabled: bool = True,
    ):
        self.db_path = self._sqlite_path(db_path)
        self.enabled = enabled
        if enabled:
            self._init_database()

    @staticmethod
    def _sqlite_path(db_path: str) -> str:
        """SQLite cannot open object-store URIs; fall back to a local file."""
        if db_path.startswith(("s3://", "http://", "https://")):
            logger.warning(
                "error_db_path %s is not a SQLite file; using ./data/errors.db",
                db_path,
            )
            return os.path.join("data", "errors.db")
        parent = Path(db_path).expanduser().parent
        if str(parent) not in {"", "."}:
            parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def _init_database(self) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT,
                    error_message TEXT,
                    agent TEXT,
                    task TEXT,
                    timestamp TEXT,
                    retry_count INTEGER,
                    resolved INTEGER,
                    resolution_method TEXT
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_type ON errors(error_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent ON errors(agent)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON errors(timestamp)")
            conn.commit()
            conn.close()
            logger.info("Error database initialized at %s", self.db_path)
        except Exception as exc:
            logger.warning("Failed to initialize error database: %s", exc)

    def record_error(self, error_record: ErrorRecord) -> None:
        if not self.enabled:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO errors (
                    error_type, error_message, agent, task, timestamp,
                    retry_count, resolved, resolution_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    error_record.error_type,
                    error_record.error_message,
                    error_record.agent,
                    error_record.task,
                    error_record.timestamp.isoformat(),
                    error_record.retry_count,
                    1 if error_record.resolved else 0,
                    error_record.resolution_method,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error("Failed to record error: %s", exc)

    def analyze_errors(self, days: int = 7) -> dict[str, Any]:
        if not self.enabled:
            return {"error_patterns": [], "success_rate": 0, "analysis_period_days": days}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT error_type, error_message, agent, COUNT(*) as count
                FROM errors
                WHERE timestamp >= datetime('now', ? || ' days')
                GROUP BY error_type, agent
                ORDER BY count DESC
                LIMIT 10
                """,
                (f"-{days}",),
            )
            error_patterns = [
                {
                    "error_type": row[0],
                    "error_message": row[1],
                    "agent": row[2],
                    "count": row[3],
                }
                for row in cursor.fetchall()
            ]
            cursor.execute(
                """
                SELECT COUNT(CASE WHEN resolved = 1 THEN 1 END) * 100.0 / COUNT(*)
                FROM errors
                WHERE timestamp >= datetime('now', ? || ' days')
                """,
                (f"-{days}",),
            )
            row = cursor.fetchone()
            success_rate = (row[0] if row else 0) or 0
            conn.close()
            return {
                "error_patterns": error_patterns,
                "success_rate": success_rate,
                "analysis_period_days": days,
            }
        except Exception as exc:
            logger.error("Failed to analyze errors: %s", exc)
            return {"error_patterns": [], "success_rate": 0, "analysis_period_days": days}

    def get_error_stats(self) -> dict[str, Any]:
        if not self.enabled:
            return {"total_errors": 0, "by_type": {}}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM errors")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT error_type, COUNT(*) FROM errors GROUP BY error_type")
            by_type = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            return {"total_errors": total, "by_type": by_type}
        except Exception as exc:
            logger.error("Failed to get error stats: %s", exc)
            return {"total_errors": 0, "by_type": {}}

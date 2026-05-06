import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class AlertEvent:
    event_type: str
    confidence_score: float
    metadata: dict
    timestamp: Optional[str] = None
    frame_number: int = 0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class Database:
    def __init__(self, db_path: str = "data/events.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                confidence_score REAL,
                metadata TEXT,
                frame_number INTEGER DEFAULT 0
            )
        """)
        # Migrate existing DBs that don't have frame_number column
        try:
            self._conn.execute("ALTER TABLE events ADD COLUMN frame_number INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
        self._conn.commit()

    def log_event(self, event: AlertEvent):
        self._conn.execute(
            "INSERT INTO events (timestamp, event_type, confidence_score, metadata, frame_number) "
            "VALUES (?, ?, ?, ?, ?)",
            (event.timestamp, event.event_type, event.confidence_score,
             json.dumps(event.metadata), event.frame_number)
        )
        self._conn.commit()

    def get_recent_events(self, limit: int = 50) -> List[dict]:
        cursor = self._conn.execute(
            "SELECT id, timestamp, event_type, confidence_score, metadata, frame_number "
            "FROM events ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

    def clear(self):
        self._conn.execute("DELETE FROM events")
        self._conn.commit()

    def close(self):
        self._conn.close()

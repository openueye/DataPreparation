from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict

import numpy as np


def require_topic_id(topics: Dict[str, int], topic_name: str, db_path: Path) -> int:
    if topic_name not in topics:
        available = ", ".join(sorted(topics)) or "<none>"
        raise ValueError(
            f"Topic '{topic_name}' was not found in {db_path.name}. Available topics: {available}"
        )
    return topics[topic_name]


def topic_timestamp_array(connection: sqlite3.Connection, topic_id: int) -> np.ndarray:
    rows = connection.execute(
        "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp",
        (topic_id,),
    ).fetchall()
    return np.asarray([row[0] for row in rows], dtype=np.int64)

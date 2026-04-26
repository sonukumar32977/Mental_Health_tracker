"""
database.py - SQLite database manager for Mental Health Sentiment Tracker.
Handles creation, insertion, and retrieval of journal entries.
"""

import sqlite3
import json
import os
from datetime import datetime

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "entries.db")


def init_db():
    """Initialize the database and create the entries table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                entry_text TEXT NOT NULL,
                emotions TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                wellness_score REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[DB Error] Failed to initialize database: {e}")


def insert_entry(entry_text: str, emotions: dict, risk_level: str, wellness_score: float):
    """
    Insert a new journal entry into the database.

    Args:
        entry_text (str): The raw journal entry text.
        emotions (dict): Dictionary of emotion labels to confidence scores.
        risk_level (str): Detected mental health risk level.
        wellness_score (float): Computed wellness score (0-100).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO entries (timestamp, entry_text, emotions, risk_level, wellness_score)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, entry_text, json.dumps(emotions), risk_level, wellness_score))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[DB Error] Failed to insert entry: {e}")


def fetch_all_entries() -> list:
    """
    Retrieve all journal entries from the database, ordered by most recent first.

    Returns:
        list: A list of dictionaries representing each entry.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, entry_text, emotions, risk_level, wellness_score FROM entries ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()

        entries = []
        for row in rows:
            entries.append({
                "id": row[0],
                "timestamp": row[1],
                "entry_text": row[2],
                "emotions": json.loads(row[3]),
                "risk_level": row[4],
                "wellness_score": row[5]
            })
        return entries
    except sqlite3.Error as e:
        print(f"[DB Error] Failed to fetch entries: {e}")
        return []


def fetch_recent_entries(limit: int = 30) -> list:
    """
    Retrieve the most recent N journal entries.

    Args:
        limit (int): Maximum number of entries to retrieve.

    Returns:
        list: A list of dictionaries representing each entry.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, entry_text, emotions, risk_level, wellness_score
            FROM entries
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()

        entries = []
        for row in rows:
            entries.append({
                "id": row[0],
                "timestamp": row[1],
                "entry_text": row[2],
                "emotions": json.loads(row[3]),
                "risk_level": row[4],
                "wellness_score": row[5]
            })
        return entries
    except sqlite3.Error as e:
        print(f"[DB Error] Failed to fetch recent entries: {e}")
        return []


def delete_entry(entry_id: int):
    """
    Delete a specific entry by its ID.

    Args:
        entry_id (int): The ID of the entry to delete.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[DB Error] Failed to delete entry: {e}")


# Initialize the database when this module is imported
init_db()

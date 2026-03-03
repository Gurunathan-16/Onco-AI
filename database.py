import sqlite3
from datetime import datetime


def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            model_type TEXT,
            result TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(image_name, model_type, result, confidence):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (image_name, model_type, result, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (
        image_name,
        model_type,
        result,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()
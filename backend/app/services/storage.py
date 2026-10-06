"""MindTrack Storage Service with Dual Persistence (MongoDB + SQLite Fallback).

If MONGODB_URI is provided and accessible, it uses MongoDB.
Otherwise, it automatically uses a reliable local SQLite database.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from backend.app.core.config import settings

class StorageService:
    def __init__(self):
        self.use_mongo = False
        self.mongo_client = None
        self.db = None
        self.collection = None
        self.storage_type = "sqlite"

        # Attempt MongoDB connection if configured
        if settings.MONGODB_URI:
            try:
                from pymongo import MongoClient
                self.mongo_client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)
                # Quick ping
                self.mongo_client.admin.command('ping')
                self.db = self.mongo_client[settings.DATABASE_NAME]
                self.collection = self.db["predictions"]
                self.use_mongo = True
                self.storage_type = "mongodb"
                print("Connected to MongoDB successfully for prediction history.")
            except Exception as e:
                print(f"MongoDB not accessible ({e}). Falling back to SQLite storage.")
                self.use_mongo = False

        if not self.use_mongo:
            self._init_sqlite()

    def _init_sqlite(self):
        """Initializes the local SQLite database schema."""
        self.storage_type = "sqlite"
        settings.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    predicted_score REAL NOT NULL,
                    wellbeing_tier TEXT NOT NULL,
                    tier_color TEXT NOT NULL,
                    model_used TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp DESC)")
            conn.commit()

    def save_prediction(
        self,
        input_data: Dict[str, Any],
        predicted_score: float,
        wellbeing_tier: str,
        tier_color: str,
        model_used: str
    ) -> str:
        """Stores a prediction record and returns its unique ID."""
        rec_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        if self.use_mongo and self.collection is not None:
            doc = {
                "_id": rec_id,
                "timestamp": now_iso,
                "input_data": input_data,
                "predicted_score": predicted_score,
                "wellbeing_tier": wellbeing_tier,
                "tier_color": tier_color,
                "model_used": model_used
            }
            self.collection.insert_one(doc)
            return rec_id

        # SQLite fallback
        with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (id, timestamp, input_data, predicted_score, wellbeing_tier, tier_color, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                rec_id,
                now_iso,
                json.dumps(input_data),
                predicted_score,
                wellbeing_tier,
                tier_color,
                model_used
            ))
            conn.commit()

        return rec_id

    def get_recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent predictions ordered from newest to oldest."""
        if self.use_mongo and self.collection is not None:
            cursor = self.collection.find().sort("timestamp", -1).limit(limit)
            records = []
            for doc in cursor:
                records.append({
                    "id": str(doc.get("_id")),
                    "timestamp": doc.get("timestamp"),
                    "input_data": doc.get("input_data"),
                    "predicted_mental_health_score": doc.get("predicted_score"),
                    "wellbeing_tier": doc.get("wellbeing_tier"),
                    "tier_color": doc.get("tier_color"),
                    "model_used": doc.get("model_used")
                })
            return records

        # SQLite fallback
        records = []
        with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, input_data, predicted_score, wellbeing_tier, tier_color, model_used
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            for row in cursor.fetchall():
                records.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "input_data": json.loads(row["input_data"]),
                    "predicted_mental_health_score": row["predicted_score"],
                    "wellbeing_tier": row["wellbeing_tier"],
                    "tier_color": row["tier_color"],
                    "model_used": row["model_used"]
                })
        return records

    def delete_prediction(self, rec_id: str) -> bool:
        """Deletes a specific prediction by ID."""
        if self.use_mongo and self.collection is not None:
            res = self.collection.delete_one({"_id": rec_id})
            return res.deleted_count > 0

        with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE id = ?", (rec_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self) -> int:
        """Clears all prediction records."""
        if self.use_mongo and self.collection is not None:
            res = self.collection.delete_many({})
            return res.deleted_count

        with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions")
            conn.commit()
            return cursor.rowcount

storage_service = StorageService()

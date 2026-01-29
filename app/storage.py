import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


TABLE_SQL = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    input_type TEXT,
    input_name TEXT,
    overall_book_count INTEGER,
    per_shelf_json TEXT,
    roi_mode TEXT,
    roi_bbox TEXT,
    count_before_filter INTEGER,
    count_after_filter INTEGER,
    quality TEXT,
    imgsz_used INTEGER,
    device_used TEXT,
    conf_used REAL,
    iou_used REAL,
    model_seg_path TEXT,
    model_cls_path TEXT,
    processing_ms REAL,
    output_path TEXT
);
"""


def init_db(db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(TABLE_SQL)
        conn.commit()
    ensure_columns(
        db_path,
        {
            "roi_bbox": "TEXT",
            "count_before_filter": "INTEGER",
            "count_after_filter": "INTEGER",
        },
    )


def ensure_columns(db_path: str | Path, columns: dict) -> None:
    with sqlite3.connect(db_path) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
        for name, col_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE requests ADD COLUMN {name} {col_type}")
        conn.commit()


def insert_request(db_path: str | Path, payload: dict) -> int:
    keys = [
        "ts",
        "input_type",
        "input_name",
        "overall_book_count",
        "per_shelf_json",
        "roi_mode",
        "roi_bbox",
        "count_before_filter",
        "count_after_filter",
        "quality",
        "imgsz_used",
        "device_used",
        "conf_used",
        "iou_used",
        "model_seg_path",
        "model_cls_path",
        "processing_ms",
        "output_path",
    ]

    values = [payload.get(k) for k in keys]
    placeholders = ",".join(["?"] * len(keys))

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO requests ({','.join(keys)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return int(cur.lastrowid)


def _range_since(range_filter: str):
    now = datetime.now()
    if range_filter == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if range_filter == "7d":
        return now - timedelta(days=7)
    return None


def fetch_requests(db_path: str | Path, limit: int | None = 50, range_filter: str = "all"):
    since = _range_since(range_filter)
    query = "SELECT * FROM requests"
    params = []
    if since:
        query += " WHERE ts >= ?"
        params.append(since.isoformat())
    query += " ORDER BY id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def summary_stats(db_path: str | Path, range_filter: str = "all"):
    rows = fetch_requests(db_path, limit=None, range_filter=range_filter)
    if not rows:
        return {
            "count": 0,
            "avg_books": 0.0,
            "avg_ms": 0.0,
        }

    count = len(rows)
    books = [r.get("overall_book_count") or 0.0 for r in rows]
    ms = [r.get("processing_ms") or 0.0 for r in rows]

    return {
        "count": count,
        "avg_books": round(sum(books) / count, 2),
        "avg_ms": round(sum(ms) / count, 2),
    }


def payload_from_metrics(metrics: dict, cfg: dict, input_type: str, input_name: str, roi_mode: str, quality: str, output_path: str | None):
    overall = metrics.get("overall") or {}
    per_shelf_json = json.dumps(metrics.get("shelves") or [], ensure_ascii=False)
    roi_bbox = metrics.get("roi_bbox")
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "input_type": input_type,
        "input_name": input_name,
        "overall_book_count": overall.get("total_books"),
        "per_shelf_json": per_shelf_json,
        "roi_mode": roi_mode,
        "roi_bbox": json.dumps(roi_bbox) if roi_bbox is not None else None,
        "count_before_filter": metrics.get("count_before_filter"),
        "count_after_filter": metrics.get("count_after_filter"),
        "quality": quality,
        "imgsz_used": metrics.get("imgsz_used") or overall.get("imgsz_used"),
        "device_used": metrics.get("device_used") or overall.get("device_used"),
        "conf_used": metrics.get("conf_used") or overall.get("conf"),
        "iou_used": metrics.get("iou_used") or overall.get("iou"),
        "model_seg_path": cfg.get("paths", {}).get("seg_model"),
        "model_cls_path": cfg.get("paths", {}).get("cls_model"),
        "processing_ms": metrics.get("processing_ms") or overall.get("processing_ms"),
        "output_path": output_path,
    }

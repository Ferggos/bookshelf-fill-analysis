# Bookshelf Fill Analysis

Minimal Gradio app that estimates bookshelf content from a single image. The app detects book spines via segmentation, groups them into shelves, counts books per shelf, and keeps a history with PDF/Excel export.

## Features
- Book‑spine segmentation with simple shelf grouping (per_shelf / books_band).
- Shelf‑level book counts + total books.
- Visual overlay: book masks + shelf bands + summary line.
- Request history in SQLite.
- Export to Excel and PDF.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

## Configuration
All settings live in `config.yaml`.

**Model paths** (required for inference):
```yaml
paths:
  seg_model: "best.pt"        # segmentation weights (required)
  cls_model: "yolo11n-cls.pt" # optional classification weights
```

**Inference defaults** (safe for 6GB GPU):
- `imgsz_fast/balanced/quality` → 640 / 768 / 960
- If CUDA OOM: retry 640, then fallback to CPU
- `roi_mode_default`: `books_band`
- `per_shelf_threshold_px` and `per_shelf_threshold_frac` control shelf grouping

## Output format (JSON)
The UI shows a human‑readable summary and keeps raw JSON under “Raw JSON”. The structure:
```json
{
  "overall": {
    "shelf_count": 2,
    "total_books": 61,
    "imgsz_used": 768,
    "device_used": "cuda",
    "processing_ms": 112.4,
    "conf": 0.2,
    "iou": 0.3
  },
  "shelves": [
    {"shelf_index": 1, "roi_bbox": [x1,y1,x2,y2], "book_count": 30},
    {"shelf_index": 2, "roi_bbox": [x1,y1,x2,y2], "book_count": 31}
  ],
  "cls": {"top1": "...", "conf": 0.42}
}
```

## History & Export
- SQLite DB: `data/app.db`
- History tab supports export to:
  - Excel (Requests + Summary)
  - PDF (summary + table + image samples)

## Repo layout
```
bookshelf-fill-v2/
  app/
    main.py
    config.py
    infer.py
    metrics.py
    storage.py
    reports.py
    utils.py
  data/
    uploads/
    results/
  scripts/
    train_seg.py
  config.yaml
  requirements.txt
  README.md
  .gitignore
```

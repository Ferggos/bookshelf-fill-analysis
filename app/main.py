import logging

import gradio as gr
import pandas as pd

from .config import load_config, repo_root
from .infer import run_image_inference
from .reports import generate_excel, generate_pdf
from .storage import init_db, insert_request, fetch_requests, summary_stats, payload_from_metrics
from .utils import ensure_dir, now_ts, save_image_rgb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = repo_root()
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
DB_PATH = DATA_DIR / "app.db"

ensure_dir(UPLOADS_DIR)
ensure_dir(RESULTS_DIR)
init_db(DB_PATH)

CFG = load_config()

UI_CSS = """
:root {
  --cold-bg: #0b1117;
  --cold-panel: #111a22;
  --cold-border: #1f2a35;
  --cold-text: #e5eef7;
  --cold-muted: #9fb2c6;
  --cold-accent: #4aa3d8;
  --cold-accent-2: #6bbbe8;
  --cold-accent-3: #2f7db3;
  --cold-input: #0f151d;
}

body, .gradio-container {
  background: var(--cold-bg) !important;
  color: var(--cold-text);
  font-family: "IBM Plex Sans", "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif;
}

.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
  color: var(--cold-text);
  letter-spacing: 0.2px;
}

.gradio-container .block,
.gradio-container .gr-panel,
.gradio-container .gr-box {
  background: var(--cold-panel) !important;
  border: 1px solid var(--cold-border) !important;
  box-shadow: none !important;
}

.gradio-container .gr-input, 
.gradio-container .gr-output,
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  background: var(--cold-input) !important;
  border: 1px solid var(--cold-border) !important;
  color: var(--cold-text) !important;
}

.gradio-container label,
.gradio-container .gr-form label {
  color: var(--cold-muted) !important;
  font-weight: 600;
}

.gradio-container .gr-button {
  background: var(--cold-accent) !important;
  border: 1px solid var(--cold-accent) !important;
  color: #ffffff !important;
  border-radius: 8px;
  padding: 10px 16px;
}

.gradio-container .gr-button:hover {
  background: var(--cold-accent-2) !important;
  border-color: var(--cold-accent-2) !important;
}

.gradio-container .tabs > .tab-nav button {
  color: var(--cold-muted) !important;
  border-bottom: 2px solid transparent !important;
}

.gradio-container .tabs > .tab-nav button.selected {
  color: var(--cold-text) !important;
  border-bottom: 2px solid var(--cold-accent) !important;
}

.gradio-container .gr-accordion {
  border: 1px dashed var(--cold-border) !important;
  background: transparent !important;
}

.gradio-container .gr-slider input[type="range"] {
  accent-color: var(--cold-accent);
}

.gradio-container .prose a {
  color: var(--cold-accent) !important;
}

.gradio-container .wrap.svelte-1f354aw,
.gradio-container .wrap {
  background: var(--cold-panel) !important;
}

.app-title {
  margin: 4px 0 16px 0;
  font-size: 28px;
  font-weight: 700;
  letter-spacing: 0.4px;
}

.app-title .accent {
  color: var(--cold-accent);
}

.app-subtitle {
  margin-top: -6px;
  color: var(--cold-muted);
  font-size: 13px;
  letter-spacing: 0.3px;
}

/* Conf/IoU sliders: make handles more visible on dark */
.gradio-container input[type="range"]::-webkit-slider-thumb {
  background: var(--cold-accent) !important;
  border: 2px solid var(--cold-accent-3) !important;
  width: 14px;
  height: 14px;
  border-radius: 999px;
}
.gradio-container input[type="range"]::-moz-range-thumb {
  background: var(--cold-accent) !important;
  border: 2px solid var(--cold-accent-3) !important;
  width: 14px;
  height: 14px;
  border-radius: 999px;
}
"""


def _defaults():
    inf = CFG.get("inference", {})
    return {
        "conf": float(inf.get("conf_default", 0.2)),
        "iou": float(inf.get("iou_default", 0.3)),
        "quality": inf.get("quality", "balanced"),
        "roi_mode": inf.get("roi_mode_default", "books_band"),
    }


def format_metrics_text(metrics: dict) -> str:
    if not metrics or metrics.get("error"):
        return f"Error: {metrics.get('error')}" if metrics else "Error: no metrics"

    overall = metrics.get("overall") or {}
    shelves = metrics.get("shelves") or []
    shelves_count = int(overall.get("shelf_count") or len(shelves))
    total_books = int(overall.get("total_books") or 0)

    def _fmt_ms(value):
        return f"{int(round(float(value)))} ms" if value is not None else "n/a"

    lines = [
        "**Overall summary**",
        f"Total shelves detected: {shelves_count}",
        f"Total books detected: {total_books}",
        f"Inference: {overall.get('imgsz_used', 'n/a')}px "
        f"on {overall.get('device_used', 'n/a')}",
        f"Thresholds: conf={overall.get('conf', 'n/a')}, "
        f"iou={overall.get('iou', 'n/a')}",
        f"Processing time: {_fmt_ms(overall.get('processing_ms'))}",
        "",
    ]

    if len(shelves) > 1:
        for shelf in shelves:
            lines.append(f"Shelf {shelf.get('shelf_index', 0)}: books={shelf.get('book_count', 0)}")
    else:
        shelf = shelves[0] if shelves else {}
        lines.append(f"Single shelf band: books={shelf.get('book_count', 0)}")

    cls_info = metrics.get("cls")
    if cls_info:
        lines.append("")
        lines.append("**Classification (ImageNet):**")
        if isinstance(cls_info, dict) and cls_info.get("error"):
            lines.append(f"  Error: {cls_info.get('error')}")
        elif isinstance(cls_info, dict):
            lines.append(f"  Top-1: {cls_info.get('top1')} ({cls_info.get('conf', 0.0):.2f})")

    return "\n".join(lines)


def process_image_ui(image, roi_mode, quality, conf, iou, run_classification):
    if image is None:
        metrics = {"error": "No image provided"}
        return None, format_metrics_text(metrics), metrics

    ts = now_ts()
    input_name = f"image_{ts}.jpg"
    upload_path = UPLOADS_DIR / input_name
    save_image_rgb(image, upload_path)

    output_name = f"result_{ts}.jpg"
    annotated, metrics, output_path = run_image_inference(
        image,
        CFG,
        roi_mode=roi_mode,
        quality=quality,
        conf=conf,
        iou=iou,
        run_classification=run_classification,
        save_dir=RESULTS_DIR,
        output_name=output_name,
    )

    payload = payload_from_metrics(
        metrics,
        CFG,
        input_type="image",
        input_name=input_name,
        roi_mode=roi_mode,
        quality=quality,
        output_path=output_path,
    )
    insert_request(DB_PATH, payload)

    summary_text = format_metrics_text(metrics)
    return annotated, summary_text, metrics


def load_history(range_filter):
    rows = fetch_requests(DB_PATH, limit=50, range_filter=range_filter)
    df = pd.DataFrame(rows)
    summary = summary_stats(DB_PATH, range_filter=range_filter)
    return df, summary


def download_excel(range_filter):
    ts = now_ts()
    path = RESULTS_DIR / f"report_{ts}.xlsx"
    return generate_excel(DB_PATH, path, range_filter=range_filter)


def download_pdf(range_filter):
    ts = now_ts()
    path = RESULTS_DIR / f"report_{ts}.pdf"
    return generate_pdf(DB_PATH, path, range_filter=range_filter)


with gr.Blocks(title="Bookshelf Fill Analysis", css=UI_CSS) as demo:
    gr.Markdown(
        """
<div class="app-title">Bookshelf <span class="accent">Fill</span> Analysis</div>
        """
    )

    defaults = _defaults()

    with gr.Tab("Image"):
        image_input = gr.Image(type="numpy", label="Input image")
        roi_mode = gr.Dropdown(["books_band", "per_shelf"], value=defaults["roi_mode"], label="ROI mode")
        quality = gr.Dropdown(["fast", "balanced", "quality"], value=defaults["quality"], label="Quality")
        conf = gr.Slider(0.05, 0.9, value=defaults["conf"], step=0.05, label="Confidence")
        iou = gr.Slider(0.05, 0.9, value=defaults["iou"], step=0.05, label="IoU")
        run_cls = gr.Checkbox(value=False, label="Run classification")
        process_btn = gr.Button("Process")
        annotated_out = gr.Image(type="numpy", label="Annotated image")
        summary_out = gr.Markdown()
        with gr.Accordion("Raw JSON", open=False):
            metrics_out = gr.JSON(label="Metrics")

        process_btn.click(
            process_image_ui,
            inputs=[image_input, roi_mode, quality, conf, iou, run_cls],
            outputs=[annotated_out, summary_out, metrics_out],
        )

    with gr.Tab("History & Stats"):
        range_filter = gr.Dropdown(["today", "7d", "all"], value="7d", label="Range")
        refresh_btn = gr.Button("Refresh")
        history_table = gr.Dataframe(label="Recent requests", interactive=False)
        stats_json = gr.JSON(label="Summary")
        excel_btn = gr.Button("Download Excel")
        pdf_btn = gr.Button("Download PDF")
        excel_file = gr.File(label="Excel")
        pdf_file = gr.File(label="PDF")

        refresh_btn.click(load_history, inputs=[range_filter], outputs=[history_table, stats_json])
        excel_btn.click(download_excel, inputs=[range_filter], outputs=[excel_file])
        pdf_btn.click(download_pdf, inputs=[range_filter], outputs=[pdf_file])

    demo.load(load_history, inputs=[range_filter], outputs=[history_table, stats_json])


if __name__ == "__main__":
    demo.launch()

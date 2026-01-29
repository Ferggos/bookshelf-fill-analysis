from pathlib import Path

import pandas as pd
from fpdf import FPDF

from .storage import fetch_requests, summary_stats


def generate_excel(db_path: str | Path, output_path: str | Path, range_filter: str = "all") -> str:
    output_path = Path(output_path)
    rows = fetch_requests(db_path, limit=None, range_filter=range_filter)
    df = pd.DataFrame(rows)
    summary = summary_stats(db_path, range_filter=range_filter)
    summary_df = pd.DataFrame([summary])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Requests")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    return str(output_path)


def _add_table(pdf: FPDF, rows: list[dict]):
    headers = ["id", "ts", "input_type", "overall_book_count"]
    col_widths = [12, 55, 30, 35]
    pdf.set_font("Helvetica", size=9)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()

    for row in rows:
        values = [
            str(row.get("id", "")),
            str(row.get("ts", "")),
            str(row.get("input_type", "")),
            str(row.get("overall_book_count", "")),
        ]
        for v, w in zip(values, col_widths):
            pdf.cell(w, 6, v[:20], border=1)
        pdf.ln()


def generate_pdf(db_path: str | Path, output_path: str | Path, range_filter: str = "all") -> str:
    output_path = Path(output_path)
    rows = fetch_requests(db_path, limit=30, range_filter=range_filter)
    summary = summary_stats(db_path, range_filter=range_filter)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "Bookshelf Fill Report", ln=True)

    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Count: {summary.get('count', 0)}", ln=True)
    pdf.cell(0, 8, f"Avg books: {summary.get('avg_books', 0)}", ln=True)
    pdf.cell(0, 8, f"Avg processing: {summary.get('avg_ms', 0)} ms", ln=True)
    pdf.ln(4)

    _add_table(pdf, rows)
    pdf.ln(4)

    image_paths = []
    for row in rows:
        path = row.get("output_path")
        if path and Path(path).suffix.lower() in {".jpg", ".jpeg", ".png"} and Path(path).exists():
            image_paths.append(path)
        if len(image_paths) >= 3:
            break

    for img in image_paths:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 8, f"Example: {Path(img).name}", ln=True)
        pdf.image(img, w=180)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    return str(output_path)

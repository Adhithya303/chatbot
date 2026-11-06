from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "processed" / "corpus.json"

CHAPTER_PDF = DATA_DIR / "echap12.pdf"
CHAPTER_URL = "https://www.indiabudget.gov.in/economicsurvey/doc/eschapter/echap12.pdf"
CHAPTER_TITLE = "Chapter 12: Employment and Skill Development: Getting Skilling Right"

TABLE_CONFIGS: dict[str, dict[str, Any]] = {
    "8.10": {
        "file": "tab8.10.pdf",
        "path": DATA_DIR / "tab8.10.pdf",
        "url": "https://www.indiabudget.gov.in/economicsurvey/doc/stat/tab8.10.pdf",
        "title": "Labour Force Participation Rate",
        "metric": "Labour Force Participation Rate",
        "abbreviation": "LFPR",
        "unit": "percent",
    },
    "8.12": {
        "file": "tab8.12.pdf",
        "path": DATA_DIR / "tab8.12.pdf",
        "url": "https://www.indiabudget.gov.in/economicsurvey/doc/stat/tab8.12.pdf",
        "title": "Unemployment Rate",
        "metric": "Unemployment Rate",
        "abbreviation": "UR",
        "unit": "percent",
    },
}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\ufffd", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def slugify(value: str) -> str:
    slug = value.lower().replace("&", "and")
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def extract_chapter_chunks(pdf_path: Path = CHAPTER_PDF) -> list[dict[str, Any]]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Chapter PDF not found: {pdf_path}")

    chunks: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != 45:
            raise ValueError(
                f"Expected 45 pages in {pdf_path.name}, found {len(pdf.pages)}"
            )

        for page_idx, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(x_tolerance=2, y_tolerance=3)
            if not words:
                raise ValueError(f"No readable text extracted on page {page_idx} of {pdf_path.name}")

            # Group words into horizontal lines by top coordinate
            lines: list[dict[str, Any]] = []
            for w in sorted(words, key=lambda item: (round(item["top"], 1), item["x0"])):
                if not lines or abs(w["top"] - lines[-1]["top"]) > 3.5:
                    lines.append(
                        {
                            "top": w["top"],
                            "bottom": w["bottom"],
                            "words": [w],
                        }
                    )
                else:
                    lines[-1]["words"].append(w)
                    lines[-1]["bottom"] = max(lines[-1]["bottom"], w["bottom"])

            printed_page: int | None = 466 + page_idx
            filtered_lines: list[dict[str, Any]] = []
            for ln in lines:
                line_words = sorted(ln["words"], key=lambda item: item["x0"])
                line_text = normalize_whitespace(" ".join(w["text"] for w in line_words))
                if not line_text:
                    continue
                # Detect running headers and page numbers
                if line_text in {"Economic Survey 2025-26", "Employment and Skill Development"}:
                    continue
                if re.fullmatch(r"\d{3}(?:\s+\d{3})?", line_text):
                    printed_page = int(line_text.split()[0])
                    continue
                ln["text"] = line_text
                filtered_lines.append(ln)

            # Group lines into paragraph-sized blocks within the single PDF page
            raw_paragraphs: list[str] = []
            current_lines: list[str] = []
            prev_bottom: float | None = None

            for ln in filtered_lines:
                txt = ln["text"]
                gap = (ln["top"] - prev_bottom) if prev_bottom is not None else 0.0
                starts_new_section = bool(
                    re.match(
                        r"^(?:12\.\d+\.?|Box\s+XII\.\s*\d+|Chart\s+XII\.\s*\d+|Table\s+XII\.\s*\d+|INTRODUCTION|EMPLOYMENT OVERVIEW|CATALYSING JOB GROWTH|SKILL ECOSYSTEM OVERVIEW|SKILLING STRATEGIES FOR YOUTH|OUTLOOK)\b",
                        txt,
                    )
                )
                is_large_gap = prev_bottom is not None and gap > 6.5

                if current_lines and (starts_new_section or is_large_gap):
                    para_text = normalize_whitespace(" ".join(current_lines))
                    if para_text:
                        raw_paragraphs.append(para_text)
                    current_lines = [txt]
                else:
                    current_lines.append(txt)
                prev_bottom = ln["bottom"]

            if current_lines:
                para_text = normalize_whitespace(" ".join(current_lines))
                if para_text:
                    raw_paragraphs.append(para_text)

            # Merge short headings into following paragraph and cap oversized paragraphs
            merged_paragraphs: list[str] = []
            buffer = ""
            for para in raw_paragraphs:
                if not buffer:
                    buffer = para
                elif len(buffer) < 90 or (len(buffer) + len(para) < 380 and not re.match(r"^12\.\d+", para)):
                    buffer = f"{buffer} {para}"
                else:
                    merged_paragraphs.append(buffer)
                    buffer = para
            if buffer:
                merged_paragraphs.append(buffer)

            page_chunks: list[str] = []
            for para in merged_paragraphs:
                if len(para) <= 1100:
                    page_chunks.append(para)
                else:
                    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", para)
                    cur = ""
                    for sent in sentences:
                        if cur and len(cur) + len(sent) + 1 > 900:
                            page_chunks.append(cur.strip())
                            cur = sent
                        else:
                            cur = f"{cur} {sent}".strip() if cur else sent
                    if cur:
                        page_chunks.append(cur.strip())

            chunk_counter = 0
            for chunk_text in page_chunks:
                cleaned = normalize_whitespace(chunk_text)
                if not cleaned or cleaned == "***":
                    continue
                chunk_counter += 1
                chunk_id = f"chapter-12-p{page_idx:02d}-c{chunk_counter:02d}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "kind": "chapter_chunk",
                        "text": cleaned,
                        "printed_page": printed_page,
                        "source": {
                            "title": CHAPTER_TITLE,
                            "file": pdf_path.name,
                            "pdf_page": page_idx,
                            "url": CHAPTER_URL,
                        },
                    }
                )

    if not chunks:
        raise ValueError(f"No chapter chunks extracted from {pdf_path.name}")
    return chunks


def _try_pdfplumber_tables(page: pdfplumber.page.Page) -> list[tuple[str, float, float]] | None:
    """Attempt native pdfplumber table extraction first; return None if rows are merged or invalid."""
    tables = page.extract_tables()
    if not tables:
        return None
    parsed_rows: list[tuple[str, float, float]] = []
    for table in tables:
        for row in table:
            if not row or len(row) < 3:
                continue
            cells = [normalize_whitespace(c or "") for c in row if c and normalize_whitespace(c)]
            if len(cells) != 3:
                continue
            geo = cells[0]
            if geo in {"States/UTs", "(1)"} or "\n" in (row[0] or ""):
                return None
            try:
                q2 = float(cells[1])
                q3 = float(cells[2])
            except ValueError:
                continue
            parsed_rows.append((geo, q2, q3))
    if len(parsed_rows) != 23:
        return None
    return parsed_rows


def _extract_table_8_10_by_words(page: pdfplumber.page.Page) -> tuple[list[tuple[str, float, float]], dict[str, Any]]:
    words = page.extract_words()
    # Group horizontal lines by top coordinate
    by_top: dict[float, list[dict[str, Any]]] = {}
    for w in words:
        matched_top = None
        for existing_top in by_top:
            if abs(w["top"] - existing_top) <= 2.5:
                matched_top = existing_top
                break
        if matched_top is None:
            by_top[w["top"]] = [w]
        else:
            by_top[matched_top].append(w)

    sorted_tops = sorted(by_top.keys())
    header_lines: list[str] = []
    source_note = ""
    definition = ""
    rows: list[tuple[str, float, float]] = []
    period_labels: list[str] = []

    for top in sorted_tops:
        line_words = sorted(by_top[top], key=lambda item: item["x0"])
        line_text = normalize_whitespace(" ".join(w["text"] for w in line_words))
        if top < 175:
            header_lines.append(line_text)
            if "April-June" in line_text or "July-Sept" in line_text:
                period_labels = [w["text"] for w in line_words]
            continue

        if line_text.startswith("Source:"):
            source_note = normalize_whitespace(line_text.removeprefix("Source:"))
            continue
        if line_text.startswith("Note:"):
            definition = normalize_whitespace(line_text.removeprefix("Note:")).rstrip(".") + "."
            continue

        if 175 <= top <= 540:
            state_words = [w["text"] for w in line_words if w["x0"] < 350]
            q2_words = [w["text"] for w in line_words if 350 <= w["x0"] < 455]
            q3_words = [w["text"] for w in line_words if w["x0"] >= 455]
            if not state_words and not q2_words and not q3_words:
                continue
            geography = normalize_whitespace(" ".join(state_words))
            if not geography or len(q2_words) != 1 or len(q3_words) != 1:
                raise ValueError(
                    f"Failed to reconstruct row in tab8.10.pdf at y={top:.1f}: {line_text}"
                )
            try:
                q2_val = float(q2_words[0])
                q3_val = float(q3_words[0])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in tab8.10.pdf for '{geography}': {line_text}"
                ) from exc
            rows.append((geography, q2_val, q3_val))

    if period_labels != ["April-June", "July-Sept"]:
        raise ValueError(
            f"Unexpected period header order in tab8.10.pdf: {period_labels}"
        )

    header_blob = " ".join(header_lines)
    age_match = re.search(r"For persons,\s*([^)]+)", header_blob, re.IGNORECASE)
    basis_match = re.search(r"according to\s+(current weekly status)", header_blob, re.IGNORECASE)

    meta_details = {
        "full_title": "Table 8.10: Labour Force Participation Rate (LFPR) (For persons, 15 years & above) (in per cent) according to current weekly status for different States",
        "age_group": f"For persons, {age_match.group(1).strip()}" if age_match else "For persons, 15 years & above",
        "status_basis": basis_match.group(1).strip() if basis_match else "current weekly status",
        "source_note": source_note,
        "definition": definition,
        "period_headers": period_labels,
    }
    return rows, meta_details


def _extract_table_8_12_by_words(page: pdfplumber.page.Page) -> tuple[list[tuple[str, float, float]], dict[str, Any]]:
    words = page.extract_words(char_dir_rotated="btt")

    rotated_words = [w for w in words if not w.get("upright", True)]
    by_x0: dict[float, list[dict[str, Any]]] = {}
    for w in rotated_words:
        matched_x = None
        for existing_x in by_x0:
            if abs(w["x0"] - existing_x) <= 3.0:
                matched_x = existing_x
                break
        if matched_x is None:
            by_x0[w["x0"]] = [w]
        else:
            by_x0[matched_x].append(w)

    sorted_xs = sorted(by_x0.keys())
    header_lines: list[str] = []
    source_note = ""
    definition = ""
    rows: list[tuple[str, float, float]] = []
    period_labels: list[str] = []

    for x0 in sorted_xs:
        # Rotated text reads bottom-to-top (decreasing bottom/top coordinate)
        col_words = sorted(by_x0[x0], key=lambda item: -item["bottom"])
        line_text = normalize_whitespace(" ".join(w["text"] for w in col_words))

        if x0 < 155:
            header_lines.append(line_text)
            if "Apr-Jun" in line_text or "July-Sept" in line_text:
                period_labels = [w["text"] for w in col_words]
            continue

        if line_text.startswith("Source:"):
            source_note = normalize_whitespace(line_text.removeprefix("Source:"))
            continue
        if line_text.startswith("Note:"):
            definition = normalize_whitespace(line_text.removeprefix("Note:")).rstrip(".") + "."
            continue

        if 155 <= x0 <= 515:
            state_words = [w["text"] for w in col_words if w["top"] > 400]
            q2_words = [w["text"] for w in col_words if 125 <= w["top"] <= 180]
            q3_words = [w["text"] for w in col_words if 75 <= w["top"] < 125]
            if not state_words and not q2_words and not q3_words:
                continue
            geography = normalize_whitespace(" ".join(state_words))
            if not geography or len(q2_words) != 1 or len(q3_words) != 1:
                raise ValueError(
                    f"Failed to reconstruct row in tab8.12.pdf at x={x0:.1f}: {line_text}"
                )
            try:
                q2_val = float(q2_words[0])
                q3_val = float(q3_words[0])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in tab8.12.pdf for '{geography}': {line_text}"
                ) from exc
            rows.append((geography, q2_val, q3_val))

    if period_labels != ["Apr-Jun", "July-Sept"]:
        raise ValueError(
            f"Unexpected period header order in tab8.12.pdf: {period_labels}"
        )

    header_blob = " ".join(header_lines)
    age_match = re.search(r"For persons,\s*([^)]+)", header_blob, re.IGNORECASE)
    basis_match = re.search(r"according to\s+(current weekly status)", header_blob, re.IGNORECASE)

    meta_details = {
        "full_title": "Table 8.12: Unemployment Rate (UR) (in per cent) according to current weekly status for different States (For persons, 15 years & above)",
        "age_group": f"For persons, {age_match.group(1).strip()}" if age_match else "For persons, 15 years & above",
        "status_basis": basis_match.group(1).strip() if basis_match else "current weekly status",
        "source_note": source_note,
        "definition": definition,
        "period_headers": period_labels,
    }
    return rows, meta_details


def extract_table_records(
    table_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if table_id not in TABLE_CONFIGS:
        raise ValueError(f"Unknown table_id: {table_id}")

    cfg = TABLE_CONFIGS[table_id]
    pdf_path: Path = cfg["path"]
    if not pdf_path.exists():
        raise FileNotFoundError(f"Table PDF not found: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            raise ValueError(f"Empty PDF file: {pdf_path.name}")
        page = pdf.pages[0]

        # Step 1: Try native pdfplumber table extraction first
        native_rows = _try_pdfplumber_tables(page)

        # Step 2: Fallback to coordinate-based word extraction when native table merges rows
        if table_id == "8.10":
            word_rows, meta_details = _extract_table_8_10_by_words(page)
        else:
            word_rows, meta_details = _extract_table_8_12_by_words(page)

        rows = native_rows if native_rows is not None else word_rows

    if len(rows) != 23:
        raise ValueError(
            f"Table {table_id} ({pdf_path.name}) extracted {len(rows)} rows; expected 23."
        )

    seen_geographies: set[str] = set()
    records: list[dict[str, Any]] = []
    metric_lower = cfg["metric"].lower()
    abbr = cfg["abbreviation"]

    for geography, q2_val, q3_val in rows:
        if geography in seen_geographies:
            raise ValueError(
                f"Duplicate geography '{geography}' in Table {table_id} ({pdf_path.name})"
            )
        seen_geographies.add(geography)

        rec_id = f"table-{table_id}-{slugify(geography)}"
        text = (
            f"Table {table_id}. {geography} {metric_lower} ({abbr}): "
            f"April\u2013June 2025 {q2_val:.1f}%; July\u2013September 2025 {q3_val:.1f}%."
        )
        records.append(
            {
                "id": rec_id,
                "kind": "table_row",
                "table_id": table_id,
                "metric": cfg["metric"],
                "abbreviation": abbr,
                "geography": geography,
                "values": {
                    "2025-Q2": q2_val,
                    "2025-Q3": q3_val,
                },
                "unit": cfg["unit"],
                "text": text,
                "source": {
                    "title": cfg["title"],
                    "file": cfg["file"],
                    "pdf_page": 1,
                    "url": cfg["url"],
                },
            }
        )

    if "All-India" not in seen_geographies:
        raise ValueError(f"Table {table_id} ({pdf_path.name}) is missing 'All-India' row.")

    metadata_record = {
        "table_id": table_id,
        "title": cfg["title"],
        "full_title": meta_details["full_title"],
        "metric": cfg["metric"],
        "abbreviation": abbr,
        "unit": cfg["unit"],
        "age_group": meta_details["age_group"],
        "status_basis": meta_details["status_basis"],
        "definition": meta_details["definition"],
        "source_note": meta_details["source_note"],
        "periods": {
            "2025-Q2": "April\u2013June 2025",
            "2025-Q3": "July\u2013September 2025",
        },
        "period_order": ["2025-Q2", "2025-Q3"],
        "row_count": len(records),
        "source": {
            "title": cfg["title"],
            "file": cfg["file"],
            "pdf_page": 1,
            "url": cfg["url"],
        },
    }
    return records, metadata_record


def validate_corpus(
    chapter_chunks: list[dict[str, Any]],
    table_8_10_rows: list[dict[str, Any]],
    table_8_12_rows: list[dict[str, Any]],
    table_metadata: dict[str, dict[str, Any]],
) -> None:
    if not chapter_chunks:
        raise ValueError("Validation failed: chapter_chunks is empty.")

    pages_seen = {c["source"]["pdf_page"] for c in chapter_chunks}
    if min(pages_seen) != 1 or max(pages_seen) != 45 or len(pages_seen) != 45:
        raise ValueError(
            f"Validation failed: chapter chunks do not span all 45 PDF pages (found {len(pages_seen)} pages)."
        )

    for tid, rows in [("8.10", table_8_10_rows), ("8.12", table_8_12_rows)]:
        if len(rows) != 23:
            raise ValueError(
                f"Validation failed: Table {tid} has {len(rows)} rows, expected 23."
            )
        geos = {r["geography"] for r in rows}
        if len(geos) != 23 or "All-India" not in geos:
            raise ValueError(
                f"Validation failed: Table {tid} does not have 23 distinct geographies including All-India."
            )
        meta = table_metadata.get(tid)
        if not meta or meta.get("period_order") != ["2025-Q2", "2025-Q3"]:
            raise ValueError(f"Validation failed: Table {tid} period order is invalid.")
        if not meta.get("definition") or not meta.get("source_note"):
            raise ValueError(f"Validation failed: Table {tid} metadata missing definition or source_note.")

    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for r in table_8_10_rows + table_8_12_rows:
        vals = r["values"]
        if list(vals.keys()) != ["2025-Q2", "2025-Q3"]:
            raise ValueError(
                f"Validation failed: Row {r['id']} period keys are not ['2025-Q2', '2025-Q3']."
            )
        lookup[(r["table_id"], r["geography"])] = (vals["2025-Q2"], vals["2025-Q3"])

    expected_checks = {
        ("8.10", "Tamil Nadu"): (58.7, 58.6),
        ("8.10", "All-India"): (55.0, 55.1),
        ("8.12", "Tamil Nadu"): (6.9, 5.7),
        ("8.12", "All-India"): (5.4, 5.2),
    }
    for (tid, geo), expected_pair in expected_checks.items():
        actual_pair = lookup.get((tid, geo))
        if actual_pair != expected_pair:
            raise ValueError(
                f"Validation failed for Table {tid} {geo}: expected {expected_pair}, got {actual_pair}"
            )


def build_corpus() -> dict[str, Any]:
    chapter_chunks = extract_chapter_chunks(CHAPTER_PDF)
    table_8_10_rows, meta_8_10 = extract_table_records("8.10")
    table_8_12_rows, meta_8_12 = extract_table_records("8.12")

    table_metadata = {
        "8.10": meta_8_10,
        "8.12": meta_8_12,
    }

    validate_corpus(chapter_chunks, table_8_10_rows, table_8_12_rows, table_metadata)

    all_records = chapter_chunks + table_8_10_rows + table_8_12_rows
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus_payload = {
        "schema_version": "1.0",
        "records": all_records,
        "table_metadata": table_metadata,
    }
    CORPUS_PATH.write_text(
        json.dumps(corpus_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Chapter chunks: {len(chapter_chunks)}")
    print(f"Table 8.10 rows: {len(table_8_10_rows)}")
    print(f"Table 8.12 rows: {len(table_8_12_rows)}")
    print(f"Output path: {CORPUS_PATH}")
    return corpus_payload


def main() -> None:
    try:
        build_corpus()
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

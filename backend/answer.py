from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.retrieval import HybridRetriever, get_retriever

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


class LLMUnavailableError(RuntimeError):
    """Raised when Gemini generation is required but unavailable or not configured."""


OTHER_INDIAN_STATES_UTS = [
    "arunachal pradesh",
    "goa",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "sikkim",
    "tripura",
    "andaman and nicobar",
    "chandigarh",
    "dadra and nagar haveli",
    "daman and diu",
    "ladakh",
    "lakshadweep",
    "puducherry",
    "pondicherry",
]

METRIC_PATTERNS: dict[str, list[str]] = {
    "8.10": [
        r"\blabour\s+force\s+participation\s+rate\b",
        r"\blabor\s+force\s+participation\s+rate\b",
        r"\blabour\s+force\s+participation\b",
        r"\blabor\s+force\s+participation\b",
        r"\blfpr\b",
        r"\btable\s*8\.10\b",
    ],
    "8.12": [
        r"\bunemployment\s+rate\b",
        r"\bunemployment\b",
        r"\bjobless\s+rate\b",
        r"\bur\b",
        r"\btable\s*8\.12\b",
    ],
}

Q2_PATTERNS = [
    r"\bapril\s*[-\u2013\u2014to]+\s*june\b",
    r"\bapr\s*[-\u2013\u2014to]+\s*jun\b",
    r"\b2025[-\s]*q2\b",
    r"\bq2\s*2025\b",
    r"\bsecond\s+quarter\s+of\s+2025\b",
]

Q3_PATTERNS = [
    r"\bjuly\s*[-\u2013\u2014to]+\s*sept(?:ember)?\b",
    r"\bjul\s*[-\u2013\u2014to]+\s*sep(?:t|tember)?\b",
    r"\b2025[-\s]*q3\b",
    r"\bq3\s*2025\b",
    r"\bthird\s+quarter\s+of\s+2025\b",
]

INVALID_QUARTER_PATTERNS = [
    r"\bjan(?:uary)?\s*[-\u2013\u2014to]+\s*mar(?:ch)?\b",
    r"\boct(?:ober)?\s*[-\u2013\u2014to]+\s*dec(?:ember)?\b",
    r"\b2025[-\s]*q1\b",
    r"\bq1\s*2025\b",
    r"\b2025[-\s]*q4\b",
    r"\bq4\s*2025\b",
]

COMPARISON_CUES = [
    r"\bchange\b",
    r"\bchanged\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bbetween\b",
    r"\bdifference\b",
    r"\btrend\b",
    r"\bboth\s+quarters\b",
    r"\bacross\s+quarters\b",
    r"\bfrom\s+april",
    r"\bfrom\s+apr",
]

DEFINITION_CUES = [
    r"\bwhat\s+does\s+(?:it|ur|lfpr|the\s+metric|that|this)\s+measure\b",
    r"\bwhat\s+(?:is|are)\s+(?:ur|lfpr)\s+defined\b",
    r"\bhow\s+is\s+(?:it|ur|lfpr)\s+defined\b",
    r"\bdefine\b",
    r"\bdefinition\b",
    r"\bwhat\s+does\s+.*\s+measure\b",
    r"\bmeaning\s+of\b",
]


def format_source(
    record: dict[str, Any],
    table_id_override: str | None = None,
) -> dict[str, Any]:
    src = record["source"]
    table_id = table_id_override or record.get("table_id")
    out: dict[str, Any] = {
        "record_id": record["id"],
        "title": src["title"],
        "url": src["url"],
        "pdf_page": int(src["pdf_page"]),
    }
    if table_id is not None:
        out["table_id"] = str(table_id)
    return out


def _detect_metrics(q_lower: str) -> list[str]:
    matched: list[str] = []
    for tid, patterns in METRIC_PATTERNS.items():
        if any(re.search(p, q_lower) for p in patterns):
            matched.append(tid)
    return matched


def _detect_geographies(
    q_lower: str, retriever: HybridRetriever
) -> tuple[list[str], list[str]]:
    known_geos: list[str] = sorted(
        {
            r["geography"]
            for r in retriever.records
            if r.get("kind") == "table_row"
        },
        key=len,
        reverse=True,
    )
    matched_known: list[str] = []
    q_norm = re.sub(r"['\u2019]s\b", "", q_lower)
    q_norm = q_norm.replace("&", " and ")
    q_norm = re.sub(r"\s+", " ", q_norm)

    for geo in known_geos:
        geo_lower = geo.lower()
        geo_alt = geo_lower.replace("&", " and ").replace("-", " ")
        geo_alt = re.sub(r"\s+", " ", geo_alt)
        if (
            re.search(rf"\b{re.escape(geo_lower)}\b", q_lower)
            or re.search(rf"\b{re.escape(geo_alt)}\b", q_norm)
        ):
            matched_known.append(geo)

    if "All-India" not in matched_known and re.search(
        r"\b(?:all[-\s]+india|national\s+average|india\s+overall)\b", q_lower
    ):
        matched_known.append("All-India")

    matched_unavailable: list[str] = []
    for other_state in OTHER_INDIAN_STATES_UTS:
        if re.search(rf"\b{re.escape(other_state)}\b", q_norm):
            matched_unavailable.append(other_state.title())

    return matched_known, matched_unavailable


def _detect_years(q_lower: str) -> tuple[bool, list[str]]:
    """Return (has_incompatible_year, incompatible_years_list)."""
    # Remove table numbers like 8.10 / 8.12 before scanning years
    cleaned = re.sub(r"\b(?:8\.10|8\.12|12\.\d+)\b", "", q_lower)
    # Remove range 2025-26 / 2025–26
    cleaned = re.sub(r"\b2025\s*[-\u2013\u2014/]\s*(?:20)?26\b", "2025", cleaned)
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", cleaned)
    bad_years = [y for y in years if y != "2025"]
    return (len(bad_years) > 0, bad_years)


def _try_deterministic_numeric_answer(
    question: str, retriever: HybridRetriever
) -> dict[str, Any] | None:
    q_lower = question.lower()
    matched_metrics = _detect_metrics(q_lower)
    matched_geos, unavailable_geos = _detect_geographies(q_lower, retriever)

    has_q2 = any(re.search(p, q_lower) for p in Q2_PATTERNS)
    has_q3 = any(re.search(p, q_lower) for p in Q3_PATTERNS)
    has_invalid_quarter = any(re.search(p, q_lower) for p in INVALID_QUARTER_PATTERNS)
    has_comparison_cue = any(re.search(p, q_lower) for p in COMPARISON_CUES)
    has_definition_cue = any(re.search(p, q_lower) for p in DEFINITION_CUES)
    has_bad_year, bad_years = _detect_years(q_lower)

    # Determine if this is a state/table numeric lookup query
    is_table_query = bool(
        matched_geos
        or unavailable_geos
        or re.search(r"\btable\s*8\.(?:10|12)\b", q_lower)
        or (matched_metrics and (has_q2 or has_q3 or has_invalid_quarter or has_bad_year))
    )

    if not is_table_query:
        return None

    # 1. Check metric ambiguity or absence
    if len(matched_metrics) == 0:
        return {
            "status": "insufficient_evidence",
            "answer": (
                "Please clarify which metric you are asking about: "
                "Labour Force Participation Rate (LFPR, Table 8.10) or "
                "Unemployment Rate (UR, Table 8.12)."
            ),
            "sources": [],
        }
    if len(matched_metrics) > 1:
        return {
            "status": "insufficient_evidence",
            "answer": (
                "Your question mentions multiple metrics (both LFPR and Unemployment Rate). "
                "Please specify a single metric to look up or compare."
            ),
            "sources": [],
        }

    table_id = matched_metrics[0]
    table_meta = retriever.table_metadata[table_id]
    metric_name = table_meta["metric"]
    abbr = table_meta["abbreviation"]

    # 2. Check unavailable or missing geography
    if unavailable_geos and not matched_geos:
        state_str = ", ".join(unavailable_geos)
        return {
            "status": "insufficient_evidence",
            "answer": (
                f"Table {table_id} ({metric_name}) in the Economic Survey 2025\u201326 Statistical Appendix "
                f"does not contain a row for {state_str}. It covers 22 States/UTs and All-India."
            ),
            "sources": [],
        }
    if len(matched_geos) == 0:
        return {
            "status": "insufficient_evidence",
            "answer": (
                f"Please specify a State/UT or All-India to look up {metric_name} ({abbr}) in Table {table_id}."
            ),
            "sources": [],
        }
    if len(matched_geos) > 1:
        return {
            "status": "insufficient_evidence",
            "answer": (
                f"Multiple geographies were detected ({', '.join(matched_geos)}). "
                f"Please ask about one State/UT or All-India at a time."
            ),
            "sources": [],
        }

    geography = matched_geos[0]

    # 3. Check requested year / quarter validity
    if has_bad_year:
        bad_year_str = ", ".join(sorted(set(bad_years)))
        return {
            "status": "insufficient_evidence",
            "answer": (
                f"Insufficient evidence: Table {table_id} only reports {metric_name} ({abbr}) "
                f"for April\u2013June 2025 and July\u2013September 2025. Data for {bad_year_str} is not available in the table."
            ),
            "sources": [],
        }

    if has_invalid_quarter:
        return {
            "status": "insufficient_evidence",
            "answer": (
                f"Insufficient evidence: Table {table_id} only reports {metric_name} ({abbr}) "
                f"for April\u2013June 2025 (2025-Q2) and July\u2013September 2025 (2025-Q3)."
            ),
            "sources": [],
        }

    # Resolve requested quarters
    if (has_q2 and has_q3) or (has_comparison_cue and not (has_q2 ^ has_q3)):
        requested_quarters = ["2025-Q2", "2025-Q3"]
    elif has_q2 and not has_q3 and not has_comparison_cue:
        requested_quarters = ["2025-Q2"]
    elif has_q3 and not has_q2 and not has_comparison_cue:
        requested_quarters = ["2025-Q3"]
    else:
        return {
            "status": "insufficient_evidence",
            "answer": (
                f"Please specify which period in 2025 you want for {geography}'s {metric_name} ({abbr}): "
                f"April\u2013June 2025, July\u2013September 2025, or a comparison between both quarters."
            ),
            "sources": [],
        }

    # Find exact structured row record
    target_record: dict[str, Any] | None = None
    for rec in retriever.records:
        if (
            rec.get("kind") == "table_row"
            and rec.get("table_id") == table_id
            and rec.get("geography") == geography
        ):
            target_record = rec
            break

    if target_record is None:
        return {
            "status": "insufficient_evidence",
            "answer": f"No row found for {geography} in Table {table_id}.",
            "sources": [],
        }

    period_labels = table_meta["periods"]
    row_values = target_record["values"]

    if len(requested_quarters) == 1:
        q_key = requested_quarters[0]
        p_label = period_labels[q_key]
        val = float(row_values[q_key])
        answer_text = (
            f"According to Table {table_id}, {geography}'s {metric_name.lower()} ({abbr}) "
            f"in {p_label} was {val:.1f}%."
        )
        comparison = {
            "metric": metric_name,
            "abbreviation": abbr,
            "table_id": table_id,
            "geography": geography,
            "periods": [p_label],
            "period_keys": [q_key],
            "values": {p_label: val, q_key: val},
            "unit": target_record["unit"],
            "change_pp": None,
            "change_percentage_points": None,
        }
    else:
        q2_val = float(row_values["2025-Q2"])
        q3_val = float(row_values["2025-Q3"])
        # Second quarter minus first in Python
        diff_pp = round(q3_val - q2_val, 2)
        abs_diff = abs(diff_pp)

        if diff_pp < 0:
            change_phrase = f"a decrease of {abs_diff:.1f} percentage points ({diff_pp:+.1f} pp)"
            direction = "decrease"
        elif diff_pp > 0:
            change_phrase = f"an increase of {abs_diff:.1f} percentage points ({diff_pp:+.1f} pp)"
            direction = "increase"
        else:
            change_phrase = "no change (0.0 percentage points)"
            direction = "unchanged"

        p_q2 = period_labels["2025-Q2"]
        p_q3 = period_labels["2025-Q3"]
        answer_text = (
            f"According to Table {table_id}, {geography}'s {metric_name.lower()} ({abbr}) "
            f"was {q2_val:.1f}% in {p_q2} and {q3_val:.1f}% in {p_q3}, "
            f"representing {change_phrase}."
        )
        comparison = {
            "metric": metric_name,
            "abbreviation": abbr,
            "table_id": table_id,
            "geography": geography,
            "periods": [p_q2, p_q3],
            "period_keys": ["2025-Q2", "2025-Q3"],
            "values": {
                p_q2: q2_val,
                p_q3: q3_val,
                "2025-Q2": q2_val,
                "2025-Q3": q3_val,
            },
            "unit": target_record["unit"],
            "change_pp": diff_pp,
            "change_percentage_points": diff_pp,
            "direction": direction,
        }

    if has_definition_cue:
        answer_text += (
            f" Note (Table {table_id}): {table_meta['definition']} "
            f"Figures are reported {table_meta['age_group'].lower()} according to "
            f"{table_meta['status_basis']} ({table_meta['source_note']})."
        )

    return {
        "status": "answered",
        "answer": answer_text,
        "sources": [format_source(target_record)],
        "comparison": comparison,
    }


OUT_OF_SCOPE_MESSAGE = (
    "I answer questions about employment, unemployment, and labour force participation "
    "using India\u2019s Economic Survey. Try asking, \u201cWhat was Tamil Nadu\u2019s "
    "unemployment rate in July\u2013September 2025?\u201d"
)

DOMAIN_SCOPE_PATTERNS = [
    r"\b(?:un)?employ\w*\b",
    r"\bjob\w*\b",
    r"\bwork\w*\b",
    r"\blabou?r\w*\b",
    r"\blfpr\b",
    r"\bflfpr\b",
    r"\bur\b",
    r"\bcws\b",
    r"\bplfs\b",
    r"\bwage\w*\b",
    r"\bsalar\w*\b",
    r"\bcasual\b",
    r"\bearning\w*\b",
    r"\bhir(?:e|es|ed|ing)\b",
    r"\bretrench\w*\b",
    r"\blayoff\w*\b",
    r"\bvacanc\w*\b",
    r"\bcareer\w*\b",
    r"\brecruit\w*\b",
    r"\bstaffing\b",
    r"\b(?:in)?formal\w*\b",
    r"\b(?:un)?organi[sz]ed\b",
    r"\bgig\b",
    r"\bplatform\b",
    r"\bfreelanc\w*\b",
    r"\bmigrant\w*\b",
    r"\bcaregiv\w*\b",
    r"\bunpaid\b",
    r"\btime\s+use\b",
    r"\btus\b",
    r"\bpink\s+tax\b",
    r"\bhostel\w*\b",
    r"\bthozhi\b",
    r"\bsakhi\s+niwas\b",
    r"\bcreche\w*\b",
    r"\bmaternity\b",
    r"\bfactor(?:y|ies)\b",
    r"\bmanufactur\w*\b",
    r"\basi\b",
    r"\bqbuse\b",
    r"\basuse\b",
    r"\bmsme\w*\b",
    r"\bsme\w*\b",
    r"\btpe\b",
    r"\bnva\b",
    r"\bemolument\w*\b",
    r"\bproductiv\w*\b",
    r"\bbuddy\b",
    r"\bewa\b",
    r"\bearned\s+wage\b",
    r"\b(?:up|re)?skill\w*\b",
    r"\bvocational\b",
    r"\btvet\b",
    r"\btrain\w*\b",
    r"\bapprentice\w*\b",
    r"\bitis?\b",
    r"\bncrf\b",
    r"\bnats\b",
    r"\bnaps\b",
    r"\bpmkvy\b",
    r"\bvishwakarma\b",
    r"\bsidh\b",
    r"\bncs\b",
    r"\be[-\s]?shram\b",
    r"\buan\b",
    r"\bepfo\b",
    r"\besic\b",
    r"\bnsdc\b",
    r"\bmsde\b",
    r"\bssc\b",
    r"\bvoucher\w*\b",
    r"\bscorecard\b",
    r"\bschool[-\s]+to[-\s]+work\b",
    r"\bstw\b",
    r"\bstem\b",
    r"\bkaushal\b",
    r"\bddu[-\s]?gky\b",
    r"\bosda\b",
    r"\bstars\b",
    r"\bspark\b",
    r"\bparakh\b",
    r"\bswayam\b",
    r"\bapaar\b",
    r"\bsocial\s+security\b",
    r"\bindustrial\s+relations\b",
    r"\boccupational\s+safety\b",
    r"\beconomic\s+survey\b",
    r"\bchapter\s*12\b",
    r"\btable\s*8\.(?:10|12)\b",
    r"\bparticipation\s+rate\b",
]


def _has_domain_scope_signal(q_lower: str) -> bool:
    return any(re.search(pattern, q_lower) for pattern in DOMAIN_SCOPE_PATTERNS)


def _is_query_relevant_to_corpus(
    question: str, retrieved: list[dict[str, Any]]
) -> bool:
    if not retrieved:
        return False

    q_lower = question.lower()
    # Reject queries explicitly asking about incompatible historical years not in corpus
    has_bad_year, bad_years = _detect_years(q_lower)
    if has_bad_year:
        # Check if any retrieved text actually mentions the requested year
        joined_text = " ".join(r["text"] for r in retrieved)
        if not any(re.search(rf"\b{re.escape(y)}\b", joined_text) for y in bad_years):
            return False

    # Require domain relevance to employment, labour force, skilling, or Chapter 12 / Tables 8.10 & 8.12
    if not _has_domain_scope_signal(q_lower):
        return False

    has_any_bm25 = any(
        (r.get("retrieval", {}).get("bm25_score") or 0.0) > 0.0
        for r in retrieved
    )
    max_cosine = max(
        (r.get("retrieval", {}).get("cosine_similarity") or 0.0)
        for r in retrieved
    )
    max_bm25 = max(
        (r.get("retrieval", {}).get("bm25_score") or 0.0)
        for r in retrieved
    )

    # If no BM25 token matched at all, or similarity is weak, reject as unrelated
    if not has_any_bm25 and max_cosine < 0.68:
        return False
    if max_cosine < 0.60 and max_bm25 < 6.0:
        return False
    return True


def _build_grounded_prompt(
    question: str,
    retrieved: list[dict[str, Any]],
    table_metadata: dict[str, dict[str, Any]],
) -> str:
    evidence_blocks: list[str] = []
    referenced_tables: set[str] = set()

    for rec in retrieved:
        rid = rec["id"]
        src = rec["source"]
        tid = rec.get("table_id")
        if tid:
            referenced_tables.add(str(tid))
        header = f"[ID: {rid}] (Source: {src['title']}, PDF page {src['pdf_page']})"
        evidence_blocks.append(f"{header}\n{rec['text']}")

    q_lower = question.lower()
    if "lfpr" in q_lower or "labour force" in q_lower or "8.10" in q_lower:
        referenced_tables.add("8.10")
    if "unemployment" in q_lower or re.search(r"\bur\b", q_lower) or "8.12" in q_lower:
        referenced_tables.add("8.12")

    notes_blocks: list[str] = []
    for tid in sorted(referenced_tables):
        meta = table_metadata.get(tid)
        if meta:
            notes_blocks.append(
                f"[ID: table-{tid}-note] (Source: {meta['title']}, Table {tid}, PDF page 1)\n"
                f"Full title: {meta['full_title']} | "
                f"Age group: {meta['age_group']} | Basis: {meta['status_basis']} | "
                f"Definition: {meta['definition']} | Source note: {meta['source_note']}"
            )

    evidence_section = "\n\n".join(evidence_blocks)
    notes_section = "\n".join(notes_blocks) if notes_blocks else "None"

    return (
        "You are a grounded research assistant for the Economic Survey 2025-26 "
        "(Chapter 12: Employment and Skill Development, and Statistical Appendix Tables 8.10 and 8.12).\n"
        "Answer the user's question ONLY using the provided evidence passages and table notes below.\n"
        "Rules:\n"
        "1. Do NOT use outside knowledge or fabricate facts, numbers, or citations.\n"
        "2. If the provided evidence does not contain enough information to answer the question, "
        'set status to "insufficient_evidence", explain briefly what is missing, and return an empty list for cited_ids.\n'
        '3. If the evidence answers the question, set status to "answered", write a concise, accurate answer (2-4 sentences), '
        "and list ONLY the exact record IDs from the evidence that directly support your answer in cited_ids.\n"
        "4. Return valid JSON matching this schema:\n"
        '{"status": "answered" | "insufficient_evidence", "answer": "string", "cited_ids": ["record_id_1"]}\n\n'
        f"EVIDENCE PASSAGES:\n{evidence_section}\n\n"
        f"APPLICABLE TABLE NOTES:\n{notes_section}\n\n"
        f"USER QUESTION:\n{question}"
    )


def call_gemini_grounded(grounded_prompt: str) -> dict[str, Any]:
    load_dotenv(ENV_PATH, override=True)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    configured_model = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.5-flash-lite"

    if not api_key:
        raise LLMUnavailableError(
            "GEMINI_API_KEY is not configured in .env. Please set GEMINI_API_KEY and GEMINI_MODEL in .env to enable generative explanations."
        )

    # Prioritize fast, responsive Flash models first so requests complete in ~1-2 seconds
    candidate_models: list[str] = []
    for m in ("gemini-3.5-flash-lite", "gemini-3.5-flash", configured_model):
        if m and m not in candidate_models:
            candidate_models.append(m)

    client = genai.Client(api_key=api_key, http_options={"timeout": 10000})
    raw_text = ""
    last_exc: Exception | None = None

    for model_name in candidate_models:
        try:
            result = client.models.generate_content(
                model=model_name,
                contents=grounded_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            raw_text = (result.text or "").strip()
            if raw_text:
                break
        except Exception as exc:
            last_exc = exc
            continue
    else:
        raise LLMUnavailableError(
            "The language model service is currently unavailable. Please verify your GEMINI_API_KEY and GEMINI_MODEL configuration."
        ) from last_exc

    if not raw_text:
        raise LLMUnavailableError("The language model returned an empty response.")

    # Strip optional markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMUnavailableError("The language model returned malformed JSON output.") from exc

    if not isinstance(parsed, dict):
        raise LLMUnavailableError("The language model output is not a JSON object.")

    return parsed


def answer_question(
    question: str,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    cleaned_q = question.strip()
    if not cleaned_q:
        raise ValueError("Question must not be blank.")

    active_retriever = retriever or get_retriever()

    # 1. Try deterministic numeric resolution first (single-quarter or two-quarter comparison)
    numeric_result = _try_deterministic_numeric_answer(cleaned_q, active_retriever)
    if numeric_result is not None:
        return numeric_result

    # 2. Hybrid retrieval (BM25 + FastEmbed + RRF, top 5)
    retrieved = active_retriever.search(cleaned_q, mode="hybrid", top_k=5)

    # 3. Guard against unrelated questions before calling LLM
    if not _is_query_relevant_to_corpus(cleaned_q, retrieved):
        return {
            "status": "insufficient_evidence",
            "answer": OUT_OF_SCOPE_MESSAGE,
            "sources": [],
        }

    # 4. Build grounded prompt and call Gemini
    grounded_prompt = _build_grounded_prompt(
        cleaned_q, retrieved, active_retriever.table_metadata
    )
    llm_payload = call_gemini_grounded(grounded_prompt)

    status = str(llm_payload.get("status", "")).strip()
    answer_text = str(llm_payload.get("answer", "")).strip()
    raw_cited_ids = llm_payload.get("cited_ids", [])

    if status not in {"answered", "insufficient_evidence"} or not answer_text:
        raise LLMUnavailableError("Invalid structured response from language model.")

    if not isinstance(raw_cited_ids, list):
        raw_cited_ids = []

    # Validate returned IDs strictly against the supplied context
    supplied_by_id: dict[str, dict[str, Any]] = {r["id"]: r for r in retrieved}
    for tid, meta in active_retriever.table_metadata.items():
        note_id = f"table-{tid}-note"
        supplied_by_id[note_id] = {
            "id": note_id,
            "table_id": tid,
            "source": meta["source"],
        }
    valid_cited_ids: list[str] = []
    for cid in raw_cited_ids:
        if isinstance(cid, str) and cid in supplied_by_id and cid not in valid_cited_ids:
            valid_cited_ids.append(cid)

    if status == "insufficient_evidence" or not valid_cited_ids:
        return {
            "status": "insufficient_evidence",
            "answer": (
                answer_text
                if status == "insufficient_evidence" and answer_text
                else "Insufficient evidence in the retrieved Economic Survey passages to answer this question."
            ),
            "sources": [],
        }

    sources = [format_source(supplied_by_id[cid]) for cid in valid_cited_ids]
    return {
        "status": "answered",
        "answer": answer_text,
        "sources": sources,
    }


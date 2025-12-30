from __future__ import annotations

from dataclasses import dataclass
import html
import json
import os
import re
from typing import Any


@dataclass(frozen=True)
class CommentarySection:
    section_id: str
    title: str
    html: str


def generate_commentary_sections(*, model: str, facts: dict) -> list[CommentarySection]:
    # If key not present, fall back deterministically (never block report generation).
    api_key = _get_api_key()
    if not api_key:
        return _deterministic_sections(
            facts,
            note="No Gemini API key found in GEMINI_API_KEY / GOOGLE_API_KEY / GOOGLE_GENAI_API_KEY; generated deterministic commentary.",
        )

    try:
        from google import genai  # type: ignore
    except Exception:
        return _deterministic_sections(
            facts,
            note="google-genai not available in the environment; generated deterministic commentary.",
        )

    client = genai.Client(api_key=api_key)

    sections_spec = _sections_spec(facts)
    out: list[CommentarySection] = []
    first_error: str | None = None

    for spec in sections_spec:
        section_id = spec["id"]
        title = spec["title"]
        payload = spec["facts"]

        prompt = _build_prompt(section_title=title, facts=payload)
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = getattr(resp, "text", None) or ""
        except Exception as e:
            if first_error is None:
                first_error = f"{type(e).__name__}: {str(e)}"
            text = ""

        cleaned = _render_commentary_text_to_html(text, section_title=title)
        if not cleaned:
            cleaned = _deterministic_section_html(title=title, facts=payload)

        out.append(CommentarySection(section_id=section_id, title=title, html=cleaned))

    if first_error is not None:
        # Surface a readable note for the user instead of silently falling back.
        return _deterministic_sections(
            facts,
            note=f"AI request failed; using deterministic commentary. Error: {first_error[:240]}",
        )

    return out


def deterministic_commentary_sections(*, facts: dict, note: str | None = None) -> list[CommentarySection]:
    return _deterministic_sections(facts, note=note)


def _get_api_key() -> str:
    """
    Support multiple env var names for convenience/backwards-compat.
    Precedence:
      - GEMINI_API_KEY (preferred)
      - GOOGLE_API_KEY (common user default)
      - GOOGLE_GENAI_API_KEY (alternate)
    """
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return ""


def _sections_spec(facts: dict) -> list[dict[str, Any]]:
    report_type = (facts or {}).get("report_type") or "competitive"

    if report_type == "brand-health":
        return [
            {"id": "executive_summary", "title": "Executive Summary", "facts": _facts_exec_summary(facts)},
            {"id": "brand_health", "title": "Brand Health Overview", "facts": _facts_brand_health(facts)},
            {"id": "drawdowns", "title": "Drawdowns", "facts": _facts_drawdowns(facts)},
            {"id": "seasonality", "title": "Seasonality", "facts": _facts_seasonality(facts)},
            {"id": "suggestions", "title": "Term Disambiguation (Suggestions)", "facts": _facts_suggestions(facts)},
            {"id": "related", "title": "Related Queries & Topics (Main Term)", "facts": _facts_related(facts)},
            {"id": "region", "title": "Interest by Region", "facts": _facts_region(facts)},
        ]

    if report_type == "category":
        return [
            {"id": "executive_summary", "title": "Executive Summary", "facts": _facts_exec_summary(facts)},
            {"id": "clusters", "title": "Cluster Definitions", "facts": _facts_category_clusters(facts)},
            {"id": "cluster_trends", "title": "Cluster Trends (Weekly)", "facts": _facts_category_trends(facts)},
            {"id": "cluster_share", "title": "Cluster Share (Proxy)", "facts": _facts_category_share(facts)},
            {"id": "cluster_yearly", "title": "Cluster Year-by-Year", "facts": _facts_category_yearly(facts)},
            {"id": "suggestions", "title": "Term Disambiguation (Suggestions)", "facts": _facts_suggestions(facts)},
            {"id": "related", "title": "Related Queries & Topics (Main Term)", "facts": _facts_related(facts)},
            {"id": "region", "title": "Interest by Region", "facts": _facts_region(facts)},
        ]

    # competitive (default)
    return [
        {"id": "executive_summary", "title": "Executive Summary", "facts": _facts_exec_summary(facts)},
        {"id": "rankings", "title": "Competitive Trends (Weekly)", "facts": _facts_competitive_trends(facts)},
        {"id": "yearly", "title": "Year-by-Year Dynamics", "facts": _facts_yearly(facts)},
        {"id": "seasonality", "title": "Seasonality", "facts": _facts_seasonality(facts)},
        {"id": "share", "title": "Share of Search (Proxy)", "facts": _facts_share(facts)},
        {"id": "similarity", "title": "Similarity", "facts": _facts_similarity(facts)},
        {"id": "suggestions", "title": "Term Disambiguation (Suggestions)", "facts": _facts_suggestions(facts)},
        {"id": "related", "title": "Related Queries & Topics (Main Term)", "facts": _facts_related(facts)},
        {"id": "region", "title": "Interest by Region", "facts": _facts_region(facts)},
    ]


def _build_prompt(*, section_title: str, facts: dict) -> str:
    # The model should *only* use facts provided; no causal claims.
    return (
        "You are writing a concise client-facing analytics commentary section for a Google Trends report.\n"
        f"Section title: {section_title}\n\n"
        "Rules:\n"
        "- Use ONLY the provided facts JSON. Do not invent or infer facts.\n"
        "- Do not make causal claims (no 'because').\n"
        "- If a fact is missing or null, say it is unavailable rather than guessing.\n"
        "- Include concrete numbers (%, ranks, years) from the facts.\n"
        "- If you mention Google Trends values, remind that they are relative (indexed), not absolute search volume.\n"
        "- Output PLAIN TEXT ONLY (no HTML, no Markdown). Use short paragraphs.\n"
        "- For bullets, use lines starting with '- '.\n"
        "- Keep it under ~8 sentences total.\n\n"
        "Facts JSON:\n"
        f"{json.dumps(facts, indent=2, sort_keys=True)}\n"
    )

def _render_commentary_text_to_html(text: str, *, section_title: str | None = None) -> str:
    """
    Convert model plain-text output into safe HTML.
    Supports paragraphs, '-' bullet lists, and inline `code` spans.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # Safety: if the model accidentally returns HTML, treat it as plain text.
    if "<" in t and ">" in t:
        t = t.replace("<", "&lt;").replace(">", "&gt;")

    lines = [ln.rstrip() for ln in t.splitlines()]
    if section_title:
        # Many models echo the section title as the first line; drop it.
        for j, ln in enumerate(lines):
            s = ln.strip()
            if not s:
                continue
            if s.lower() == section_title.strip().lower():
                lines = lines[j + 1 :]
            break

    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.startswith("- "):
            items: list[str] = []
            while i < len(lines):
                l = lines[i].strip()
                if not l:
                    break
                if not l.startswith("- "):
                    break
                items.append(_render_inline_code(html.escape(l[2:].strip())))
                i += 1
            if items:
                lis = "\n".join(f"<li>{it}</li>" for it in items)
                blocks.append(f"<ul>\n{lis}\n</ul>")
            continue

        # paragraph: gather until blank line or bullet start
        para_lines: list[str] = []
        while i < len(lines):
            l = lines[i].strip()
            if not l:
                break
            if l.startswith("- "):
                break
            para_lines.append(l)
            i += 1
        para_text = " ".join(para_lines)
        blocks.append(f"<p>{_render_inline_code(html.escape(para_text))}</p>")

    return "\n".join(blocks).strip()


_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def _render_inline_code(s: str) -> str:
    # `code` -> <code>code</code>
    return _INLINE_CODE_RE.sub(lambda m: f"<code>{html.escape(m.group(1))}</code>", s)


def _deterministic_sections(facts: dict, note: str | None = None) -> list[CommentarySection]:
    specs = _sections_spec(facts)
    out: list[CommentarySection] = []
    for s in specs:
        title = s["title"]
        payload = s["facts"]
        html_snip = _deterministic_section_html(title=title, facts=payload)
        if note and s["id"] == "executive_summary":
            html_snip = f"<p><code>{html.escape(note)}</code></p>" + html_snip
        out.append(CommentarySection(section_id=s["id"], title=title, html=html_snip))
    return out


def _deterministic_section_html(*, title: str, facts: dict) -> str:
    # Lightweight fallback; meant to be replaced by AI output when available.
    return f"<p><b>{html.escape(title)}:</b> Generated from computed metrics.</p>"


def _facts_exec_summary(facts: dict) -> dict:
    return {
        "params": facts.get("params"),
        "main_term": facts.get("main_term"),
        "overall": facts.get("overall"),
        "rank_changes": facts.get("rank_changes"),
        "leaders_by_year": facts.get("leaders_by_year"),
        "main_term_flags": facts.get("main_term_flags"),
    }


def _facts_rankings(facts: dict) -> dict:
    years = facts.get("years", [])
    start_year = years[0] if years else None
    end_year = years[-1] if years else None
    return {
        "start_year": start_year,
        "end_year": end_year,
        "leaders_by_year": facts.get("leaders_by_year"),
        "rank_changes": facts.get("rank_changes"),
    }


def _facts_competitive_trends(facts: dict) -> dict:
    years = facts.get("years", [])
    return {
        "years": years,
        "leaders_by_year": facts.get("leaders_by_year"),
        "rank_changes": facts.get("rank_changes"),
        "competitive": facts.get("competitive"),
    }


def _facts_yearly(facts: dict) -> dict:
    return {
        "years": facts.get("years"),
        "overall": facts.get("overall"),
        "yoy_extrema": facts.get("yoy_extrema"),
    }


def _facts_seasonality(facts: dict) -> dict:
    return {
        "seasonality_peaks": facts.get("seasonality_peaks"),
        "params": facts.get("params"),
        "main_term": facts.get("main_term"),
    }


def _facts_share(facts: dict) -> dict:
    return {
        "note": "Share-of-search is computed as term / sum(terms) per week within this comparison set.",
        "terms": facts.get("terms"),
        "years": facts.get("years"),
    }


def _facts_related(facts: dict) -> dict:
    return {
        "availability": facts.get("data_availability"),
        "related_excerpts": facts.get("related_excerpts"),
        "main_term": facts.get("main_term"),
    }


def _facts_suggestions(facts: dict) -> dict:
    return {
        "availability": facts.get("data_availability"),
        "suggestions_summary": facts.get("suggestions_summary"),
    }


def _facts_region(facts: dict) -> dict:
    return {
        "availability": facts.get("data_availability"),
        "region_top": facts.get("region_top"),
    }


def _facts_similarity(facts: dict) -> dict:
    return {
        "note": "Similarity is correlation of weekly series; +1 means move together, -1 means move opposite.",
        "terms": facts.get("terms"),
        "main_term": facts.get("main_term"),
        "competitive": facts.get("competitive"),
    }


def _facts_brand_health(facts: dict) -> dict:
    return {
        "main_term": facts.get("main_term"),
        "brand_health": facts.get("brand_health"),
        "main_term_flags": facts.get("main_term_flags"),
        "years": facts.get("years"),
    }


def _facts_drawdowns(facts: dict) -> dict:
    return {
        "main_term": facts.get("main_term"),
        "main_term_flags": facts.get("main_term_flags"),
        "brand_health": facts.get("brand_health"),
    }


def _facts_category_clusters(facts: dict) -> dict:
    return {"category": facts.get("category"), "availability": facts.get("data_availability")}


def _facts_category_trends(facts: dict) -> dict:
    return {"category": facts.get("category")}


def _facts_category_share(facts: dict) -> dict:
    return {"category": facts.get("category")}


def _facts_category_yearly(facts: dict) -> dict:
    return {"category": facts.get("category")}



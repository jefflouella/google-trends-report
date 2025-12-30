from __future__ import annotations

from gtrends_analyzer.ai import deterministic_commentary_sections


def _min_facts(report_type: str) -> dict:
    # Only the keys needed for section selection + deterministic rendering.
    return {
        "report_type": report_type,
        "params": {"geo": "US", "timeframe": "today 5-y", "gprop": ""},
        "main_term": "JR Cigars",
        "terms": ["JR Cigars", "Holts"],
        "years": [2024, 2025],
    }


def test_ai_sections_competitive_ids() -> None:
    secs = deterministic_commentary_sections(facts=_min_facts("competitive"))
    ids = [s.section_id for s in secs]
    assert ids == [
        "executive_summary",
        "rankings",
        "yearly",
        "seasonality",
        "share",
        "similarity",
        "suggestions",
        "related",
        "region",
    ]


def test_ai_sections_brand_health_ids() -> None:
    secs = deterministic_commentary_sections(facts=_min_facts("brand-health"))
    ids = [s.section_id for s in secs]
    assert ids == [
        "executive_summary",
        "brand_health",
        "drawdowns",
        "seasonality",
        "suggestions",
        "related",
        "region",
    ]


def test_ai_sections_category_ids() -> None:
    secs = deterministic_commentary_sections(facts=_min_facts("category"))
    ids = [s.section_id for s in secs]
    assert ids == [
        "executive_summary",
        "clusters",
        "cluster_trends",
        "cluster_share",
        "cluster_yearly",
        "suggestions",
        "related",
        "region",
    ]



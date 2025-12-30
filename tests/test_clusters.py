from __future__ import annotations

from gtrends_analyzer.metrics import parse_clusters


def test_parse_clusters_named() -> None:
    clusters = parse_clusters(["Retailers: JR Cigars, Holts, Cigars International"])
    assert "Retailers" in clusters
    assert clusters["Retailers"] == ["JR Cigars", "Holts", "Cigars International"]


def test_parse_clusters_duplicate_names_get_suffix() -> None:
    clusters = parse_clusters(["A: one", "A: two"])
    assert "A" in clusters
    assert any(k.startswith("A (") for k in clusters.keys())


def test_parse_clusters_no_colon_creates_single_term_cluster() -> None:
    clusters = parse_clusters(["Foo Bar"])
    assert list(clusters.values())[0] == ["Foo Bar"]



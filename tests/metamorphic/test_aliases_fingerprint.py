"""Alias normalisation, fingerprint de-duplication and declaration hints.

* Case/dash variants of one codec name must not create duplicate candidates.
* Encodings that decode to the *same* payload collapse into one top-level
  match (fingerprint de-duplication); distinct payloads stay separate.
* An embedded charset declaration steers priority but cannot override bytes
  that do not decode under the declared codec.
"""

from __future__ import annotations

import pytest

from charset_normalizer import from_bytes

from .byte_model import (
    LARGE_CORPORA,
    Corpus,
    build_fixture,
    declaration,
    independent_decode,
)
from .relations import assert_no_duplicate_alias_candidates


@pytest.mark.parametrize(
    "corpus",
    [c for c in LARGE_CORPORA if c.aliases],
    ids=lambda c: c.name,
)
def test_alias_case_and_dash_do_not_duplicate(corpus: Corpus) -> None:
    fixture = build_fixture(corpus)
    assert_no_duplicate_alias_candidates(
        fixture.data, (corpus.encoding, *corpus.aliases)
    )


def test_fingerprint_dedup_collapses_same_payload() -> None:
    # Pure ASCII bytes decode to the identical string under every ASCII
    # superset codec: one top-level result, alternatives folded into leaves.
    data = b"Just some plain English sentence repeated enough to matter. " * 30
    results = from_bytes(data, cp_isolation=["ascii", "utf_8", "cp1252", "iso8859_1"])
    assert len(results) == 1
    winner = results.best()
    assert winner is not None
    assert "ascii" in winner.could_be_from_charset
    # Distinct top-level results always carry distinct fingerprints.
    fingerprints = [match.fingerprint for match in results]
    assert len(set(fingerprints)) == len(fingerprints)
    assert independent_decode(data, winner.encoding) == data.decode("ascii")


def test_distinct_payloads_produce_distinct_top_level() -> None:
    corpus = next(c for c in LARGE_CORPORA if c.name == "cyrillic_cp1251")
    fixture = build_fixture(corpus)
    # koi8_r decodes the same cp1251 byte stream to different (mojibake)
    # text, so it survives as its own top-level result with a distinct
    # fingerprint instead of being folded into the cp1251 family.
    results = from_bytes(fixture.data, cp_isolation=["cp1251", "koi8_r"])
    top = [match.encoding for match in results]
    assert {"cp1251", "koi8_r"} <= set(top)
    winner = results["cp1251"]
    assert independent_decode(fixture.data, "cp1251") == str(winner)
    assert independent_decode(fixture.data, "koi8_r") != str(winner)
    fingerprints = [match.fingerprint for match in results]
    assert len(set(fingerprints)) == len(fingerprints)


def test_wrong_declaration_does_not_override_bytes() -> None:
    corpus = next(c for c in LARGE_CORPORA if c.name == "cyrillic_cp1251")
    fixture = build_fixture(
        corpus, declaration=declaration("ISO-8859-5", template_index=1)
    )
    results = from_bytes(fixture.data)
    winner = results.best()
    assert winner is not None
    # Bytes are cp1251; iso8859_5 decodes them to mojibake and must not win.
    assert frozenset(winner.could_be_from_charset) == corpus.family
    assert str(winner) == fixture.expected_text


def test_correct_declaration_keeps_winner() -> None:
    corpus = next(c for c in LARGE_CORPORA if c.name == "cyrillic_cp1251")
    fixture = build_fixture(
        corpus, declaration=declaration("WINDOWS-1251", template_index=0)
    )
    winner = from_bytes(fixture.data).best()
    assert winner is not None
    # A confirmed declaration wins on its own fast path with a single match.
    assert "cp1251" in winner.could_be_from_charset
    assert str(winner) == fixture.expected_text


def test_declaration_spelling_variants_equivalent() -> None:
    corpus = next(c for c in LARGE_CORPORA if c.name == "chinese_gb18030")
    winners = set()
    for alias in ("gb18030", "GB-18030", "Gb18030"):
        fixture = build_fixture(
            corpus, declaration=declaration(alias, template_index=2)
        )
        winner = from_bytes(fixture.data).best()
        assert winner is not None
        winners.add(winner.encoding)
        assert str(winner) == fixture.expected_text
    assert winners == {"gb18030"}

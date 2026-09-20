"""Sampling-layout metamorphic relations.

Changing chunk_size/steps or appending trailing ASCII whitespace must not
change the winner's payload or encoding family for losslessly decodable,
distinguishing inputs. Short or genuinely ambiguous inputs get only the
weak, soundness-level assertions.
"""

from __future__ import annotations

import os
import string

import pytest

from charset_normalizer import from_bytes

from .byte_model import (
    AMBIGUOUS_CORPORA,
    DEFAULT_LAYOUTS,
    HEAVY_LAYOUTS,
    LARGE_CORPORA,
    SHORT_CORPORA,
    WEAK_LARGE_CORPORA,
    Corpus,
    build_fixture,
    independent_decode,
)
from .relations import (
    assert_strong_invariance,
    assert_weak_invariance,
    candidate_families,
    detect,
    result_evidence,
)

HEAVY = os.environ.get("CHARSET_NORMALIZER_MT_HEAVY") == "1"
LAYOUTS = HEAVY_LAYOUTS if HEAVY else DEFAULT_LAYOUTS


@pytest.mark.parametrize("corpus", LARGE_CORPORA, ids=lambda c: c.name)
def test_winner_invariant_across_sampling_layouts(corpus: Corpus) -> None:
    assert_strong_invariance(build_fixture(corpus), LAYOUTS)


@pytest.mark.parametrize("corpus", WEAK_LARGE_CORPORA, ids=lambda c: c.name)
def test_long_but_contested_keeps_true_family(corpus: Corpus) -> None:
    # Winner may move; true family reachability + self-consistency must hold.
    assert_weak_invariance(build_fixture(corpus), LAYOUTS)


@pytest.mark.parametrize("corpus", SHORT_CORPORA, ids=lambda c: c.name)
def test_short_inputs_keep_correct_candidate(corpus: Corpus) -> None:
    # Short inputs: the heuristic may be unsure (winner may move), but the
    # true family must remain reachable and every result self-consistent.
    assert_weak_invariance(build_fixture(corpus), LAYOUTS)


@pytest.mark.parametrize("corpus", AMBIGUOUS_CORPORA, ids=lambda c: c.name)
def test_ambiguous_inputs_only_soundness(corpus: Corpus) -> None:
    fixture = build_fixture(corpus)
    for chunk_size, steps in LAYOUTS:
        results = from_bytes(fixture.data, chunk_size=chunk_size, steps=steps)
        assert results
        # No universal winner is asserted for pure ASCII. The bytes decode
        # identically under every reported codec, and ascii stays reachable.
        families = candidate_families(results)
        assert any("ascii" in family for family in families)
        for match in results:
            assert (
                independent_decode(fixture.data, match.encoding) == fixture.corpus.text
            )


@pytest.mark.parametrize(
    "suffix", [b"", b" ", b"\t\r\n ", b" " * 256], ids=["none", "sp", "mixed", "pad"]
)
def test_appending_ascii_whitespace_preserves_payload(suffix: bytes) -> None:
    corpus = next(c for c in LARGE_CORPORA if c.name == "cyrillic_cp1251")
    fixture = build_fixture(corpus)
    base = from_bytes(fixture.data).best()
    padded = from_bytes(fixture.data + suffix).best()
    assert base is not None and padded is not None
    assert str(padded) == str(base) + suffix.decode("ascii")
    assert frozenset(padded.could_be_from_charset) == corpus.family


def test_whitespace_is_ascii_only() -> None:
    # Guards the metamorphic premise: appended suffixes are pure ASCII.
    for char in string.whitespace:
        assert ord(char) < 0x80


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_newline_and_repetition_boundaries_preserve_winner(newline: str) -> None:
    # Repeating segments joined by different newline styles only relocates
    # byte offsets; the lossless payload and family must stay invariant.
    corpus = next(c for c in LARGE_CORPORA if c.name == "cyrillic_cp1251")
    fixture = build_fixture(corpus, repeat_segments=18, newline=newline)
    assert_strong_invariance(
        fixture, ((512, 5), (128, 8), (64, 12)), trailing_whitespace=()
    )


@pytest.mark.parametrize(
    "corpus_name", ["russian_utf8", "cyrillic_utf16", "cyrillic_utf32"]
)
def test_committed_single_result_has_stable_evidence_set(corpus_name: str) -> None:
    # The source commits to a single answer for these (clean UTF / BOM fast
    # paths), so the full evidence-bearing set is asserted equal across
    # layouts rather than just the winner.
    corpus = next(c for c in LARGE_CORPORA if c.name == corpus_name)
    fixture = build_fixture(corpus)
    reference = result_evidence(detect(fixture.data, LAYOUTS[0]))
    assert len(reference) == 1
    for layout in LAYOUTS[1:]:
        assert result_evidence(detect(fixture.data, layout)) == reference


@pytest.mark.parametrize("corpus", LARGE_CORPORA, ids=lambda c: c.name)
def test_independent_decoder_roundtrip(corpus: Corpus) -> None:
    # bytes->str->bytes anchored to stdlib codecs, never CharsetMatch.output.
    # Re-encoding the recovered text with the codec used to *generate* the
    # body reproduces exactly that body (BOM/header framing excluded).
    fixture = build_fixture(corpus)
    winner = detect(fixture.data, DEFAULT_LAYOUTS[0]).best()
    assert winner is not None
    decoded = independent_decode(fixture.data, winner.encoding, strip_bom=fixture.bom)
    assert decoded == fixture.expected_text

    generator_codec = {"utf_16": "utf_16_le", "utf_32": "utf_32_le"}.get(
        corpus.encoding, corpus.encoding
    )
    expected_body = fixture.expected_text.encode(generator_codec)
    assert decoded.encode(generator_codec, errors="strict") == expected_body

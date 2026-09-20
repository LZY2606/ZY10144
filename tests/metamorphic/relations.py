"""Metamorphic relation assertions shared by the test modules.

Two strengths of claim:

* :func:`assert_strong_invariance` — for losslessly decodable, sufficiently
  distinguishing inputs the winner, its decoded payload and its encoding
  family are invariant across sampling layouts and trailing ASCII
  whitespace.
* :func:`assert_weak_invariance` — for short/genuinely ambiguous inputs only
  soundness is required: evidence is self-consistent and the true encoding
  stays reachable across layouts (the winner may move).

All payload comparisons go through :mod:`tests.metamorphic.byte_model`
(independent stdlib codec), never through another ``CharsetMatch.output``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from charset_normalizer import CharsetMatch, CharsetMatches, from_bytes
from charset_normalizer.utils import iana_name

from .byte_model import (
    DEFAULT_LAYOUTS,
    TRAILING_WHITESPACE,
    Corpus,
    Fixture,
    independent_decode,
)


@dataclass(frozen=True)
class Evidence:
    """Layout-independent snapshot of a result's proven properties."""

    encoding: str
    fingerprint: int
    chaos: float
    has_bom: bool
    could_be: frozenset[str]


def candidate_families(results: CharsetMatches) -> set[frozenset[str]]:
    return {frozenset(match.could_be_from_charset) for match in results}


def result_evidence(results: CharsetMatches) -> set[Evidence]:
    return {
        Evidence(
            encoding=match.encoding,
            fingerprint=match.fingerprint,
            chaos=round(match.chaos, 6),
            has_bom=match.byte_order_mark,
            could_be=frozenset(match.could_be_from_charset),
        )
        for match in results
    }


def detect(data: bytes, layout: tuple[int, int], **kwargs) -> CharsetMatches:
    chunk_size, steps = layout
    return from_bytes(data, chunk_size=chunk_size, steps=steps, **kwargs)


def assert_evidence_self_consistent(
    results: CharsetMatches, data: bytes, *, expect_bom: bool | None = None
) -> None:
    """Every returned match must satisfy the bytes->str contract on its own.

    Independent verification: strictly decoding ``data`` with each match's
    own codec yields exactly ``str(match)`` (after BOM removal). This never
    uses ``CharsetMatch.output`` as the oracle.
    """
    assert results, "expected at least one plausible candidate"
    for match in results:
        assert 0.0 <= match.chaos <= 1.0
        assert 0.0 <= match.coherence <= 1.0
        if expect_bom is not None:
            assert match.byte_order_mark is expect_bom

        payload = str(match)
        # Independent oracle for the bytes->str relation.
        decoded = independent_decode(
            data,
            match.encoding,
            strip_bom=match.byte_order_mark,
        )
        assert decoded == payload, (
            f"match {match.encoding} str() disagrees with the stdlib codec"
        )
        # Every leaf alias must name the same decoded payload.
        leaf_payloads = {str(leaf) for leaf in match.submatch}
        assert leaf_payloads <= {payload}
        # The canonical name and every alias normalise consistently.
        names = [match.encoding, *match.could_be_from_charset]
        assert all(iana_name(name, strict=False) for name in names)


def assert_correct_candidate_present(results: CharsetMatches, corpus: Corpus) -> None:
    """Weak claim: the true encoding is reachable in the result set.

    Exact leaf grouping varies with sampling (the detector folds codecs that
    produce the same payload under a given run), so membership is checked by
    encoding name rather than by an exact family-set equality.
    """
    reachable = {name for family in candidate_families(results) for name in family}
    assert reachable & set(corpus.family), (
        f"true family {set(corpus.family)} missing from {sorted(reachable)}"
    )


def _winning_match(results: CharsetMatches) -> CharsetMatch:
    best = results.best()
    assert best is not None, "expected a winning candidate"
    return best


def assert_strong_invariance(
    fixture: Fixture,
    layouts: Sequence[tuple[int, int]] = DEFAULT_LAYOUTS,
    *,
    trailing_whitespace: Iterable[bytes] = TRAILING_WHITESPACE,
) -> None:
    """Winner payload + family are fixed across layouts and whitespace.

    Only the properties the source actually promises are held fixed: the
    winner's decoded payload and its encoding family. Measured chaos is
    layout dependent (chunks may cut words mid-sequence), so it is only
    bounded by the detector's own acceptance threshold, never pinned to a
    literal, and ordering is not asserted (the source promises tie-breaks
    only for a fixed evidence set).
    """

    def check(data: bytes, *, expect_text: str) -> None:
        for layout in layouts:
            results = detect(data, layout)
            assert_evidence_self_consistent(results, data, expect_bom=fixture.bom)
            winner = _winning_match(results)
            assert str(winner) == expect_text, (
                f"layout {layout} changed the decoded payload"
            )
            assert (
                independent_decode(data, winner.encoding, strip_bom=fixture.bom)
                == expect_text
            )
            assert frozenset(winner.could_be_from_charset) == fixture.family, (
                f"layout {layout} moved family to {set(winner.could_be_from_charset)}"
            )
            # Source guarantee: a non-fallback winner passed initial chaos
            # probing, hence its mean chaos is below the default threshold.
            assert winner.chaos < 0.2

    check(fixture.data, expect_text=fixture.expected_text)

    # Appending ASCII whitespace shifts offsets/stride but no encoded glyph.
    # Fixed-width Unicode codecs (UTF-16/32) are byte-aligned: a raw ASCII
    # byte would corrupt a code unit, so the transform does not apply there.
    if fixture.encoding not in {"utf_16", "utf_32"}:
        for whitespace in trailing_whitespace:
            check(
                fixture.data + whitespace,
                expect_text=fixture.expected_text + whitespace.decode("ascii"),
            )


def assert_weak_invariance(
    fixture: Fixture,
    layouts: Sequence[tuple[int, int]] = DEFAULT_LAYOUTS,
) -> None:
    """Candidate reachability + self-consistency; winner is free to move."""
    for layout in layouts:
        results = detect(fixture.data, layout)
        assert_evidence_self_consistent(results, fixture.data)
        assert_correct_candidate_present(results, fixture.corpus)


def assert_no_duplicate_alias_candidates(data: bytes, aliases: Sequence[str]) -> None:
    """Case/dash variants of one codec name must not spawn duplicate results."""
    canonical = iana_name(aliases[0], strict=False)
    seen: set[str] = set()
    for alias in aliases:
        results = from_bytes(data, cp_isolation=[alias])
        # Each spelling isolates exactly one canonical encoding.
        names = [name for match in results for name in match.could_be_from_charset]
        assert names, f"alias {alias!r} yielded no candidate"
        assert set(names) <= {canonical}
        assert iana_name(alias, strict=False) == canonical
        seen.update(names)
    assert seen == {canonical}

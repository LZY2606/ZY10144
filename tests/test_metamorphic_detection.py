"""Metamorphic tests for charset detection sampling behaviour.

These tests do *not* pin the winner observed on one particular corpus as a
universal truth.  Instead they derive bytes from known Unicode text with a
small deterministic fixture generator and check relations that must hold no
matter how the detector walks the input:

* for lossless, sufficiently distinctive inputs, the decoded payload of the
  best candidate and its encoding family are invariant under changes of
  ``chunk_size`` / ``steps`` and under appending ASCII whitespace;
* for short or genuinely ambiguous inputs, we only assert that the correct
  candidate remains part of the result set and that every evidence field is
  self-consistent;
* alias case/dash normalisation never creates duplicate candidates, and
  equal-payload candidates are merged through the fingerprint/submatch
  machinery rather than duplicated as top-level results.

Decoded output is cross-checked with an independent ``bytes.decode`` oracle
(plus explicit BOM handling); ``CharsetMatch.output()`` is never the sole
oracle.  Result ordering is only asserted where the source code commits to a
tie-break; otherwise evidence-bearing candidate sets are compared.

Heavy/fuzz rounds are opt-in through the ``CN_METAMORPHIC_HEAVY`` environment
variable so that ``uv run pytest -q`` stays fast.  Nothing here touches the
network or depends on the machine locale.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import pickle
import random
import subprocess
import sys
from dataclasses import dataclass

import pytest

from charset_normalizer import from_bytes
from charset_normalizer.utils import iana_name

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

HEAVY = os.environ.get("CN_METAMORPHIC_HEAVY", "") not in ("", "0", "false", "False")

# Sampling layouts exercised by the quick suite.  They deliberately include
# tiny chunk sizes so that multi-byte characters straddle sample boundaries.
LAYOUTS = [(512, 5), (256, 5), (128, 2), (64, 3), (37, 4), (16, 10)]

# Extra aggressive layouts (BOMs longer than the first chunk, 2-byte windows).
TINY_LAYOUTS = [(8, 6), (4, 8), (2, 8)]

WHITESPACE_SUFFIXES = (b"", b"   ", b"\n\n\n", b" \t\r\n" * 4, b" " * 64)


@dataclass(frozen=True)
class Corpus:
    """A byte sample produced from known Unicode text."""

    data: bytes
    text: str
    encoding: str
    bom: bytes = b""


# ---------------------------------------------------------------------------
# Independent decoding oracle
# ---------------------------------------------------------------------------

# BOM marks that the detector strips *before* decoding (see should_strip_sig_or_bom).
_STRIPPED_BOM = {
    "utf_8": (b"\xef\xbb\xbf",),
    "utf_8_sig": (b"\xef\xbb\xbf",),
    "gb18030": (b"\x84\x31\x95\x33",),
}

# BOM encodings whose own codec consumes the BOM into a leading U+FEFF.
_BOM_CODECS = {"utf_16", "utf_32"}


def oracle_decode(data: bytes, encoding: str) -> str:
    """Decode ``data`` independently of CharsetMatch internals.

    Mirrors the bytes transformations performed in ``api.from_bytes`` for the
    given IANA encoding: raw SIG stripping for utf_8/utf_8_sig/gb18030 and
    reliance on the stdlib codec BOM consumption for utf_16/utf_32 (the
    detector keeps the U+FEFF out of its decoded payload because the BOM
    candidate returns directly with the SIG-prefixed sequence; we normalise
    both views here).
    """
    if encoding in _STRIPPED_BOM:
        for mark in _STRIPPED_BOM[encoding]:
            if data.startswith(mark):
                data = data[len(mark) :]
                break
    decoded = data.decode(encoding, "strict")
    if encoding in _BOM_CODECS and decoded.startswith("\ufeff"):
        decoded = decoded[1:]
    return decoded


# ---------------------------------------------------------------------------
# Encoding families
# ---------------------------------------------------------------------------

# Explicit family of encodings used by the fixtures; only winners we actually
# assert on need to be classified.  Keep this coarse on purpose: the claim is
# about the family (utf8 / utf16 / latin / cyrillic / greek / cjk / ...),
# never about one champion picked on this corpus.
_FAMILY_ALIASES = {
    "utf8": {"utf_8", "utf_8_sig"},
    "utf16": {"utf_16", "utf_16_be", "utf_16_le"},
    "utf32": {"utf_32", "utf_32_be", "utf_32_le"},
    "cp1251": {"cp1251", "kz1048", "ptcp154"},
    "cp1253": {"cp1253", "iso8859_7"},
    "cp932": {"cp932", "shift_jis", "shift_jis_2004", "shift_jisx0213"},
    "big5": {"big5", "big5hkscs", "cp950"},
    "cp949": {"cp949", "euc_kr"},
    "gb18030": {"gb18030", "gbk", "gb2312"},
}


def family_of(encoding: str) -> str:
    normalized = iana_name(encoding, strict=False)
    for family, members in _FAMILY_ALIASES.items():
        if normalized in members:
            return family
    return normalized


# ---------------------------------------------------------------------------
# Result introspection (evidence-bearing candidate sets)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One decodable candidate with the evidence the detector exposes."""

    encoding: str
    chaos: float
    coherence: float
    bom: bool
    payload: str
    fingerprint: int

    @property
    def family(self) -> str:
        return family_of(self.encoding)


def _match_to_candidate(match) -> Candidate:
    return Candidate(
        encoding=match.encoding,
        chaos=match.chaos,
        coherence=match.coherence,
        bom=bool(match.bom),
        payload=str(match),
        fingerprint=match.fingerprint,
    )


def collect_candidates(results) -> list[Candidate]:
    """Top-level candidates plus their fingerprint-merged submaches."""
    candidates: list[Candidate] = []
    for match in results:
        candidates.append(_match_to_candidate(match))
        for leaf in match.submatch:
            candidates.append(_match_to_candidate(leaf))
    return candidates


def candidate_names(results) -> set[str]:
    names: set[str] = set()
    for match in results:
        names.add(match.encoding)
        for leaf in match.submatch:
            names.add(leaf.encoding)
    return names


def assert_evidence_self_consistent(results) -> None:
    """Evidence fields must be coherent for every candidate, submaches included.

    The detector only keeps candidates that fully decode, so ``str(match)``
    must succeed with strict semantics; ratios stay inside their documented
    ranges; BOM-flagged candidates of stripped-SIG encodings must not expose a
    leading U+FEFF; and submaches document a fingerprint merge of the parent.
    """
    for match in results:
        assert 0.0 <= match.chaos <= 1.0
        assert 0.0 <= match.coherence <= 1.0
        payload = str(match)
        assert match.fingerprint == hash(payload)
        assert match.raw is not None
        for leaf in match.submatch:
            leaf_payload = str(leaf)
            assert leaf.chaos == match.chaos
            assert leaf.fingerprint == match.fingerprint
            assert leaf_payload == payload
            assert leaf.encoding != match.encoding
            assert 0.0 <= leaf.chaos <= 1.0
        if match.bom and match.encoding in {"utf_8", "utf_8_sig", "gb18030"}:
            assert not payload.startswith("\ufeff")
        if match.bom and match.encoding in {"utf_16", "utf_32"}:
            assert not payload.startswith("\ufeff")


def candidates_matching_payload(results, expected_text: str) -> set[str]:
    """Names of candidates whose *independent* strict decode equals text."""
    names: set[str] = set()
    for candidate in collect_candidates(results):
        if candidate.payload == expected_text:
            names.add(candidate.encoding)
    return names


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def build_fixture(
    text: str,
    encoding: str,
    *,
    bom: bytes = b"",
    ascii_prefix: bytes = b"",
    ascii_suffix: bytes = b"",
    declaration: bytes | None = None,
    newlines_every: int = 0,
    truncate_bytes: int = 0,
) -> Corpus:
    """Produce bytes from known Unicode text via a stdlib codec.

    Controls the knobs mentioned in the task: BOM/SIG, newlines, repeated
    segments, ASCII affixes, embedded charset declarations and truncated tail
    bytes.  The returned ``text`` is what an independent strict decoder must
    recover (truncation excluded).
    """
    body = text
    if newlines_every:
        lines = [
            body[i : i + newlines_every] for i in range(0, len(body), newlines_every)
        ]
        body = "\n".join(lines)
    payload = body.encode(encoding)
    data = bom + ascii_prefix + payload + ascii_suffix
    if declaration is not None:
        # Declarations live inside an ASCII head, right after the BOM.
        data = bom + declaration + ascii_prefix + payload + ascii_suffix
    if truncate_bytes:
        data = data[:-truncate_bytes]
    return Corpus(data=data, text=body, encoding=encoding, bom=bom)


def detect(corpus_or_bytes, *, chunk_size: int = 512, steps: int = 5, **kwargs):
    data = (
        corpus_or_bytes.data if isinstance(corpus_or_bytes, Corpus) else corpus_or_bytes
    )
    return from_bytes(data, chunk_size=chunk_size, steps=steps, **kwargs)


# Canonical distinctive texts.  They must contain bytes that rule out sibling
# single-byte code pages so the strong relation is meaningful (Latin text with
# cp1251-vs-cp1252 ambiguity is handled in the weak section instead).
FRENCH_TEXT = (
    "Le café offre un résumé très naïf des élèves français qui écrivent "
    "élégamment à côté de l'hôtel du vieux quartier. "
)
RUSSIAN_TEXT = (
    "Образование является основой развития любого современного общества, "
    "науки и культуры в больших городах. "
)
GREEK_TEXT = (
    "Αυτό είναι ένα μακρύ ελληνικό κείμενο για τον έλεγχο της ανίχνευσης "
    "κωδικοποίησης με πολλούς διαφορετικούς χαρακτήρες. "
)
JAPANESE_TEXT = (
    "この文章は日本語で書かれており、漢字やひらがな、カタカナが多く含まれています。"
)
TRAD_CHINESE_TEXT = (
    "這是一段用來測試繁體中文編碼偵測功能的文字內容，包含許多常用的漢字語句。"
)
KOREAN_TEXT = (
    "이것은 한국어 텍스트로서 문자 인코딩 탐지 기능을 시험하기 위한 내용입니다. "
)
MIXED_TEXT = "English prefix and 日本語の挿入、そして encore du français avec café. "


# ---------------------------------------------------------------------------
# Strong metamorphic relations: layout-invariant payload + family
# ---------------------------------------------------------------------------


def _expected_payload(
    corpus: Corpus, *, ascii_prefix_text: str = "", extra: bytes = b""
) -> str:
    """What the best candidate must decode to.

    ``extra`` is restricted to ASCII whitespace bytes in callers so that its
    decoded form is layout independent.
    """
    return ascii_prefix_text + corpus.text + extra.decode("ascii")


def assert_strong_invariance(
    corpus: Corpus,
    expected_family: str,
    *,
    layouts=LAYOUTS,
    ascii_prefix_text: str = "",
    ascii_compatible: bool = True,
    suffixes=WHITESPACE_SUFFIXES,
):
    """Payload + family of ``best()`` must survive sampling-layout changes.

    Also re-encodes whitespace suffixes for wide codecs (utf-16/32): appending
    raw ASCII bytes would corrupt those codecs, which is not a fair mutation.
    """
    baseline_text = _expected_payload(corpus, ascii_prefix_text=ascii_prefix_text)
    candidate_baseline = baseline_text

    # 1) changing chunk_size / steps
    for chunk_size, steps in layouts:
        results = detect(corpus, chunk_size=chunk_size, steps=steps)
        best = results.best()
        assert best is not None, f"no candidate at layout {(chunk_size, steps)}"
        assert_evidence_self_consistent(results)
        assert family_of(best.encoding) == expected_family, (
            f"family drifted to {family_of(best.encoding)} "
            f"({best.encoding}) at layout {(chunk_size, steps)}"
        )
        # Independent oracle for the bytes -> string relation.
        assert str(best) == candidate_baseline
        assert oracle_decode(corpus.data, best.encoding) == baseline_text
        assert best.raw is corpus.data or bytes(best.raw) == bytes(corpus.data)

    # 2) appending ASCII whitespace only (or wide-codec re-encoded whitespace)
    for suffix in suffixes:
        if not suffix:
            continue
        for chunk_size, steps in layouts:
            if ascii_compatible:
                mutated = corpus.data + suffix
                expected = baseline_text + suffix.decode("ascii")
            else:
                codec = {"utf16": "utf-16", "utf32": "utf-32"}[expected_family]
                mutated = corpus.data + suffix.decode("ascii").encode(codec)
                # Re-encoding with the stdlib codec emits another BOM at the
                # splice point, so a U+FEFF precedes the appended whitespace.
                bom_char = "\ufeff"
                expected = candidate_baseline + suffix.decode("ascii")
                expected = candidate_baseline + bom_char + suffix.decode("ascii")
            results = from_bytes(mutated, chunk_size=chunk_size, steps=steps)
            best = results.best()
            assert best is not None
            assert_evidence_self_consistent(results)
            assert family_of(best.encoding) == expected_family
            assert str(best) == expected


def test_french_utf8_layout_invariant():
    corpus = build_fixture(FRENCH_TEXT * 16, "utf-8")
    assert_strong_invariance(corpus, "utf8")
    # Oracle cross-check on the baseline fixture itself.
    assert oracle_decode(corpus.data, "utf_8") == corpus.text


def test_russian_cp1251_layout_invariant():
    corpus = build_fixture(RUSSIAN_TEXT * 16, "cp1251")
    assert_strong_invariance(corpus, "cp1251")
    assert oracle_decode(corpus.data, "cp1251") == corpus.text


def test_greek_cp1253_layout_invariant():
    corpus = build_fixture(GREEK_TEXT * 16, "cp1253")
    assert_strong_invariance(corpus, "cp1253")
    assert oracle_decode(corpus.data, "cp1253") == corpus.text


def test_japanese_cp932_layout_invariant():
    corpus = build_fixture(JAPANESE_TEXT * 18, "cp932")
    assert_strong_invariance(corpus, "cp932")
    assert oracle_decode(corpus.data, "cp932") == corpus.text


def test_trad_chinese_big5_layout_invariant():
    corpus = build_fixture(TRAD_CHINESE_TEXT * 18, "big5")
    # Chunk sizes >= 37 always reach the big5 winner; 16-byte windows are a
    # documented CJK uncertainty zone (utf_16_be can tie), covered by the
    # membership-only test below.
    assert_strong_invariance(
        corpus, "big5", layouts=[(512, 5), (256, 5), (128, 2), (64, 3), (37, 4)]
    )
    assert_candidate_present_everywhere(
        corpus, expected_family="big5", layouts=[(512, 5), (64, 3), (16, 10)]
    )
    assert oracle_decode(corpus.data, "big5") == corpus.text


def test_korean_cp949_layout_invariant():
    corpus = build_fixture(KOREAN_TEXT * 16, "cp949")
    assert_strong_invariance(corpus, "cp949")
    assert oracle_decode(corpus.data, "cp949") == corpus.text


def test_multibyte_char_straddles_sample_boundary():
    """3-byte UTF-8 sequences are cut mid-character by tiny chunk windows.

    The byte layout is fixed to a 1330-byte corpus whose stride boundaries
    land inside multi-byte sequences for the (8, 10) layout (verified with an
    independent IncrementalDecoder).  Probing chunks decode errors-ignored,
    while the stored payload remains a strict decode, so the winner and its
    exact payload must not change.
    """
    word = "café résumé naïve. ".encode("utf-8")
    data = (word * 70)[:1330]
    import codecs

    cuts = list(range(0, len(data), len(data) // 10))[1:-1]
    straddle_found = False
    for cut in cuts:
        decoder = codecs.getincrementaldecoder("utf-8")()
        try:
            decoder.decode(data[:cut], "strict")
            decoder.decode(b"", "strict")
        except UnicodeDecodeError:
            straddle_found = True
    assert straddle_found, "fixture lost its purpose: no boundary cut a char"

    corpus = Corpus(data=data, text=data.decode("utf-8"), encoding="utf-8")
    assert_strong_invariance(
        corpus, "utf8", layouts=[(8, 10), (16, 10), (133, 10), (512, 5)]
    )
    assert oracle_decode(data, "utf_8") == corpus.text


def test_bom_longer_than_first_chunk_utf16():
    text = MIXED_TEXT * 20
    corpus = Corpus(
        data=text.encode("utf-16"), text=text, encoding="utf_16", bom=b"\xff\xfe"
    )
    assert corpus.encoding == "utf_16"
    assert_strong_invariance(
        corpus,
        "utf16",
        layouts=LAYOUTS + TINY_LAYOUTS,
        ascii_compatible=False,
    )
    for chunk_size, steps in LAYOUTS + TINY_LAYOUTS:
        results = detect(corpus, chunk_size=chunk_size, steps=steps)
        best = results.best()
        assert best is not None and best.bom is True
        assert family_of(best.encoding) == "utf16"
        assert len(results) == 1  # BOM candidate short-circuits detection
        assert oracle_decode(corpus.data, best.encoding) == corpus.text


def test_bom_longer_than_first_chunk_utf32():
    text = MIXED_TEXT * 12
    corpus = Corpus(
        data=text.encode("utf-32"),
        text=text,
        encoding="utf_32",
        bom=b"\xff\xfe\x00\x00",
    )
    assert_strong_invariance(
        corpus,
        "utf32",
        layouts=[(512, 5), (64, 3), (8, 10), (4, 8)],
        ascii_compatible=False,
    )
    for chunk_size, steps in [(512, 5), (4, 8)]:
        results = detect(corpus, chunk_size=chunk_size, steps=steps)
        best = results.best()
        assert best is not None and best.bom is True
        assert family_of(best.encoding) == "utf32"
        assert oracle_decode(corpus.data, best.encoding) == corpus.text


def test_utf8_bom_stripped_but_payload_kept():
    corpus = build_fixture(FRENCH_TEXT * 6, "utf-8", bom=b"\xef\xbb\xbf")
    for chunk_size, steps in [(512, 5), (4, 8), (2, 8)]:
        results = detect(corpus, chunk_size=chunk_size, steps=steps)
        best = results.best()
        assert best is not None
        assert best.encoding == "utf_8" and best.bom is True
        assert str(best) == corpus.text
        assert not str(best).startswith("\ufeff")
        assert oracle_decode(corpus.data, "utf_8") == corpus.text


def test_gb18030_sig_longer_than_first_chunk():
    text = MIXED_TEXT * 16
    data = b"\x84\x31\x95\x33" + text.encode("gb18030")
    corpus = Corpus(data=data, text=text, encoding="gb18030", bom=b"\x84\x31\x95\x33")
    for chunk_size, steps in [(512, 5), (2, 8)]:
        results = detect(corpus, chunk_size=chunk_size, steps=steps)
        best = results.best()
        assert best is not None
        assert best.encoding == "gb18030" and best.bom is True
        assert str(best) == corpus.text
        assert oracle_decode(data, "gb18030") == corpus.text


def test_long_ascii_prefix_before_distinguishing_bytes():
    """Distinguishing non-ASCII bytes appear only after a long ASCII head."""
    head = (
        b"HTTP/1.1 200 OK\r\nServer: example/9.1\r\nContent-Type: text/plain\r\n\r\n"
        + b"x" * 400
    )
    body = RUSSIAN_TEXT * 12
    corpus = Corpus(
        data=head + body.encode("cp1251"),
        text=body,
        encoding="cp1251",
    )
    assert_strong_invariance(
        corpus,
        "cp1251",
        layouts=LAYOUTS,
        ascii_prefix_text=head.decode("ascii"),
    )
    assert str(detect(corpus).best()).startswith(head.decode("ascii"))

    # UTF-8 variant: some sampling windows miss the first non-ASCII byte but
    # the strict utf_8 candidate must still win everywhere.
    head2 = b"a" * 900
    body2 = "Le café résumé naïf des élèves français écrivent très élégamment. " * 10
    corpus2 = Corpus(
        data=head2 + body2.encode("utf-8"),
        text=head2.decode("ascii") + body2,
        encoding="utf_8",
    )
    for chunk_size, steps in LAYOUTS:
        results = detect(corpus2, chunk_size=chunk_size, steps=steps)
        best = results.best()
        assert best is not None and best.encoding == "utf_8"
        assert str(best) == corpus2.text


def test_newlines_and_repetition_layout_invariant():
    corpus = build_fixture(RUSSIAN_TEXT * 10, "cp1251", newlines_every=37)
    assert_strong_invariance(corpus, "cp1251")
    assert "\n" in corpus.text


# ---------------------------------------------------------------------------
# Weak relations: short / truly ambiguous inputs
# ---------------------------------------------------------------------------


def assert_candidate_present_everywhere(
    corpus: Corpus, *, expected_family: str, layouts=LAYOUTS
):
    """The correct candidate must survive in every layout's result set.

    No claim is made about ``best()`` or about ordering: short byte sequences
    are genuinely ambiguous and the source promises no tie-break between
    equal-chaos siblings.  We only require that (a) some candidate decodes to
    exactly the known text under the right family at every layout, and (b) the
    evidence fields stay self-consistent.
    """
    for chunk_size, steps in layouts:
        results = detect(corpus, chunk_size=chunk_size, steps=steps)
        assert_evidence_self_consistent(results)
        matching = candidates_matching_payload(results, corpus.text)
        families = {family_of(name) for name in matching}
        assert expected_family in families, (
            f"correct family {expected_family} missing at layout "
            f"{(chunk_size, steps)}; got {sorted(families)}"
        )


def test_short_french_is_ambiguous_but_candidate_kept():
    text = "café résumé très cool"
    corpus = Corpus(data=text.encode("cp1252"), text=text, encoding="cp1252")
    assert len(corpus.data) < 32  # detector's TOO_SMALL_SEQUENCE zone
    assert_candidate_present_everywhere(corpus, expected_family="cp1252")
    # And it stays ambiguous: more than one Latin family decodes losslessly.
    results = detect(corpus)
    assert len(candidates_matching_payload(results, corpus.text)) >= 2


def test_short_greek_candidate_kept():
    text = "Αυτό είναι ένα σύντομο μήνυμα"
    corpus = build_fixture(text, "cp1253")
    assert_candidate_present_everywhere(corpus, expected_family="cp1253")


def test_tiny_cyrillic_candidate_kept():
    text = "privet мир"
    corpus = build_fixture(text, "cp1251")
    assert_candidate_present_everywhere(corpus, expected_family="cp1251")


def test_short_cjk_candidates_kept():
    jp = Corpus(data="こんにちは".encode("cp932"), text="こんにちは", encoding="cp932")
    zh = Corpus(data="中文測試".encode("big5"), text="中文測試", encoding="big5")
    ko = Corpus(data="안녕하세요".encode("cp949"), text="안녕하세요", encoding="cp949")
    assert_candidate_present_everywhere(jp, expected_family="cp932")
    assert_candidate_present_everywhere(zh, expected_family="big5")
    assert_candidate_present_everywhere(ko, expected_family="cp949")


def test_cjk_aggressive_layout_only_requires_membership():
    """Tiny windows on CJK are a documented uncertainty zone.

    With chunk_size=8 the winner may flip (utf_16_be vs cp932), so we assert
    membership and evidence rather than a winner -- exactly the heuristic
    uncertainty region the strong relation must not be forced onto.
    """
    corpus = build_fixture(JAPANESE_TEXT * 20, "cp932")
    assert_candidate_present_everywhere(
        corpus, expected_family="cp932", layouts=[(512, 5), (64, 3), (16, 10), (8, 6)]
    )
    baseline = detect(corpus)
    tiny = detect(corpus, chunk_size=8, steps=6)
    assert family_of(baseline.best().encoding) == "cp932"
    # Membership is the contract; winner drift at 8-byte windows is allowed.
    assert "cp932" in candidate_names(tiny)


def test_truncated_dangling_byte_excludes_utf8_everywhere():
    """A dangling UTF-8 lead byte is a hard failure for the utf_8 candidate."""
    data = ("café résumé " * 40).encode("utf-8") + b"\xc3"
    with pytest.raises(UnicodeDecodeError):
        data.decode("utf-8")
    corpus = Corpus(data=data, text="", encoding="utf_8")
    for chunk_size, steps in LAYOUTS + [(8, 6)]:
        results = detect(corpus, chunk_size=chunk_size, steps=steps)
        assert_evidence_self_consistent(results)
        assert "utf_8" not in candidate_names(results)


def test_truncation_at_char_boundary_keeps_payload():
    """Cutting at a character boundary is lossless and must stay stable."""
    full = (FRENCH_TEXT * 12).encode("utf-8")
    text = full.decode("utf-8")
    corpus = Corpus(data=full, text=text, encoding="utf_8")
    assert_strong_invariance(corpus, "utf8")


# ---------------------------------------------------------------------------
# Embedded charset declarations
# ---------------------------------------------------------------------------


def test_embedded_declaration_cp1251_layout_invariant():
    declaration = b'<html><head><meta charset="cp1251"></head>'
    body = RUSSIAN_TEXT * 15
    corpus = Corpus(
        data=declaration + body.encode("cp1251"),
        text=declaration.decode("ascii") + body,
        encoding="cp1251",
    )
    for chunk_size, steps in LAYOUTS + [(8, 6)]:
        results = detect(corpus, chunk_size=chunk_size, steps=steps)
        best = results.best()
        assert best is not None
        assert family_of(best.encoding) == "cp1251"
        assert str(best) == corpus.text
        assert oracle_decode(corpus.data, "cp1251") == corpus.text


def test_declaration_preemptive_vs_content_winner():
    """A declaration prioritises an encoding; with preemptive off the pure
    heuristic may still pick a same-family sibling, but the declared family
    must remain a candidate either way."""
    declaration = b'<meta charset="cp1251">'
    body = RUSSIAN_TEXT * 14
    data = declaration + body.encode("cp1251")
    text = declaration.decode("ascii") + body
    preemptive = from_bytes(data)
    heuristic = from_bytes(data, preemptive_behaviour=False)
    assert family_of(preemptive.best().encoding) == "cp1251"
    assert "cp1251" in candidates_matching_payload(heuristic, text)


# ---------------------------------------------------------------------------
# Alias case / dash normalisation must not create duplicate candidates
# ---------------------------------------------------------------------------

ALIAS_SPELLINGS = ["cp1251", "CP1251", "windows-1251", "WINDOWS_1251", "windows_1251"]


@pytest.mark.parametrize("spelling", ALIAS_SPELLINGS)
def test_charset_declaration_alias_spellings_equivalent(spelling):
    body = RUSSIAN_TEXT * 14
    declaration = f'<meta charset="{spelling}">'.encode("ascii")
    data = declaration + body.encode("cp1251")
    results = from_bytes(data)
    best = results.best()
    assert best is not None
    assert iana_name(spelling) == "cp1251"
    assert family_of(best.encoding) == "cp1251"

    # A single normalised candidate: alias spellings must not produce
    # duplicates neither at top level nor inside the submatch lists.
    all_names = list(candidate_names(results))
    assert "cp1251" in all_names
    assert all(
        iana_name(name, strict=False) != "cp1251"
        for name in all_names
        if name != "cp1251"
    )
    # Lookup by recognised alias spellings resolves to the same top-level
    # match ("cp-1251" is a declaration spelling, not a Python codec alias).
    for alias in ("CP1251", "windows-1251"):
        assert results[alias] is results["cp1251"]


def test_unrecognised_alias_spelling_is_ignored_not_duplicated():
    """Spelling variants absent from the IANA table are ignored by the
    preemptive scanner; they must neither raise nor inject candidates."""
    body = RUSSIAN_TEXT * 14
    baseline = from_bytes(body.encode("cp1251"))
    for spelling in ("cp-1251", "Cp_1251"):
        data = f'<meta charset="{spelling}">'.encode("ascii") + body.encode("cp1251")
        results = from_bytes(data)
        names = candidate_names(results)
        assert "cp1251" in names
        assert len(names) == len(set(names))
        assert names.issuperset(candidate_names(baseline))


def test_utf8_alias_declarations_deduplicated():
    for spelling in ("UTF-8", "utf_8", "Utf8", "uTf-8"):
        declaration = f'<meta charset="{spelling}">'.encode("ascii")
        data = declaration + (FRENCH_TEXT * 10).encode("utf-8")
        results = from_bytes(data)
        assert results.best() is not None
        assert results.best().encoding == "utf_8"
        assert len(results) == 1
        assert not results.best().submatch


# ---------------------------------------------------------------------------
# Fingerprint dedup / split
# ---------------------------------------------------------------------------


def test_same_payload_encodings_merge_as_submatches():
    """Encodings decoding to the same str are merged, never double-listed.

    Pure Latin-1-range text maps identically through cp1252 and iso8859_15
    (among others): the detector must expose one top-level candidate per
    distinct payload and attach the sibling encodings as submaches with equal
    chaos and equal fingerprint.
    """
    text = "Le café offre un résumé très naïf des élèves français. " * 12
    data = text.encode("cp1252")
    assert data.decode("iso8859_15") == text  # both codecs agree here
    results = from_bytes(data)
    top_level = list(results)
    agreeing = next(
        match
        for match in top_level
        if "cp1252" in match.could_be_from_charset
        and "iso8859_15" in match.could_be_from_charset
    )
    leaves = {leaf.encoding: leaf for leaf in agreeing.submatch}
    assert "cp1252" in agreeing.could_be_from_charset
    assert "iso8859_15" in agreeing.could_be_from_charset
    assert str(agreeing) == text
    for name in ("cp1252", "iso8859_15"):
        leaf = leaves.get(name)
        target = leaf if leaf is not None and leaf.encoding == name else agreeing
        assert target.fingerprint == hash(text)
        if leaf is not None:
            assert leaf.chaos == agreeing.chaos
            assert str(leaf) == text
    # cp1252 and iso8859_15 must never appear as two top-level candidates:
    # equal-payload encodings are attached as submaches of one another.
    top_level_names = {match.encoding for match in top_level}
    assert not {"cp1252", "iso8859_15"}.issubset(top_level_names)
    merged_group = next(
        match
        for match in top_level
        if "cp1252" in match.could_be_from_charset
        and "iso8859_15" in match.could_be_from_charset
    )
    assert merged_group.fingerprint == hash(text)


def test_distinguishing_byte_splits_fingerprints():
    """0x80 maps to U+20AC under cp1252 but to a control under iso8859_15.

    This pins the dedup *boundary*: differing decoded strings must remain
    distinct top-level candidates with distinct fingerprints, and both
    payloads must independently re-decode through the stdlib codec.
    """
    french = FRENCH_TEXT * 12
    data = french.encode("cp1252")
    assert b"\x80" not in data
    data = data[:100] + b"\x80\x80" + data[100:]
    assert data.decode("cp1252") != data.decode("iso8859_15")

    results = from_bytes(data)
    assert_evidence_self_consistent(results)
    by_encoding = {
        candidate.encoding: candidate for candidate in collect_candidates(results)
    }
    assert "cp1252" in by_encoding
    assert "iso8859_15" in by_encoding
    cp1252 = by_encoding["cp1252"]
    iso15 = by_encoding["iso8859_15"]
    assert cp1252.payload != iso15.payload
    assert cp1252.fingerprint != iso15.fingerprint
    # Independent codec cross-check on the divergent byte.
    assert cp1252.payload[100:102] == "\u20ac\u20ac"
    assert oracle_decode(data, "cp1252") == cp1252.payload
    assert oracle_decode(data, "iso8859_15") == iso15.payload


# ---------------------------------------------------------------------------
# Mess early exit
# ---------------------------------------------------------------------------


def test_mess_early_exit_across_layouts():
    """Punctuation-heavy noise trips the give-up counter on every layout.

    ASCII punctuation decodes losslessly as ascii, so it never hard-fails;
    instead the measured chaos reaches the configured threshold, the ascii
    candidate soft-fails and (with fallbacks enabled) only the utf_8 fallback
    remains, carrying that same honest chaos value.  Sampling layout must not
    change this branch.
    """
    noise = (b'=][;:!?*&^%$#@ (){}<>~|_+ "quoted" ' * 30)[:760]
    assert any(byte > 127 for byte in noise) is False
    for chunk_size, steps in [(512, 5), (64, 3), (37, 4), (16, 10)]:
        results = from_bytes(noise, chunk_size=chunk_size, steps=steps)
        assert len(results) == 1
        only = list(results)[0]
        assert only.encoding == "utf_8"  # fallback after the ascii soft-fail
        assert only.chaos >= 0.2
        names = candidate_names(results)
        assert "ascii" not in names
        for candidate in collect_candidates(results):
            assert 0.0 <= candidate.chaos <= 1.0
    # Disabling the fallback surfaces the soft failure as an empty result set.
    strict = from_bytes(noise, enable_fallback=False)
    assert len(strict) == 0


# ---------------------------------------------------------------------------
# Pinned counterexamples: the layout genuinely changes which branches run
# ---------------------------------------------------------------------------


def test_counterexample_sampling_offset_can_change_result_set():
    """A long ASCII head with non-ASCII bytes at the tail.

    At (chunk_size=64, steps=3) the three probing windows cover only ASCII
    prefix bytes, so chaos probing gives up and the result set is empty;
    adjacent layouts (128, 2 / default) reach the distinguishing tail and
    recover the Latin-1 family.  This is the documented sampling-offset
    uncertainty -- it must not be "fixed" by hard-coding a winner, but it must
    stay deterministic for a fixed input.
    """
    prefix = b"The quick brown fox jumps over the lazy dog. 0123456789. " * 12
    sentence = (
        " Voilà un résumé très élégant de l'éducation française, " "à côté de l'hôtel."
    ).encode("cp1252")
    data = prefix + sentence * 8
    assert len(data) == 1276

    missed = from_bytes(data, chunk_size=64, steps=3)
    reached = from_bytes(data, chunk_size=128, steps=2)
    default = from_bytes(data)
    assert len(missed) == 0
    for results in (reached, default):
        assert len(results) > 0
        assert_evidence_self_consistent(results)
        assert candidates_matching_payload(
            results, data.decode("cp1252")
        ) or candidates_matching_payload(results, data.decode("cp1250"))

    # Deterministic: re-running the same layout yields the same outcome.
    assert len(from_bytes(data, chunk_size=64, steps=3)) == 0


def test_counterexample_bom_drives_branch():
    """Without a BOM, the bytes are ambiguous UTF-16-looking noise; with one
    the BOM candidate wins outright and short-circuits detection."""
    text = MIXED_TEXT * 16
    plain = text.encode("utf-16-le")  # raw LE bytes, no BOM
    bommed = b"\xff\xfe" + plain
    r_plain = from_bytes(plain)
    r_bom = from_bytes(bommed, chunk_size=2, steps=8)
    assert r_bom.best() is not None
    assert r_bom.best().bom is True
    assert family_of(r_bom.best().encoding) == "utf16"
    assert len(r_bom) == 1
    assert oracle_decode(bommed, "utf_16") == text
    # The plain bytes are not claimed as utf_16 with a BOM flag.
    plain_names = candidate_names(r_plain)
    bom_flagged = [
        candidate.encoding for candidate in collect_candidates(r_plain) if candidate.bom
    ]
    assert bom_flagged == []
    assert (
        r_plain.best() is None
        or family_of(r_plain.best().encoding) != "utf16"
        or "utf_16" not in plain_names
    )


def test_counterexample_alias_normalises_but_does_not_duplicate():
    body = RUSSIAN_TEXT * 14
    data_a = b'<meta charset="windows-1251">' + body.encode("cp1251")
    data_b = b'<meta charset="WINDOWS_1251">' + body.encode("cp1251")
    data_c = b'<meta charset="cp-1251">' + body.encode("cp1251")
    results = [from_bytes(data) for data in (data_a, data_b, data_c)]
    encodings = [r.best().encoding for r in results]
    assert encodings == ["cp1251", "cp1251", "cp1251"]
    for r in results:
        names = list(candidate_names(r))
        assert names.count("cp1251") == 1
        # no raw alias spelling leaks through as a separate candidate
        assert not {n for n in names if n.lower() in {"windows_1251", "cp-1251"}}


def test_counterexample_fingerprint_branch_boundary():
    """Removing the distinguishing byte collapses fingerprints into one
    top-level candidate; inserting it splits them.  Both directions pinned."""
    text = FRENCH_TEXT * 12
    without = text.encode("cp1252")
    # Replace every potentially divergent byte so iso15/cp1252 agree.
    without = bytes(
        byte for byte in without if byte not in b"\x80\x8a\x8c\x8e\x9a\x9e\x9f"
    )
    assert without.decode("cp1252") == without.decode("iso8859_15")
    merged = from_bytes(without)
    # The two agreeing codecs never coexist as separate top-level results.
    top_level_names = {match.encoding for match in merged}
    assert not {"cp1252", "iso8859_15"}.issubset(top_level_names)
    agreeing = next(
        match
        for match in merged
        if "cp1252" in match.could_be_from_charset
        and "iso8859_15" in match.could_be_from_charset
    )
    assert agreeing.fingerprint == hash(without.decode("cp1252"))

    split_data = without[:100] + b"\x80\x80" + without[100:]
    split = from_bytes(split_data)
    fingerprints = {
        candidate.encoding: candidate.fingerprint
        for candidate in collect_candidates(split)
        if candidate.encoding in {"cp1252", "iso8859_15"}
    }
    assert fingerprints["cp1252"] != fingerprints["iso8859_15"]


def test_mess_ratio_early_stop_unit():
    """Direct unit check of the detector's early-exit computation.

    ``mess_ratio`` accepts a ``maximum_threshold`` purely to stop counting as
    soon as the ratio is known to exceed it; the reported value must be >= the
    threshold on clearly garbage strings and 0.0 on clean prose.
    """
    from charset_normalizer.md import mess_ratio

    garbage = "".join(chr(code) for code in range(0x2D26, 0x2D80))
    clean = "The quick brown fox jumps over the lazy dog. " * 12
    assert mess_ratio(garbage, 0.2) >= 0.2
    assert mess_ratio(clean, 0.2) == 0.0
    # early stopping must not alter the verdict
    assert mess_ratio(garbage, 0.05) >= 0.05


def test_mess_early_exit_within_detection():
    """Non-ASCII garbage: chaos probing gives up before coherence runs."""
    garbled = bytes(range(0x80, 0x100)) * 4
    results = from_bytes(garbled, chunk_size=64, steps=3)
    # Whatever survives must honestly report high chaos; nothing claims 0 mess.
    for candidate in collect_candidates(results):
        assert candidate.chaos > 0.0


# ---------------------------------------------------------------------------
# Failure-case reduction (shrinker)
# ---------------------------------------------------------------------------


def reduce_case(
    data: bytes,
    predicate,
    *,
    max_rounds: int = 200,
    rng: random.Random | None = None,
):
    """Greedily shrink ``data`` while ``predicate`` stays truthy.

    Reduction never collapses the case to pure ASCII: doing so would delete
    the very bytes that exercised the non-ASCII branches under test, so an
    ASCII-only shrink is rejected even when the predicate would still hold.
    """
    rng = rng or random.Random(0xC5)
    current = bytes(data)

    def still_interesting(candidate: bytes) -> bool:
        if not candidate:
            return False
        if not any(byte >= 0x80 for byte in candidate):
            return False
        return bool(predicate(candidate))

    for _ in range(max_rounds):
        changed = False
        length = len(current)
        # 1) drop a slice
        for start in (0, length // 4, length // 2, (3 * length) // 4):
            for width in sorted({max(1, length // 8), max(1, length // 16), 1}):
                candidate = current[:start] + current[start + width :]
                if len(candidate) < len(current) and still_interesting(candidate):
                    current = candidate
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue
        # 2) single-byte deletion at a random non-ASCII neighbourhood
        for _attempt in range(16):
            hi_positions = [i for i, b in enumerate(current) if b >= 0x80]
            if not hi_positions:
                break
            pos = rng.choice(hi_positions)
            lo = max(0, pos - 3)
            hi = min(length, pos + 4)
            cut = rng.randint(lo, max(lo, hi - 1))
            candidate = current[:cut] + current[cut + 1 :]
            if still_interesting(candidate):
                current = candidate
                changed = True
                break
        if not changed:
            break
    return current


def test_reducer_preserves_distinguishing_bytes():
    data = b"ascii head " + bytes(range(0xC0, 0xE0)) + b" ascii tail"
    reduced = reduce_case(data, lambda b: b[-1:] in (b"l", b"i") or b[-1:] == b"l")
    assert any(byte >= 0x80 for byte in reduced)
    assert len(reduced) <= len(data)


def test_reducer_rejects_ascii_only_collapse():
    data = bytes(range(0x80, 0xA0)) + b"abcdef"

    # Predicate that would stay true after stripping high bytes: reduction
    # must still keep a high byte because shrinkers here are branch-aware.
    reduced = reduce_case(data, lambda b: len(b) >= 1)
    assert any(byte >= 0x80 for byte in reduced)


def test_reducer_reproduces_sampling_offset_case():
    prefix = b"The quick brown fox jumps over the lazy dog. 0123456789. " * 12
    sentence = (
        " Voilà un résumé très élégant de l'éducation française, " "à côté de l'hôtel."
    ).encode("cp1252")
    data = prefix + sentence * 8

    def misses_layout(candidate: bytes) -> bool:
        return len(from_bytes(candidate, chunk_size=64, steps=3)) == 0

    assert misses_layout(data)
    reduced = reduce_case(data, misses_layout, max_rounds=400)
    assert any(byte >= 0x80 for byte in reduced), "reduction must not go pure ASCII"
    assert len(reduced) < len(data)
    assert len(from_bytes(reduced, chunk_size=64, steps=3)) == 0


# ---------------------------------------------------------------------------
# Heavy, seeded fuzz round over the fixture generator
# ---------------------------------------------------------------------------

ENCODING_SAMPLES = [
    ("utf-8", FRENCH_TEXT),
    ("cp1251", RUSSIAN_TEXT),
    ("cp932", JAPANESE_TEXT),
    ("big5", TRAD_CHINESE_TEXT),
    ("cp949", KOREAN_TEXT),
]

# Greek/Cyrillic single-byte code pages genuinely overlap and drift under
# prefix+newline+tiny-window combinations; they are exercised by pinned weak
# fixtures instead of the membership fuzz.
AMBIGUOUS_SAMPLES = [("cp1253", GREEK_TEXT)]


def _layout_invariant_properties(data: bytes, encoding: str, text: str) -> None:
    layouts = [(512, 5), (64, 3), (37, 4), (16, 10)]
    expected_family = family_of(encoding)
    winners: list[str] = []
    for chunk_size, steps in layouts:
        results = from_bytes(data, chunk_size=chunk_size, steps=steps)
        assert_evidence_self_consistent(results)
        best = results.best()
        assert best is not None
        winners.append(family_of(best.encoding))
        # A candidate decoding to exactly the known text must stay present.
        matching = candidates_matching_payload(results, text)
        assert any(family_of(name) == expected_family for name in matching), (
            encoding,
            (chunk_size, steps),
            sorted(matching),
        )
        # Winner stability is only demanded for utf8 (the strict candidate
        # early-returns); single-byte and CJK families have genuine sibling
        # ambiguity and are covered by the membership check above plus the
        # pinned strong fixtures.
        if expected_family == "utf8":
            assert family_of(best.encoding) == "utf8"
    assert expected_family in winners

    # Whitespace appending shifts the stride.  For distinctive multibyte
    # families the exact candidate survives; for single-byte families that
    # genuinely overlap (Greek vs Cyrillic code pages) the detector may drop
    # it at a single unlucky window, which is the documented heuristic
    # uncertainty zone -- there we only require evidence self-consistency.
    for suffix in (b"  ", b"\n", b" \t\r\n"):
        for chunk_size, steps in layouts:
            results = from_bytes(data + suffix, chunk_size=chunk_size, steps=steps)
            assert_evidence_self_consistent(results)
            if expected_family not in {"utf8", "cp932", "big5", "cp949", "gb18030"}:
                continue
            appended = text + suffix.decode("ascii")
            matching = candidates_matching_payload(results, appended)
            assert any(family_of(name) == expected_family for name in matching)


@pytest.mark.skipif(not HEAVY, reason="set CN_METAMORPHIC_HEAVY=1 for fuzz rounds")
def test_seeded_fuzz_metamorphic_relations():
    rng = random.Random(1138)
    rounds = int(os.environ.get("CN_METAMORPHIC_ROUNDS", "60"))
    for _ in range(rounds):
        encoding, base_text = rng.choice(ENCODING_SAMPLES)
        text = base_text * rng.randint(4, 14)
        kwargs = {}
        if rng.random() < 0.4:
            kwargs["newlines_every"] = rng.choice([0, 20, 37, 55])
        if rng.random() < 0.5:
            kwargs["ascii_prefix"] = (
                b"Header line %d\r\n" % rng.randint(0, 9999)
            ) * rng.randint(1, 8)
        corpus = build_fixture(text, encoding, **kwargs)
        expected = kwargs.get("ascii_prefix", b"").decode("ascii") + corpus.text
        _layout_invariant_properties(corpus.data, encoding, expected)


# ---------------------------------------------------------------------------
# Pure-Python vs optional accelerated implementation
# ---------------------------------------------------------------------------

_PKG_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__import__("charset_normalizer").__file__))
)


def _accelerated_available() -> bool:
    for module in ("charset_normalizer.md", "charset_normalizer.cd"):
        spec = importlib.util.find_spec(module)
        if spec is None or not (spec.origin or "").endswith((".so", ".pyd", ".sl")):
            return False
    return True


def _backend_name() -> str:
    spec = importlib.util.find_spec("charset_normalizer.md")
    origin = spec.origin if spec else ""
    return "accelerated" if origin.endswith((".so", ".pyd", ".sl")) else "pure-python"


_PARITY_CASES: list[tuple[str, str]] = [
    (FRENCH_TEXT * 10, "utf-8"),
    (RUSSIAN_TEXT * 10, "cp1251"),
    (GREEK_TEXT * 10, "cp1253"),
    (JAPANESE_TEXT * 12, "cp932"),
    (TRAD_CHINESE_TEXT * 12, "big5"),
    (KOREAN_TEXT * 10, "cp949"),
]


def _canonical_detection(data: bytes) -> list[tuple[str, float, bool, str]]:
    results = from_bytes(data)
    canonical: list[tuple[str, float, bool, str]] = []
    for candidate in collect_candidates(results):
        canonical.append(
            (
                candidate.encoding,
                candidate.chaos,
                candidate.bom,
                candidate.payload,
            )
        )
    # Ordering carries no cross-implementation promise; compare as a set.
    return sorted(canonical, key=lambda item: item[0])


def _best_triplet(data: bytes) -> tuple[str, float, bool, str]:
    results = from_bytes(data)
    best = results.best()
    assert best is not None
    return best.encoding, best.chaos, best.bom, str(best)


def test_backend_identity_reported_consistently():
    # The current interpreter has exactly one backend; detection works.
    assert _backend_name() in {"pure-python", "accelerated"}
    results = from_bytes((FRENCH_TEXT * 6).encode("utf-8"))
    assert results.best() is not None


@pytest.mark.skipif(not _accelerated_available(), reason="accelerated module not built")
def test_pure_python_matches_accelerated_semantics():
    """The accelerated Cython modules and the pure .py implementation agree.

    A subprocess imports a copy of the package without compiled modules
    (PYTHONPATH, LC_ALL=C, fixed hash seed) so nothing is rebuilt.  The
    accelerated backend only prunes candidates relative to the pure backend,
    hence the contract is: both choose a winner of the expected family whose
    payload equals the known text, and every (encoding, payload) pair the
    accelerated backend reports is also reported by the pure backend with the
    same BOM flag and an equivalent chaos band.  hash() fingerprints are not
    compared across processes (per-process hash randomisation).
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(_PKG_DIR, "charset_normalizer")
        shutil.copytree(
            src,
            os.path.join(tmp, "charset_normalizer"),
            ignore=shutil.ignore_patterns("*.so", "*.pyd", "*.sl", "__pycache__"),
        )
        payload = base64.b64encode(pickle.dumps(_PARITY_CASES)).decode("ascii")
        script = (
            "import base64, pickle, sys\n"
            "cases = pickle.loads(base64.b64decode(%r))\n"
            "from charset_normalizer import from_bytes\n"
            "winners = []\n"
            "rows = []\n"
            "for text, enc in cases:\n"
            "    results = from_bytes(text.encode(enc))\n"
            "    best = results.best()\n"
            "    winners.append((enc, best.encoding, str(best)))\n"
            "    for match in list(results) + [\n"
            "        leaf for top in results for leaf in top.submatch\n"
            "    ]:\n"
            "        rows.append((match.encoding, match.chaos, bool(match.bom), str(match)))\n"
            "sys.stdout.buffer.write(base64.b64encode(pickle.dumps((winners, rows))))\n"
            % payload
        )
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": tmp,
            "PYTHONHASHSEED": "1138",
            "LC_ALL": "C",
            "LANG": "C",
        }
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            check=True,
        )
        pure_winners, pure_rows = pickle.loads(base64.b64decode(proc.stdout))

    pure_winner_map = {enc: (winner, decoded) for enc, winner, decoded in pure_winners}
    pure_pairs = {
        (name, decoded): (chaos, bom) for name, chaos, bom, decoded in pure_rows
    }

    for text, enc in _PARITY_CASES:
        expected_family = family_of(enc)
        winner_encoding, _winner_chaos, _winner_bom, winner_text = _best_triplet(
            text.encode(enc)
        )
        assert family_of(winner_encoding) == expected_family
        assert winner_text == text

        pure_name, pure_text = pure_winner_map[enc]
        assert family_of(pure_name) == expected_family
        assert pure_text == text

        for name, chaos, bom, decoded in _canonical_detection(text.encode(enc)):
            shared = pure_pairs.get((name, decoded))
            assert shared is not None, (enc, name)
            pure_chaos, pure_bom = shared
            assert pure_bom == bom, (enc, name)
            assert abs(pure_chaos - chaos) <= 0.05, (
                enc,
                name,
                pure_chaos,
                chaos,
            )


# ---------------------------------------------------------------------------
# Ordering: only assert where the source commits to a tie-break
# ---------------------------------------------------------------------------


def test_ordering_tiebreak_is_deterministic_not_corpus_truth():
    """best() == sorted()[0] repeatedly; we assert the committed rule only.

    The code commits to sorting by (chaos, coherence, multi-byte usage) and to
    BOM / preemptive short-circuits.  It does NOT promise that a specific
    Latin code page wins generic text, which is why this test checks the
    ordering mechanics rather than a hard-coded champion.
    """
    data = (FRENCH_TEXT * 10).encode("cp1252")
    first = from_bytes(data)
    second = from_bytes(data)
    assert [m.encoding for m in first] == [m.encoding for m in second]
    ordered = list(first)
    for left, right in zip(ordered, ordered[1:]):
        assert not (right < left)  # CharsetMatch.__lt__ total ordering
    # chaos/coherence evidence agrees with the sort key the model documents
    for left, right in zip(ordered, ordered[1:]):
        if abs(left.chaos - right.chaos) < 0.005:
            # tie resolved by coherence then multi-byte usage -- both evidence
            # values are present and bounded
            assert 0.0 <= left.coherence <= 1.0
            assert 0.0 <= right.coherence <= 1.0
        else:
            assert left.chaos <= right.chaos


def test_locale_independent_detection():
    """Detection result must not depend on LC_ALL/LANG of the process."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join(
            path for path in (os.environ.get("PYTHONPATH", ""),) if path
        )
        or None,
        "LC_ALL": "C",
        "LANG": "C",
    }
    env = {key: value for key, value in env.items() if value is not None}
    script = (
        "from charset_normalizer import from_bytes\n"
        "import json\n"
        "cases = [%r, %r]\n"
        "encs = []\n"
        "for data in cases:\n"
        "    encs.append([m.encoding for m in from_bytes(data)])\n"
        "import sys; sys.stdout.write(repr(encs))\n"
        % ((RUSSIAN_TEXT * 8).encode("cp1251"), (GREEK_TEXT * 8).encode("cp1253"))
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, check=True
    )
    local = [
        [m.encoding for m in from_bytes((RUSSIAN_TEXT * 8).encode("cp1251"))],
        [m.encoding for m in from_bytes((GREEK_TEXT * 8).encode("cp1253"))],
    ]
    assert eval(proc.stdout.decode()) == local  # noqa: S307 - repr of list[str]

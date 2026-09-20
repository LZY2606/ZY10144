"""Boundary-sensitive metamorphic cases.

Covers:
* a multibyte character landing exactly on a sampling boundary (ASCII lead
  padding shifts 2-byte characters through both cut positions while chunk
  sizes vary across boundary residues);
* a BOM longer than the first sampled chunk, including the no-strip BOM
  branch (UTF-16/32 keep the signature during probing);
* a long ASCII prefix after which the distinguishing bytes only appear.
"""

from __future__ import annotations

import pytest

from charset_normalizer import from_bytes

from .byte_model import Corpus, build_fixture, independent_decode
from .relations import assert_strong_invariance

_JAPANESE = (
    "これは日本語の文章です。文字コードの判定テスト、境界をまたぐ配置を確認します。"
    "繰り返しによって長さを確保し、どの位置で切っても復元できること。"
)


def _shift_jis_corpus() -> Corpus:
    from .byte_model import LARGE_CORPORA

    return next(c for c in LARGE_CORPORA if c.name == "japanese_shift_jis")


@pytest.mark.parametrize("padding", [0, 1, 2, 3])
@pytest.mark.parametrize("chunk_size", [40, 41, 63, 64, 127, 128])
def test_multibyte_char_straddles_sampling_boundary(
    padding: int, chunk_size: int
) -> None:
    # The 1..3 byte ASCII lead shifts every Shift_JIS pair by a different
    # residue modulo 2, so combined with varying chunk sizes some sampled
    # boundary is guaranteed to fall between the lead/trail byte of a
    # multibyte character. The detector's boundary repair must recover the
    # exact text in every case (independently oracle-checked).
    body = _JAPANESE.encode("shift_jis")
    data = (b"a" * padding) + body
    expected = ("a" * padding) + _JAPANESE

    results = from_bytes(data, chunk_size=chunk_size, steps=8)
    winner = results.best()
    assert winner is not None
    assert str(winner) == expected
    assert independent_decode(data, winner.encoding) == expected
    assert frozenset(winner.could_be_from_charset) == frozenset(
        {"cp932", "shift_jis", "shift_jis_2004", "shift_jisx0213"}
    )


def test_straddling_keeps_winner_across_layouts() -> None:
    # Same relation over a sweep of straddling sizes on the large corpus.
    corpus = _shift_jis_corpus()
    layouts = tuple((size + cut, 6) for size in (63, 127, 255, 511) for cut in (0, 1))
    assert_strong_invariance(build_fixture(corpus), layouts, trailing_whitespace=())


@pytest.mark.parametrize("bom_encoding", ["utf_16", "utf_32"])
def test_bom_longer_than_first_chunk(bom_encoding: str) -> None:
    from .byte_model import LARGE_CORPORA

    name = {"utf_16": "cyrillic_utf16", "utf_32": "cyrillic_utf32"}[bom_encoding]
    corpus = next(c for c in LARGE_CORPORA if c.name == name)
    fixture = build_fixture(corpus, bom=True)
    bom_length = 2 if bom_encoding == "utf_16" else 4
    # First chunk shorter than / equal to / slightly longer than the BOM.
    tiny_layouts = tuple(
        (max(1, bom_length + delta), 30) for delta in (-1, 0, 2, 6)
    ) + ((512, 5),)
    for chunk_size, steps in tiny_layouts:
        results = from_bytes(fixture.data, chunk_size=chunk_size, steps=steps)
        winner = results.best()
        assert winner is not None
        assert winner.byte_order_mark is True
        assert frozenset(winner.could_be_from_charset) == corpus.family
        # No-strip BOM branch must still recover the BOM-free text.
        assert str(winner) == fixture.expected_text
        assert (
            independent_decode(fixture.data, winner.encoding) == fixture.expected_text
        )


@pytest.mark.parametrize(
    "corpus_name",
    [
        "cyrillic_after_ascii_prefix_cp1251",
        "japanese_after_ascii_prefix_sjis",
        "chinese_after_ascii_prefix_gb18030",
    ],
)
def test_distinguishing_bytes_after_long_ascii_prefix(corpus_name: str) -> None:
    from .byte_model import LARGE_CORPORA

    corpus = next(c for c in LARGE_CORPORA if c.name == corpus_name)
    fixture = build_fixture(corpus)
    # The ASCII prefix alone would be indistinguishable; layouts that sample
    # the prefix before the tail must still converge once the tail is seen.
    layouts = ((512, 5), (256, 5), (128, 8), (96, 12), (64, 16))
    assert_strong_invariance(fixture, layouts, trailing_whitespace=())


def test_utf8_sig_is_stripped_from_payload() -> None:
    text = _JAPANESE
    fixture = build_fixture(
        Corpus(
            name="utf8_bom",
            text=text,
            encoding="utf_8",
            family=frozenset({"utf_8"}),
        ),
        bom=True,
    )
    winner = from_bytes(fixture.data).best()
    assert winner is not None
    assert winner.byte_order_mark is True
    assert str(winner) == text  # leading \ufeff removed

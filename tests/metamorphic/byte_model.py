"""Byte-model fixtures for the charset-normalizer metamorphic tests.

Everything here is derived from Unicode text we *know* and the codec tables
shipped with the interpreter: :func:`encode` is the independent source of
truth for the bytes->str relation. Nothing in this module inspects a
``CharsetMatch`` to decide what an "expected" answer is; classification of a
fixture as strong/ambiguous is driven solely by the bytes and the known
generator encoding.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field

from charset_normalizer.utils import iana_name

# Fixed seed + bounded corpora: the random module is only used to place
# repetition boundaries, never to decide pass/fail expectations.
SEED = 1138

# Sampling layouts that force the detector's chunk cutter to walk different
# offsets. All keep payload > chunk_size * steps where it matters (large
# fixtures are long enough), and values deliberately straddle typical
# multibyte character widths (1/2/4 bytes).
DEFAULT_LAYOUTS: tuple[tuple[int, int], ...] = (
    (512, 5),
    (256, 5),
    (128, 6),
    (103, 11),
    (64, 12),
)

HEAVY_LAYOUTS: tuple[tuple[int, int], ...] = DEFAULT_LAYOUTS + (
    (511, 5),
    (257, 7),
    (71, 13),
    (46, 17),
    (32, 20),
)

# Trailing ASCII whitespace: appending it changes byte length and therefore
# sampling stride without touching any encoded glyph.
TRAILING_WHITESPACE: tuple[bytes, ...] = (
    b" ",
    b"\n",
    b"  \t \r\n ",
    b" " * 64,
)


@dataclass(frozen=True)
class Corpus:
    """A known Unicode text with the encoding it is generated through."""

    name: str
    text: str
    encoding: str
    family: frozenset[str] = field(default_factory=frozenset)
    aliases: tuple[str, ...] = ()
    bom: bool = False

    def __post_init__(self) -> None:
        # Normalise the family through the same IANA table the detector uses
        # so assertions compare canonical codec names.
        normalised = frozenset(
            iana_name(member, strict=False) for member in self.family
        )
        if iana_name(self.encoding, strict=False) not in normalised:
            normalised = normalised | {iana_name(self.encoding, strict=False)}
        object.__setattr__(self, "family", normalised)


# Real text, repeated to lengths large enough that changing chunk_size/steps
# actually changes which byte offsets are sampled. Families below contain
# only codec names known to decode these exact bytes to the *same* string
# (verified in tests through the independent decoder), not "the top hit on
# some external corpus".
_CYRILLIC_SENTENCE = (
    "Всеки человек рождается свободным и равным в своем достоинстве и правах. "
    "Они наделены разумом и совестью и должны поступать в духе братства. "
)
_HEBREW_SENTENCE = (
    "כל בני אדם נולדו בני חורין ושווים בערכם ובזכויותיהם. "
    "כולם חוננו בתבונה ובמצפון, ולפיכך חובה לנהוג איש ברעהו ברוח אחוה. "
)
_ARABIC_SENTENCE = (
    "العقلية والتنويم المغناطيسي أو الاقتراح، نص عربي طويل بما يكفي "
    "لاختبار الترميز بشكل صحيح ومستقر عبر مختلف مواضع أخذ العينات. "
)
_JAPANESE_SENTENCE = (
    "これは日本語の文章です。文字コードの判定を行うためには、十分な長さの"
    "テキストを繰り返し用意する必要があります。境界をまたぐ場合も同様です。"
)
_CHINESE_SENTENCE = (
    "这是一段用于测试字符编码自动检测功能的中文文本，需要足够的长度才能让"
    "采样覆盖到输入序列中各个不同的区块位置并保持判定稳定。"
)
_LATIN_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while a quiet wind moves "
    "through the old forest and the river keeps flowing to the distant sea. "
)
_RUSSIAN_WITH_LATIN = _LATIN_SENTENCE * 4 + _CYRILLIC_SENTENCE * 10
_JAPANESE_WITH_LATIN = _LATIN_SENTENCE * 4 + _JAPANESE_SENTENCE * 8
_CHINESE_WITH_LATIN = _LATIN_SENTENCE * 4 + _CHINESE_SENTENCE * 8

LARGE_CORPORA: tuple[Corpus, ...] = (
    Corpus(
        name="cyrillic_cp1251",
        text=_CYRILLIC_SENTENCE * 12,
        encoding="cp1251",
        # Members independently verified to decode these bytes identically.
        family=frozenset({"cp1251", "kz1048", "ptcp154"}),
        aliases=("windows-1251", "WINDOWS-1251", "Windows_1251"),
    ),
    Corpus(
        name="arabic_cp1256",
        text=_ARABIC_SENTENCE * 10,
        encoding="cp1256",
        family=frozenset({"cp1256"}),
        aliases=("windows-1256",),
    ),
    Corpus(
        name="japanese_shift_jis",
        text=_JAPANESE_SENTENCE * 10,
        encoding="shift_jis",
        family=frozenset({"cp932", "shift_jis", "shift_jis_2004", "shift_jisx0213"}),
        aliases=("SHIFT-JIS", "Shift_JIS"),
    ),
    Corpus(
        name="chinese_gb18030",
        text=_CHINESE_SENTENCE * 10,
        encoding="gb18030",
        family=frozenset({"gb18030", "gbk", "gb2312"}),
        aliases=("GB18030", "gb18030_2000", "GB18030-2000"),
    ),
    Corpus(
        name="japanese_euc_jp",
        text=_JAPANESE_SENTENCE * 10,
        encoding="euc_jp",
        family=frozenset({"euc_jp", "euc_jis_2004", "euc_jisx0213"}),
        aliases=("EUC-JP",),
    ),
    Corpus(
        name="russian_utf8",
        text=_CYRILLIC_SENTENCE * 10,
        encoding="utf_8",
        family=frozenset({"utf_8"}),
        aliases=("UTF-8", "utf8", "Utf-8"),
    ),
    Corpus(
        name="cyrillic_after_ascii_prefix_cp1251",
        text=_RUSSIAN_WITH_LATIN,
        encoding="cp1251",
        family=frozenset({"cp1251", "kz1048", "ptcp154"}),
    ),
    Corpus(
        name="japanese_after_ascii_prefix_sjis",
        text=_JAPANESE_WITH_LATIN,
        encoding="shift_jis",
        family=frozenset({"cp932", "shift_jis", "shift_jis_2004", "shift_jisx0213"}),
    ),
    Corpus(
        name="chinese_after_ascii_prefix_gb18030",
        text=_CHINESE_WITH_LATIN,
        encoding="gb18030",
        family=frozenset({"gb18030", "gbk", "gb2312"}),
    ),
    Corpus(
        name="cyrillic_utf16",
        text=(_CYRILLIC_SENTENCE + _LATIN_SENTENCE) * 6,
        encoding="utf_16",
        family=frozenset({"utf_16"}),
        bom=True,
    ),
    Corpus(
        name="cyrillic_utf32",
        text=(_CYRILLIC_SENTENCE + _LATIN_SENTENCE) * 6,
        encoding="utf_32",
        family=frozenset({"utf_32"}),
        bom=True,
    ),
)

# Long but genuinely contested: the winner is allowed to move with the
# sampling layout, yet the true encoding family must remain reachable and all
# evidence stay self-consistent. (cp1255 Hebrew ties with cp1251 at a few
# chunk sizes; this is a heuristic uncertainty region, not a hard truth.)
WEAK_LARGE_CORPORA: tuple[Corpus, ...] = (
    Corpus(
        name="hebrew_cp1255",
        text=_HEBREW_SENTENCE * 12,
        encoding="cp1255",
        family=frozenset({"cp1255", "iso8859_8"}),
        aliases=("windows-1255",),
    ),
)

# Short corpora: below/around TOO_SMALL_SEQUENCE (32 bytes) or only a few
# chunks. The heuristic is allowed uncertainty here, so tests make only weak
# claims. These particular texts were chosen (through experiment, not as a
# universal rule) because the true encoding still appears in the result set.
SHORT_CORPORA: tuple[Corpus, ...] = (
    Corpus(
        name="short_cp1251_32",
        text=_CYRILLIC_SENTENCE[:20],
        encoding="cp1251",
        family=frozenset({"cp1251", "kz1048", "ptcp154"}),
    ),
    Corpus(
        name="short_sjis_40",
        text=_JAPANESE_SENTENCE[:16],
        encoding="shift_jis",
        family=frozenset({"cp932", "shift_jis", "shift_jis_2004"}),
    ),
    Corpus(
        name="short_gb18030_36",
        text=_CHINESE_SENTENCE[:12],
        encoding="gb18030",
        family=frozenset({"gb18030", "gbk"}),
    ),
)

# Genuinely ambiguous bytes: pure ASCII is a valid decoding under virtually
# every codec. No winner is asserted; only self-consistency and reachability.
AMBIGUOUS_CORPORA: tuple[Corpus, ...] = (
    Corpus(
        name="pure_ascii",
        text="Plain English text, no non-ASCII bytes at all. " * 2,
        encoding="ascii",
        family=frozenset({"ascii"}),
    ),
)

# Encoded payloads that embed an ASCII charset/encoding declaration. The
# declaration changes the preemptive branch but must not fabricate truth:
# a wrong declaration on undecodable bytes loses, a right one keeps the
# winner. Declaration spellings deliberately mix case and dashes.
DECLARATION_TEMPLATES: tuple[str, ...] = (
    '<meta charset="{name}">\n',
    '<?xml version="1.0" encoding="{name}"?>\n',
    "# coding: {name}\n",
)


@dataclass(frozen=True)
class Fixture:
    """Generated bytes plus everything needed to assert the relation."""

    corpus: Corpus
    data: bytes
    # The text the detector is expected to recover for *strong* assertions.
    expected_text: str
    bom: bool
    truncated_tail: int = 0

    @property
    def encoding(self) -> str:
        return self.corpus.encoding

    @property
    def family(self) -> frozenset[str]:
        return self.corpus.family


def build_fixture(
    corpus: Corpus,
    *,
    bom: bool | None = None,
    declaration: str | None = None,
    ascii_prefix: bytes = b"",
    truncate_tail: int = 0,
    repeat_segments: int = 1,
    newline: str | None = None,
) -> Fixture:
    """Encode the known text through its codec table and return a fixture.

    BOM bytes are prepended explicitly from the documented signatures so the
    generator never relies on detector behaviour. ``truncate_tail`` removes
    trailing bytes after encoding to create dangling multibyte sequences.
    ``repeat_segments`` repeats the text joined by ``newline`` (defaulting
    to the codec-neutral space) to control segment boundaries.
    """
    if bom is None:
        bom = corpus.bom
    text = corpus.text
    if repeat_segments > 1:
        joiner = newline if newline is not None else " "
        text = joiner.join([corpus.text] * repeat_segments)
    body = text.encode(_body_codec(corpus.encoding))

    header = ascii_prefix
    if declaration is not None:
        header = header + declaration.encode("ascii")

    expected = (header.decode("ascii") if header else "") + text

    # BOM/signature always comes first; an ASCII declaration (when present)
    # follows it, mirroring real documents.
    data = b"".join((_bom_bytes(corpus.encoding) if bom else b"", header, body))

    if truncate_tail:
        data = data[: len(data) - truncate_tail]

    return Fixture(
        corpus=corpus,
        data=data,
        expected_text=expected,
        bom=bom,
        truncated_tail=truncate_tail,
    )


def _body_codec(encoding: str) -> str:
    # The bare codecs emit no signature, letting us prepend exactly one BOM.
    return {"utf_8": "utf_8", "utf_16": "utf_16_le", "utf_32": "utf_32_le"}.get(
        encoding, encoding
    )


def _bom_bytes(encoding: str) -> bytes:
    if encoding == "utf_8":
        return b"\xef\xbb\xbf"
    if encoding == "utf_16":
        return codecs.BOM_UTF16_LE
    if encoding == "utf_32":
        return codecs.BOM_UTF32_LE
    if encoding == "gb18030":
        return b"\x84\x31\x95\x33"
    return b""


def declaration(alias: str, *, template_index: int = 0) -> str:
    return DECLARATION_TEMPLATES[template_index].format(name=alias)


def independent_decode(data: bytes, encoding: str, *, strip_bom: bool = True) -> str:
    """Decode strictly using a fresh stdlib codec.

    This is deliberately independent of ``CharsetMatch``: it is the oracle
    that pins down the bytes->str relation a result must satisfy.
    """
    text = codecs.decode(data, encoding, errors="strict")
    if strip_bom and text and text[0] == "\ufeff":
        text = text[1:]
    return text


def shrink_preserving_branch(
    data: bytes, encoding: str, *, predicate, keep_high_bytes: bool = True
) -> bytes:
    """Greedily shrink ``data`` while ``predicate(shrunk)`` stays true.

    Used to minimise failing inputs. It refuses to reduce the sample to pure
    ASCII (that would silently drop the multibyte/high-byte branch the
    failure came from), and only keeps a shrunken form that remains strictly
    decodable through the generator codec.
    """

    def viable(candidate: bytes) -> bool:
        if keep_high_bytes and all(byte < 0x80 for byte in candidate):
            return False
        try:
            codecs.decode(candidate, encoding, errors="strict")
        except (UnicodeError, LookupError):
            return False
        try:
            return bool(predicate(candidate))
        except (UnicodeError, LookupError, ValueError, AssertionError):
            return False

    current = bytes(data)
    changed = True
    while changed and len(current) > 1:
        changed = False
        # Remove whole chunks first, then single bytes at either end.
        for take in (max(1, len(current) // 2), 1):
            for cut in (take, -take):
                if cut > 0:
                    candidate = current[cut:]
                else:
                    candidate = current[: len(current) + cut]
                if candidate and candidate != current and viable(candidate):
                    current = candidate
                    changed = True
                    break
            if changed:
                break
    return current

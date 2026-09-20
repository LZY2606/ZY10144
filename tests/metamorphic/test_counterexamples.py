"""Pinned, branch-discriminating counterexamples.

These are fixed witnesses that prove the tests actually traverse distinct
branches: a sampling-offset witness (same bytes, different measured evidence
depending on chunk layout), plus BOM / alias / fingerprint witnesses. They
are asserted as *properties*, never as "this corpus's winner is universal".

The bounded mutation run (CHARSET_NORMALIZER_MT_MUTATE=1) re-discovers the
sampling-offset witness with a fixed seed and asserts the pinned one still
distinguishes layouts.
"""

from __future__ import annotations

import os
import random

import pytest

from charset_normalizer import from_bytes

MUTATE = os.environ.get("CHARSET_NORMALIZER_MT_MUTATE") == "1"

_LATIN_WORDS = (
    b"the quick brown fox jumps over a lazy dog while a quiet wind moves "
    b"through the old forest "
)
# cp1252-decodable high-byte burst; locally raises measured mess.
_BURST = bytes([0xC3, 0xA9, 0x83, 0xC2, 0x99, 0xE2, 0x80, 0x94, 0x93])


def _burst_payload(seed: int, size: int = 2200) -> bytes:
    rng = random.Random(seed)
    out = bytearray()
    while len(out) < size:
        out += _LATIN_WORDS
        if rng.random() < 0.12:
            pos = rng.randrange(0, max(1, len(out)))
            out[pos:pos] = _BURST
    return bytes(out[:size])


def _cp1252_evidence(data: bytes, chunk_size: int, steps: int):
    results = from_bytes(
        data,
        chunk_size=chunk_size,
        steps=steps,
        cp_isolation=["cp1252", "cp1250", "cp1251"],
    )
    if not results:
        return None
    winner = results.best()
    return winner.encoding, round(winner.chaos, 4)


def test_sampling_offset_changes_evidence_on_weak_corpus() -> None:
    # Pinned witness (see module docstring). On a strong corpus the payload
    # is invariant; on this locally-bursty weak corpus the *evidence* (and
    # even acceptance) legitimately shifts with chunk offsets. This proves
    # the strong tests are not vacuously passing on a single code path.
    data = _burst_payload(seed=1)
    evidence = {
        layout: _cp1252_evidence(data, *layout)
        for layout in ((512, 5), (103, 11), (71, 13), (46, 17))
    }
    # At least one layout rejects while another accepts (different branch).
    assert any(value is None for value in evidence.values())
    assert any(value is not None for value in evidence.values())
    # Even where accepted, measured chaos differs across offsets.
    accepted = [value for value in evidence.values() if value is not None]
    assert len({chaos for _, chaos in accepted}) >= 2


@pytest.mark.skipif(not MUTATE, reason="set CHARSET_NORMALIZER_MT_MUTATE=1")
def test_mutation_rediscovers_offset_witness() -> None:
    # Bounded, seeded search: confirms witness existence without pinning the
    # suite to one magic byte string. Fixed seed keeps it reproducible.
    found = None
    for seed in range(120):
        data = _burst_payload(seed=seed)
        values = {
            _cp1252_evidence(data, chunk_size, steps)
            for chunk_size, steps in ((512, 5), (103, 11), (71, 13), (46, 17))
        }
        if None in values and any(v is not None for v in values):
            found = seed
            break
    assert found is not None, (
        "seeded corpus unexpectedly stopped distinguishing layouts"
    )
    assert found == 1, "pinned witness drifted; update test_sampling_offset witness"


def test_bom_branch_witness() -> None:
    text = "Всеки человек рождается свободным. " * 16
    bare = from_bytes(text.encode("utf-8")).best()
    bom = from_bytes(b"\xef\xbb\xbf" + text.encode("utf-8")).best()
    assert bare is not None and bom is not None
    assert bare.byte_order_mark is False
    assert bom.byte_order_mark is True
    # Same payload family, but the BOM flag is a distinct branch result.
    assert str(bom) == text


def test_alias_branch_witness() -> None:
    data = ("Всеки человек рождается свободным. " * 16).encode("cp1251")
    via_dash = from_bytes(data, cp_isolation=["windows-1251"]).best()
    via_upper = from_bytes(data, cp_isolation=["WINDOWS_1251"]).best()
    assert via_dash is not None and via_upper is not None
    assert via_dash.encoding == via_upper.encoding == "cp1251"
    assert via_dash.fingerprint == via_upper.fingerprint


def test_fingerprint_branch_witness() -> None:
    text = "Всеки человек рождается свободным. " * 16
    # ASCII bytes collapse across codecs; cyrillic bytes do not.
    ascii_data = b"plain ascii sentence repeated to length here. " * 30
    collapsed = from_bytes(ascii_data, cp_isolation=["ascii", "cp1252", "utf_8"])
    distinct = from_bytes(text.encode("cp1251"), cp_isolation=["cp1251", "koi8_r"])
    assert len(collapsed) == 1
    assert "ascii" in collapsed.best().could_be_from_charset
    # cp1251 vs koi8_r decode to different strings -> different fingerprints.
    encodings = {match.encoding for match in distinct}
    assert {"cp1251", "koi8_r"} <= encodings
    fps = {match.encoding: match.fingerprint for match in distinct}
    assert fps["cp1251"] != fps["koi8_r"]

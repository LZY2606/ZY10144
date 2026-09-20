"""Mess probing early-exit and dangling multibyte tails.

The early-stop optimisation (give up after a bounded number of chaotic
chunks) is a sampling-layout optimisation: it must reject exactly the
encodings the full pass rejects. Truncated multibyte tails are invalid
input; the detector may return nothing, but must never raise or return a
candidate whose own codec cannot decode the bytes.
"""

from __future__ import annotations

import logging
import random

import pytest

from charset_normalizer import from_bytes
from charset_normalizer.constant import TRACE

from .byte_model import SEED, shrink_preserving_branch
from .relations import assert_evidence_self_consistent

# Bytes every single-byte codec maps (cp1252 undefined slots removed), drawn
# from a fixed seed. Under single-byte codecs this is high-mess gibberish;
# the relation under test is the *rejection decision*, not a guessed winner.
_CP1252_DEFINED = bytes(
    b for b in range(0x80, 0x100) if b not in (0x81, 0x8D, 0x8F, 0x90, 0x9D)
)
_ISOLATION = ["cp1252", "cp1251", "cp1250", "iso8859_5"]


def _gibberish(seed: int = SEED, size: int = 2600) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.choice(_CP1252_DEFINED) for _ in range(size))


class _TraceRecorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def trace_recorder():
    recorder = _TraceRecorder()
    logger = logging.getLogger("charset_normalizer")
    previous_level = logger.level
    logger.setLevel(TRACE)
    logger.addHandler(recorder)
    try:
        yield recorder
    finally:
        logger.removeHandler(recorder)
        logger.setLevel(previous_level)


def _rejected_encodings(data: bytes, *, chunk_size: int, steps: int) -> set[str]:
    # Rejection is observed via results; the trace log confirms the path.
    results = from_bytes(
        data,
        chunk_size=chunk_size,
        steps=steps,
        cp_isolation=_ISOLATION,
    )
    accepted = {name for match in results for name in match.could_be_from_charset}
    return set(_ISOLATION) - accepted


def test_mess_early_exit_matches_full_sampling(trace_recorder) -> None:
    data = _gibberish()
    full_rejected = _rejected_encodings(data, chunk_size=512, steps=5)

    # Layouts that trigger the give-up counter at different chunk indices.
    for chunk_size, steps in ((128, 20), (64, 40), (40, 40)):
        rejected = _rejected_encodings(data, chunk_size=chunk_size, steps=steps)
        assert rejected == full_rejected, (
            f"early-stop layout ({chunk_size},{steps}) rejected {rejected} "
            f"but full sampling rejected {full_rejected}"
        )

    gave_up = [m for m in trace_recorder.messages if "Gave up" in m]
    assert gave_up, "expected the mess early-exit branch to be exercised"


def test_gibberish_yields_no_optimistic_winner(trace_recorder) -> None:
    data = _gibberish()
    results = from_bytes(data, chunk_size=64, steps=40, cp_isolation=_ISOLATION)
    # High-mess bytes must not be promoted; no fallback under isolation of
    # non-privileged codecs.
    assert len(results) == 0
    assert any(
        "excluded because of initial chaos probing" in m
        for m in trace_recorder.messages
    )


@pytest.mark.parametrize("encoding", ["shift_jis", "euc_jp", "utf_8", "utf_16"])
@pytest.mark.parametrize("cut", [1, 2, 3])
def test_truncated_multibyte_tail_never_raises(encoding: str, cut: int) -> None:
    text = "これは日本語の文章です。コード判定のために十分な長さのテキストを用意。" * 14
    data = text.encode(encoding)[:-cut]
    results = from_bytes(data)  # must not raise on dangling bytes
    if results:
        # Any survivor must satisfy the strict bytes->str contract itself.
        assert_evidence_self_consistent(results, data)


def test_truncated_tail_appended_ascii_still_invalid_for_true_codec() -> None:
    import codecs

    text = "これは日本語の文章です。コード判定。" * 14
    dangling = text.encode("shift_jis")[:-1]
    # The true codec must reject both the dangling tail and its whitespace
    # variant: ASCII cannot complete a dangling lead byte.
    for candidate in (dangling, dangling + b"   \n "):
        with pytest.raises(UnicodeDecodeError):
            codecs.decode(candidate, "shift_jis", errors="strict")
    assert from_bytes(dangling).best() is None


def test_shrink_keeps_high_bytes_and_branch() -> None:
    corpus_text = "Всеки человек рождается свободным и равным в достоинстве. " * 12
    data = corpus_text.encode("cp1251")

    # Property used both for shrinking and as the failing condition: winner
    # family stays cp1251 and payload round-trips through stdlib cp1251.
    def property_holds(candidate: bytes) -> bool:
        results = from_bytes(candidate)
        winner = results.best()
        return (
            winner is not None
            and "cp1251" in winner.could_be_from_charset
            and str(winner) == candidate.decode("cp1251")
        )

    reduced = shrink_preserving_branch(data, "cp1251", predicate=property_holds)
    assert len(reduced) <= len(data)
    # The reducer must not quietly turn the sample into pure ASCII.
    assert any(byte >= 0x80 for byte in reduced)
    assert property_holds(reduced)

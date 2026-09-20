"""Metamorphic tests for charset-normalizer.

These tests never hardcode "encoding X is always number one on corpus Y".
Instead they generate bytes from *known* Unicode text through the real codec
tables (the byte model) and assert metamorphic relations:

* for losslessly decodable, sufficiently distinguishing inputs the winner's
  decoded payload and encoding *family* stay invariant across sampling
  layouts (``chunk_size`` / ``steps``) and under trailing ASCII whitespace;
* for short or genuinely ambiguous inputs only the weak-but-sound relations
  hold (the correct candidate stays reachable, and the evidence carried by
  every result is self-consistent with an independent decoder).

Run the fast set with the default ``uv run pytest -q``; the heavier rounds
are opt-in via ``CHARSET_NORMALIZER_MT_HEAVY=1`` and the bounded mutation run
via ``CHARSET_NORMALIZER_MT_MUTATE=1``.
"""

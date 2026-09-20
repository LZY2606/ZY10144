"""Semantic parity between the pure-Python and accelerated heuristic cores.

The optional compiled backends (``charset_normalizer.md`` / ``.cd``) must
yield the same detection conclusions as the pure-Python modules. The fast
suite runs an in-process self-consistency check against whichever backend is
currently imported. The cross-backend comparison runs when a compiled module
is present in the source tree (built with
``CHARSET_NORMALIZER_USE_CYTHON=1 python setup.py build_ext --inplace``) or
when ``CHARSET_NORMALIZER_MT_HEAVY=1`` is set.

Both flavours are loaded in *fresh subprocesses*: the accelerated run uses
the installed package normally (the ``.so`` shadows the ``.py``), the pure
run loads the ``.py`` files directly via an import shim that ignores any
``.so``. Compared evidence is limited to bytes->str conclusions (winner,
payloads, families). Nothing here depends on locale or network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from charset_normalizer import from_bytes
from charset_normalizer import md as current_md

from .byte_model import (
    LARGE_CORPORA,
    SHORT_CORPORA,
    WEAK_LARGE_CORPORA,
    build_fixture,
)

HEAVY = os.environ.get("CHARSET_NORMALIZER_MT_HEAVY") == "1"


def _is_compiled(module: object) -> bool:
    return getattr(module, "__file__", "").endswith((".so", ".pyd", ".dylib"))


def _probes() -> list[dict]:
    probes = []
    for corpus in (*LARGE_CORPORA, *WEAK_LARGE_CORPORA, *SHORT_CORPORA):
        fixture = build_fixture(corpus)
        for chunk_size, steps in ((512, 5), (103, 11), (64, 12)):
            probes.append(
                {"data": list(fixture.data), "cs": chunk_size, "steps": steps}
            )
    return probes


def _summarise(results) -> dict:
    return {
        "best": results.best().encoding if results.best() else None,
        "payloads": sorted({str(match) for match in results}),
        "families": sorted(sorted(match.could_be_from_charset) for match in results),
    }


def test_current_backend_self_consistent() -> None:
    backend = "compiled" if _is_compiled(current_md) else "python"
    for corpus in LARGE_CORPORA:
        fixture = build_fixture(corpus)
        results = from_bytes(fixture.data)
        assert results, f"{backend} backend gave no result for {corpus.name}"
        assert str(results.best())  # winner must lazily decode


def _src_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(here, "src", "charset_normalizer")


def _compiled_artifact_exists() -> bool:
    return any(
        name.endswith((".so", ".pyd", ".dylib")) for name in os.listdir(_src_dir())
    )


def _run(script: str, probes: list[dict]) -> list[dict]:
    proc = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(probes),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


_COMPILED_SCRIPT = r"""
import sys, json
from charset_normalizer import from_bytes
from charset_normalizer import md
payload = json.loads(sys.stdin.read())
out = []
for item in payload:
    res = from_bytes(bytes(item["data"]), chunk_size=item["cs"], steps=item["steps"])
    best = res.best()
    out.append({
        "backend": "compiled" if md.__file__.endswith((".so", ".pyd", ".dylib")) else "python",
        "best": best.encoding if best else None,
        "payloads": sorted({str(m) for m in res}),
        "families": sorted(sorted(m.could_be_from_charset) for m in res),
    })
sys.stdout.write(json.dumps(out, ensure_ascii=False))
"""

# Loads the pure-python modules by explicit file path, so a sibling .so is
# ignored. Used as the independent oracle for the accelerated build.
_PURE_SCRIPT = r"""
import sys, json, types, importlib.util
base = %r

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

pkg = types.ModuleType("charset_normalizer")
pkg.__path__ = [base]
sys.modules["charset_normalizer"] = pkg
load("charset_normalizer.constant", base + "/constant.py")
load("charset_normalizer.utils", base + "/utils.py")
load("charset_normalizer.md", base + "/md.py")
load("charset_normalizer.cd", base + "/cd.py")
load("charset_normalizer.models", base + "/models.py")
load("charset_normalizer.legacy", base + "/legacy.py")
api = load("charset_normalizer.api", base + "/api.py")

payload = json.loads(sys.stdin.read())
out = []
for item in payload:
    res = api.from_bytes(bytes(item["data"]), chunk_size=item["cs"], steps=item["steps"])
    best = res.best()
    out.append({
        "best": best.encoding if best else None,
        "payloads": sorted({str(m) for m in res}),
        "families": sorted(sorted(m.could_be_from_charset) for m in res),
    })
sys.stdout.write(json.dumps(out, ensure_ascii=False))
"""


@pytest.mark.skipif(
    not (_is_compiled(current_md) or (_compiled_artifact_exists() and HEAVY)),
    reason="needs an accelerated build or CHARSET_NORMALIZER_MT_HEAVY=1",
)
def test_python_and_compiled_conclusions_match() -> None:
    probes = _probes()
    src_dir = _src_dir()

    compiled = _run(_COMPILED_SCRIPT, probes)
    if not all(row["backend"] == "compiled" for row in compiled):
        pytest.skip("accelerated module not importable in subprocess")

    pure = _run(_PURE_SCRIPT % src_dir, probes)

    assert len(pure) == len(compiled) == len(probes)
    for index, (pure_row, compiled_row) in enumerate(zip(pure, compiled)):
        assert pure_row["best"] == compiled_row["best"], index
        assert pure_row["payloads"] == compiled_row["payloads"], index
        assert pure_row["families"] == compiled_row["families"], index

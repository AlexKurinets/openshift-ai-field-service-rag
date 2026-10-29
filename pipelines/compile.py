"""Compile both pipelines to IR YAML for upload to Data Science Pipelines.

    python pipelines/compile.py            # writes pipelines/compiled/*.yaml
    python pipelines/compile.py --check    # exits 1 if the committed YAML is stale (CI)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_gate_pipeline import eval_gate  # noqa: E402
from ingest_pipeline import nightly_ingest  # noqa: E402
from kfp import compiler  # noqa: E402

PIPELINES = {"eval_gate.yaml": eval_gate, "nightly_ingest.yaml": nightly_ingest}


def compile_to(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fn in PIPELINES.items():
        path = out_dir / name
        compiler.Compiler().compile(fn, str(path))
        paths[name] = path
    return paths


def main() -> int:
    target = HERE / "compiled"
    if "--check" not in sys.argv:
        for p in compile_to(target).values():
            print(f"wrote {p.relative_to(HERE.parent)}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        stale = [name for name, p in compile_to(Path(tmp)).items()
                 if not (target / name).exists() or (target / name).read_text() != p.read_text()]
    if stale:
        print(f"stale compiled pipelines: {', '.join(stale)}; run python pipelines/compile.py")
        return 1
    print("compiled pipelines are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())

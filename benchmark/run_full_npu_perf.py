"""Run TileLang-Ascend tests file-by-file and preserve performance/failure data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="testing/python")
    parser.add_argument("--output", default="benchmark/results/full_npu_perf")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args()

    repo = Path.cwd()
    output = repo / args.output
    output.mkdir(parents=True, exist_ok=True)
    test_files = sorted((repo / args.root).rglob(args.pattern))
    manifest: list[dict[str, object]] = []
    env = os.environ.copy()
    env.setdefault("TILELANG_CACHE_DIR", str(output / "jit_cache"))

    for index, test_file in enumerate(test_files, 1):
        relative = test_file.relative_to(repo)
        slug = str(relative).replace("/", "__").replace("\\", "__").replace(".py", "")
        json_path = output / f"{slug}.json"
        log_path = output / f"{slug}.log"
        command = [
            "python",
            "-m",
            "pytest",
            "-q",
            str(relative),
            "-p",
            "benchmark.pytest_npu_perf_plugin",
            "--npu-perf-output",
            str(json_path),
            "--tb=long",
        ]
        print(f"[{index}/{len(test_files)}] {relative}", flush=True)
        started = time.monotonic()
        status = "completed"
        returncode = None
        try:
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    command,
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
            returncode = proc.returncode
            if returncode < 0:
                status = "aborted"
            elif returncode not in {0, 1, 5}:
                status = "process_error"
        except subprocess.TimeoutExpired:
            status = "timeout"
        manifest.append(
            {
                "file": str(relative),
                "status": status,
                "returncode": returncode,
                "elapsed_s": time.monotonic() - started,
                "result": str(json_path.relative_to(repo)) if json_path.exists() else None,
                "log": str(log_path.relative_to(repo)),
            }
        )
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

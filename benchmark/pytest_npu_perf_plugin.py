"""Pytest plugin that records TileLang JIT device time and test outcomes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

_CURRENT_NODEID = "<collection>"
_KERNEL_RECORDS: list[dict[str, object]] = []
_TEST_RECORDS: dict[str, dict[str, object]] = {}
_ORIGINAL_CALL = None


def _classify_failure(text: str) -> str:
    lowered = text.lower()
    compile_markers = (
        "compileerror",
        "compilation failed",
        "build failed",
        "codegen",
        "lowering",
        "syntax error",
        "ptoas",
        "bisheng",
    )
    wrong_markers = (
        "assertionerror",
        "mismatched elements",
        "not equal to tolerance",
        "torch.testing",
    )
    if any(marker in lowered for marker in wrong_markers):
        return "wrong_result"
    if any(marker in lowered for marker in compile_markers):
        return "compile_error"
    return "runtime_error"


def pytest_addoption(parser):
    parser.addoption("--npu-perf-output", action="store", default=None)


def pytest_configure(config):
    global _ORIGINAL_CALL
    try:
        import torch
        from tilelang.jit import JITKernel
    except Exception:
        return

    if _ORIGINAL_CALL is not None:
        return
    _ORIGINAL_CALL = JITKernel.__call__

    def timed_call(self, *args, **kwargs):
        started = time.perf_counter_ns()
        start_event = end_event = None
        if hasattr(torch, "npu") and torch.npu.is_available():
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            start_event.record()
        try:
            result = _ORIGINAL_CALL(self, *args, **kwargs)
        except BaseException as exc:
            _KERNEL_RECORDS.append(
                {
                    "nodeid": _CURRENT_NODEID,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                    "host_ms": (time.perf_counter_ns() - started) / 1e6,
                }
            )
            raise
        record: dict[str, object] = {
            "nodeid": _CURRENT_NODEID,
            "status": "ok",
            "host_ms": (time.perf_counter_ns() - started) / 1e6,
        }
        if start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            record["device_ms"] = float(start_event.elapsed_time(end_event))
        _KERNEL_RECORDS.append(record)
        return result

    JITKernel.__call__ = timed_call


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global _CURRENT_NODEID
    _CURRENT_NODEID = item.nodeid


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when not in {"setup", "call", "teardown"}:
        return
    entry = _TEST_RECORDS.setdefault(item.nodeid, {"nodeid": item.nodeid})
    entry[f"{report.when}_outcome"] = report.outcome
    entry[f"{report.when}_duration_s"] = report.duration
    if getattr(report, "wasxfail", None):
        entry["wasxfail"] = str(report.wasxfail)
    if report.failed:
        text = getattr(report, "longreprtext", str(report.longrepr))
        entry["failure_phase"] = report.when
        entry["failure_text"] = text[-12000:]
        entry["category"] = _classify_failure(text)


def pytest_sessionfinish(session, exitstatus):
    output = session.config.getoption("--npu-perf-output")
    if not output:
        return
    for entry in _TEST_RECORDS.values():
        if "category" in entry:
            continue
        if entry.get("wasxfail"):
            entry["category"] = "expected_failure"
        elif entry.get("call_outcome") == "passed":
            entry["category"] = "correct"
        elif entry.get("setup_outcome") == "skipped" or entry.get("call_outcome") == "skipped":
            entry["category"] = "skipped"
        else:
            entry["category"] = "runtime_error"
    payload = {
        "exitstatus": int(exitstatus),
        "pid": os.getpid(),
        "tests": list(_TEST_RECORDS.values()),
        "kernels": _KERNEL_RECORDS,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

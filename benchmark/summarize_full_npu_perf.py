"""Summarize full TileLang-Ascend correctness and NPU timing results."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def first_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:500] if lines else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    args = parser.parse_args()
    root = Path(args.result_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    tests: list[dict[str, object]] = []
    kernels: list[dict[str, object]] = []
    process_failures: list[dict[str, object]] = []
    for item in manifest:
        result_name = item.get("result")
        if result_name:
            payload = json.loads(Path(result_name).read_text(encoding="utf-8"))
            tests.extend(payload.get("tests", []))
            kernels.extend(payload.get("kernels", []))
        if item["status"] != "completed" or not result_name:
            process_failures.append(item)

    counts = Counter(str(test.get("category", "unknown")) for test in tests)
    timings: dict[str, list[float]] = defaultdict(list)
    for kernel in kernels:
        if kernel.get("status") == "ok" and kernel.get("device_ms") is not None:
            timings[str(kernel["nodeid"])].append(float(kernel["device_ms"]))

    test_csv = root / "test_classification.csv"
    with test_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["nodeid", "category", "call_duration_s", "failure_phase", "summary", "log"],
        )
        writer.writeheader()
        log_by_file = {str(i["file"]): str(i["log"]) for i in manifest}
        for test in tests:
            file_name = str(test["nodeid"]).split("::", 1)[0]
            writer.writerow(
                {
                    "nodeid": test.get("nodeid"),
                    "category": test.get("category"),
                    "call_duration_s": test.get("call_duration_s", ""),
                    "failure_phase": test.get("failure_phase", ""),
                    "summary": first_line(str(test.get("failure_text", ""))),
                    "log": log_by_file.get(file_name, ""),
                }
            )

    perf_csv = root / "kernel_performance.csv"
    with perf_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["nodeid", "calls", "median_ms", "min_ms", "max_ms", "samples_ms"],
        )
        writer.writeheader()
        for nodeid, values in sorted(timings.items()):
            writer.writerow(
                {
                    "nodeid": nodeid,
                    "calls": len(values),
                    "median_ms": f"{statistics.median(values):.6f}",
                    "min_ms": f"{min(values):.6f}",
                    "max_ms": f"{max(values):.6f}",
                    "samples_ms": json.dumps(values),
                }
            )

    try:
        import torch
        import torch_npu
        import tilelang

        npu_name = torch.npu.get_device_name(torch.npu.current_device())
        torch_version = torch.__version__
        torch_npu_version = torch_npu.__version__
        tilelang_version = tilelang.__version__
    except Exception as exc:
        npu_name = torch_version = torch_npu_version = tilelang_version = f"unavailable: {exc}"

    report = root / "FULL_NPU_PERFORMANCE_REPORT.md"
    lines = [
        "# TileLang-Ascend 全量正确性与性能测试报告",
        "",
        "## 1. 范围与口径",
        "",
        "本报告覆盖 `testing/python` 下的测试文件。正确性结论沿用仓库 pytest 断言；性能数据为",
        "TileLang `JITKernel.__call__` 外层通过 `torch.npu.Event` 记录的设备流时延。pytest 总耗时",
        "仅用于诊断，不作为 kernel 性能。一次测试可能启动零个、一个或多个 kernel。",
        "",
        "## 2. 运行环境",
        "",
        "| 项目 | 参数 |",
        "| --- | --- |",
        f"| NPU | {npu_name} |",
        f"| 驱动/CANN | `{command('npu-smi', 'info', '-t', 'board', '-i', '0')[:300]}` |",
        f"| 操作系统 | {platform.platform()} |",
        f"| CPU/架构 | {platform.machine()}，{command('nproc')} vCPU |",
        f"| Python | {platform.python_version()} |",
        f"| PyTorch | {torch_version} |",
        f"| torch_npu | {torch_npu_version} |",
        f"| TileLang | {tilelang_version} |",
        f"| Git 分支/提交 | `{command('git', 'branch', '--show-current')}` / `{command('git', 'rev-parse', 'HEAD')}` |",
        "",
        "## 3. 结果总览",
        "",
        f"- 测试文件：{len(manifest)}",
        f"- pytest 用例记录：{len(tests)}",
        f"- 有 NPU Event 数据的用例：{len(timings)}",
        f"- NPU kernel 调用样本：{sum(len(v) for v in timings.values())}",
        "",
        "| 分类 | 数量 | 含义 |",
        "| --- | ---: | --- |",
        f"| 正确运行 | {counts['correct']} | pytest 断言通过 |",
        f"| 运行但结果错误 | {counts['wrong_result']} | 数值/张量断言失败 |",
        f"| 运行时错误 | {counts['runtime_error']} | Python/CANN/设备运行异常 |",
        f"| 无法编译 | {counts['compile_error']} | lowering/codegen/编译阶段失败 |",
        f"| 预期失败 | {counts['expected_failure']} | 测试显式 xfail |",
        f"| 跳过 | {counts['skipped']} | 环境或条件不满足 |",
        f"| 中断/超时文件 | {len(process_failures)} | 子进程 abort、异常退出或超时 |",
        "",
        "逐用例明细见 `test_classification.csv`；逐 kernel 时延见 `kernel_performance.csv`；",
        "失败完整堆栈保存在各测试文件对应的 `.log` 中。",
        "",
        "## 4. 错误与正确答案记录",
        "",
    ]
    failures = [t for t in tests if t.get("category") in {"wrong_result", "runtime_error", "compile_error"}]
    if failures:
        lines += ["| 分类 | 用例 | 摘要 |", "| --- | --- | --- |"]
        for test in failures[:100]:
            summary = first_line(str(test.get("failure_text", ""))).replace("|", "\\|")
            lines.append(f"| {test.get('category')} | `{test.get('nodeid')}` | {summary} |")
    else:
        lines.append("未记录到非预期失败。")
    lines += [
        "",
        "对于数值断言失败，pytest 完整日志保留实际张量、参考张量、最大绝对/相对误差；CSV",
        "保存摘要和日志路径，避免在总报告中截断大张量。",
        "",
        "## 5. 测试中遇到的问题与解决方法",
        "",
        "| 问题 | 处理方法 |",
        "| --- | --- |",
        "| 普通 pytest 不输出 kernel 性能 | 包装 `JITKernel.__call__`，用 NPU Event 在同一设备流计时 |",
        "| JIT 首次编译会放大主机耗时 | 分开保存 `host_ms` 和 `device_ms`，性能表只使用后者 |",
        "| 单个 abort 会终止整套测试 | 每个测试文件使用独立子进程，保留退出码和独立日志 |",
        "| 参数化用例数量多、同步计时较慢 | 后台分文件运行，增量写入 `manifest.json`，中断后已有结果仍可用 |",
        "| 一个用例可能启动多个 kernel | 保存全部样本，并按 pytest nodeid 汇总 calls/median/min/max |",
        "| 小 kernel 易受固定启动开销影响 | 报告设备端到端时延，不把结果解释为单条指令吞吐 |",
        "| 预期失败会混入失败统计 | 单独识别 pytest xfail，不计入非预期错误 |",
        "",
        "## 6. 限制",
        "",
        "- 当前每次测试中的 kernel 调用只采一份 Event 样本；同一参数用例若只调用一次，则该值不具备",
        "  微基准的统计稳定性。关键算子应另做 warmup、多次迭代和多 sample 的专项 benchmark。",
        "- 测试未调用 `JITKernel` 的纯编译/IR 用例可以有正确性结论，但没有设备性能数据。",
        "- 性能只适用于本报告记录的 910B2C、CANN、PyTorch 和 TileLang 提交。",
        "",
    ]
    if process_failures:
        lines += ["## 7. 中断、异常退出或超时文件", "", "| 文件 | 状态 | 退出码 | 日志 |", "| --- | --- | ---: | --- |"]
        for item in process_failures:
            lines.append(f"| `{item['file']}` | {item['status']} | {item['returncode']} | `{item['log']}` |")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)
    print(dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

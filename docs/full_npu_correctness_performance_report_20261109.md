# TileLang-Ascend 全量正确性与性能测试报告（Ascend 910B2C）

## 1. 结论

本轮在 Ascend 910B2C 上逐文件执行 `testing/python` 下 58 个测试文件，共收集
2,502 个 pytest 用例结果：

| 分类 | 数量 | 结论 |
| --- | ---: | --- |
| 正确运行 | 2,490 | pytest 原有正确性断言通过 |
| 能运行但结果错误 | 0 | 未观察到数值或张量对比错误 |
| 运行时出错或进程中断 | 0 | 没有文件级 abort、信号退出或超时 |
| 无法通过编译 | 8 | 6 项 CANN API 不兼容；2 项为故意非法输入负例提前在编译阶段失败 |
| 跳过 | 4 | 测试条件不满足或仓库标记跳过 |

性能采集共得到 1,977 次 NPU kernel 调用样本，覆盖 1,931 个正确 pytest 用例。
其余正确用例属于纯编译、IR、源码检查，或没有经过 `JITKernel.__call__`，因此没有伪造
设备性能数据。

## 2. 运行环境

| 项目 | 参数 |
| --- | --- |
| NPU | Ascend 910B2C，单卡，测试前无其他 NPU 进程 |
| NPU 驱动及 `npu-smi` | 26.0.rc1 |
| HBM | 65,536 MiB；测试前占用约 3,487 MiB |
| 操作系统 | openEuler 24.03 用户环境；Linux 5.15.0-113-generic，x86_64 |
| CPU | 128 vCPU，2 socket × 32 core × 2 thread；云环境型号展示值不作为物理 CPU 结论 |
| CANN | 9.0.0 |
| Python | 3.11.15 |
| PyTorch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| TileLang-Ascend | 0.1.4，基础提交 `a6de467c54e57c3328aa49f9bc7c8d1be11261e7` |
| 测试分支 | `codex/fix-pto-sort-transpose` |
| 性能采集工具分支 | `codex/full-npu-performance-report` |

`lscpu` 上报的 CPU 型号与 128 vCPU 拓扑存在不一致，可能是云平台虚拟化展示，因此本报告
只如实记录 vCPU 拓扑。kernel 计时采用 NPU Event，不使用 CPU 型号推断设备性能。

## 3. 测试方法

全量运行器将 58 个测试文件分别放入独立 pytest 子进程：

1. 保留仓库原有 pytest 输入生成、参考实现和正确性断言。
2. 在 `tilelang.jit.JITKernel.__call__` 外记录 `torch.npu.Event`。
3. 起始 Event 入当前 NPU 流后调用原 kernel，结束 Event 入流并同步。
4. 保存设备时延 `device_ms`，同时单独保留主机调用耗时，不将 JIT 编译时间算作 kernel 性能。
5. 每个测试文件独立保存 JSON 和完整 pytest 日志。
6. 每个文件最长运行 1,800 秒，单文件异常不会导致其他文件结果丢失。
7. 增量写入 `manifest.json`，即使 SSH 中断也保留已经完成的结果。

运行命令：

```bash
source /root/tilelang-ascend/set_env.sh
nohup python benchmark/run_full_npu_perf.py \
  --output benchmark/results/full_npu_perf_20261109 \
  --timeout 1800 \
  > benchmark/results/full_npu_perf_20261109.master.log 2>&1 &
```

## 4. 正确算子性能

逐用例性能保存在：

```text
benchmark/results/full_npu_perf_20261109/kernel_performance.csv
```

CSV 字段包括：

- `nodeid`：pytest 参数化用例，包含算子、shape、dtype 和后端等信息；
- `calls`：该用例经过 `JITKernel.__call__` 的次数；
- `median_ms`、`min_ms`、`max_ms`：该用例采集样本的设备时延；
- `samples_ms`：原始 NPU Event 样本。

### 4.1 总体分布

以 1,931 个有设备数据的 pytest 用例的中位时延统计：

| 分位点 | 时延 |
| ---: | ---: |
| 最小 | 0.054300 ms |
| P50 | 0.363420 ms |
| P90 | 0.422220 ms |
| P95 | 0.480300 ms |
| P99 | 0.927520 ms |
| 最大 | 22.311180 ms |

这些数据用于全量筛查，不应直接当作严格的算子微基准。大多数参数化用例只执行一次，
没有统一进行预热和多轮采样；关键算子的正式性能结论仍应使用专门 benchmark。

### 4.2 时延较高的代表性用例

| 设备时延 | 用例 |
| ---: | --- |
| 22.311180 ms | `elementwise::test_pow_slice[shape0-int32-ascendc]` |
| 14.338960 ms | `elementwise::test_pow_slice[shape0-float16-ascendc]` |
| 13.750240 ms | `elementwise::test_pow_slice[shape0-float-ascendc]` |
| 2.655160 ms | `l1_to_l0::test_gemm_v0_implicit[...,bfloat16,...,ascendc]` |
| 2.279360 ms | `mma::test_select_op[shape0-ascendc]` |
| 1.963100 ms | `mma::test_select_op[shape0-pto]` |

`pow_slice` 明显高于主体分布，适合作为后续专项 profile 和优化候选。这里仅报告观察值，
尚未用多轮预热基准确认其稳定性。

## 5. 错误分类与记录

逐用例分类表保存在：

```text
benchmark/results/full_npu_perf_20261109/test_classification.csv
```

CSV 包含用例、分类、失败阶段、错误摘要和完整日志路径。每个测试文件的完整堆栈保存在同一
结果目录下的 `.log` 文件。

### 5.1 能运行但结果错误

本轮为 0 项，因此没有需要列出的“实际结果与正确答案”。pytest 中的 `torch.testing`、
NumPy 和普通断言均未报告非预期数值差异。

### 5.2 CANN 版本不兼容导致的编译失败：6 项

全部来自 `testing/python/language/test_exception_dump.py`，AscendC 和 PTO 各 3 项：

- `test_exception_dump_npu_correctness`
- `test_exception_dump_callback_magic_not_found`
- `test_exception_dump_callback_via_hw_exception`

Bisheng 编译 `src/tl_templates/ascend/exception_dump.h` 时报告：

```text
unknown type name 'acldumpTensorInfo'
use of undeclared identifier 'ACL_DUMP_TENSOR_INPUT'
use of undeclared identifier 'acldumpSaveExceptionInfo'
```

原因是当前 CANN 9.0.0 头文件没有仓库 Exception Dump 实现依赖的新接口。该组用例在
kernel 运行前就失败，因此没有实际输出，也不能记录 kernel 性能。可选解决方法是升级到
提供这些接口的 CANN 版本，或者为 CANN 9.0.0 增加条件编译和旧版 API 兼容实现；不能仅靠
修改 pytest 规避。

### 5.3 故意非法输入负例提前编译失败：2 项

来自：

- `test_sort_actual_num_zero_crashes[float32]`
- `test_sort_actual_num_zero_crashes[float16]`

测试明确传入非法的 `actual_num=0`，根据文档合法范围应为 `[1, 8160]`。测试原本希望编译
成功后在 NPU 上触发 `RuntimeError`，但当前 `InjectTmpBuffer` 在编译阶段先检查到空 workspace：

```text
InternalError: Expected a non-empty workspace view for Sort<half>
```

因此它们不是正常算子用例，也不是正确输入产生错误结果；但严格按 pytest 结果仍是失败，
因为异常发生得早于测试的 `pytest.raises(RuntimeError)` 代码块。解决方式有两种：

1. 若希望非法参数在编译期拒绝，应把 `tilelang.compile(...)` 一并放入 `pytest.raises`，并明确
   测试编译期参数校验；
2. 若必须验证硬件运行时异常，则需允许 `actual_num=0` 通过 workspace pass，但这会主动启动
   已知非法 kernel，可能污染后续 NPU 状态，不建议作为常规全量测试。

## 6. 测试过程中遇到的问题与解决方法

| 问题 | 处理方法 |
| --- | --- |
| 普通 pytest 只给通过/失败，不记录 kernel 性能 | 包装 `JITKernel.__call__`，使用同一 NPU 流上的 Event 计时 |
| pytest 总耗时包含 JIT、数据生成和断言 | 分开保存 `host_ms` 与 `device_ms`，性能 CSV 只使用设备时间 |
| 一个 abort 可能终止整套测试 | 每个测试文件使用独立子进程，并保存退出码与日志 |
| 参数化用例很多，同步计时明显变慢 | 使用后台运行、增量 manifest；最长文件 elementwise 约 18 分钟 |
| 一个 pytest 用例可能调用多个 kernel | 保存全部 Event 样本，并按 nodeid 汇总 calls/min/max/median |
| 纯编译或 IR 测试没有设备调用 | 保留正确性结论，但性能列为空，不用 pytest 时间伪造性能 |
| SSH 大文件传输及服务器 GitHub 访问不稳定 | 测试结果直接写入 `/root` 持久化目录，脚本先提交到独立 GitHub 分支 |
| 普通 Python 启动因环境 `.pth` 初始化偶发卡顿 | 结果汇总使用 `python -S`，只依赖标准库读取 JSON/生成 CSV |
| Exception Dump 无法编译 | 定位为 CANN 9.0.0 缺少所需 dump API，单独归类而非算子数值错误 |

## 7. 输出文件

服务器结果目录：

```text
/root/tilelang-ascend/benchmark/results/full_npu_perf_20261109/
```

主要文件：

- `manifest.json`：58 个测试文件的退出状态、耗时、JSON 和日志路径；
- `test_classification.csv`：2,502 个用例的分类和错误摘要；
- `kernel_performance.csv`：1,931 个正确用例、1,977 个 NPU 时延样本；
- `*.json`：逐测试文件的结构化原始结果；
- `*.log`：逐测试文件的完整 pytest 输出和错误堆栈；
- `full_npu_perf_20261109.master.log`：全量运行顺序和进度。

采集与汇总工具位于 GitHub 独立分支：

```text
Kernel-OPT-Cross/tilelang-ascend: codex/full-npu-performance-report
```

## 8. 限制和后续建议

- 本轮完成的是全量正确性筛查及一次调用级性能采样，不是每个算子的严谨稳定性 benchmark。
- 对需要正式汇报的代表性算子，应再执行统一 warmup、重复迭代、多 sample，并与
  AscendC、PTO 或 torch_npu 基线进行比较。
- 优先专项分析 `pow_slice`，其次是 GEMM/MMA、Sort、Transpose 和 Reduce。
- Exception Dump 需要匹配更高版本 CANN，或单独实现 CANN 9.0.0 兼容层。
- Sort 非法输入测试建议改为明确的编译期拒绝测试，避免故意触发 NPU 硬件异常。

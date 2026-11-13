# Full NPU test result recovery

The original Ascend server and its per-file JSON and pytest logs were released before those raw
files were copied locally. The original stdout/stderr, complete stack traces, timestamps, process
exit records, and `manifest.json` therefore cannot be recovered byte-for-byte.

The `reconstructed/` directory is generated only from the surviving, previously committed files:

- `test_classification.csv`: 2,502 pytest case classifications, durations, failure phases, failure
  summaries, and original log paths;
- `kernel_performance.csv`: 1,931 pytest cases with 1,977 NPU Event timing samples;
- `docs/full_npu_correctness_performance_report_20261109.md`: environment and methodology.

Every recovered `.log` and `.json` is explicitly marked as reconstructed. They must not be presented
as original pytest output. Information absent from the surviving sources is recorded as unavailable
instead of being invented.

To reproduce the recovered files:

```bash
node benchmark/reconstruct_full_npu_results.mjs
```

The recovered summaries retain the information needed for the result table and technical report:
test node IDs, categories, call durations, failure stages and summaries, per-case NPU timing samples,
and the mapping to the original per-file log names.

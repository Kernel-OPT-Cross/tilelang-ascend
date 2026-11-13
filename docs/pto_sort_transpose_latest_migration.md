# PTO Sort/Transpose latest-base migration

## Scope

The PTO Sort non-aligned-tail and Transpose `int64` fix was migrated from commit
`b110ed454c419bb8a9f8d2e7ea516a2207f58b19c` onto upstream `ascendc_pto` commit
`4805777d691b0c586e647fc0f975fa4cc306c00e`.

The only cherry-pick conflict was in `GetTypeLen`: upstream had added an explicit `int4b_t`
rejection while the fix added 8-byte integer sizes. The migrated code preserves both behaviors.
No recovered performance data was rebased or relabeled; the existing results remain tied to their
original `a6de467c` code baseline and CANN 9.0.0 environment.

## Validation performed

- `git diff --check origin/ascendc_pto...HEAD`: passed.
- Python syntax compilation for the Sort and Transpose test modules: passed on Python 3.12.4.
- Conflict-marker and patch review: passed.
- `pytest --collect-only` was attempted but could not initialize this Windows checkout because the
  compiled TVM Python module is not installed (`ModuleNotFoundError: No module named 'tvm'`).
- `clang-format` is not installed in this local environment.

## Required Ascend validation

This migration has not been compiled or executed on Ascend hardware. Before merging, run at least:

- PTO Sort source-tail checks for `actual_num=131` and `actual_num=100`;
- PTO Transpose `int64` checks for `16 x 16` and `32 x 32`;
- the complete Sort and Transpose test files on a supported Ascend NPU/CANN environment.

The B64 scalar fallback is correctness-oriented. Additional coverage for `uint64`, rectangular
shapes, and valid-shape/slice cases is recommended before treating it as a general B64 transpose
implementation.

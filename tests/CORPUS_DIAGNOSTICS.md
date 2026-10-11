# Compact corpus diagnostics

A final digest is an efficient oracle for a large captured domain, but provides
little help locating a failure. Keep its coverage and provenance; add a few
established checkpoints rather than materializing millions of output words.

## Dense unary domains

`unary_checkpoints.h` contains 88 cumulative word-wise FNV checkpoints, one per
2^20 inputs, for the RCP/SQRT/RSQ/EXP/LOG domains in `unary_exact_hardware.h`.
The vector unary tests, pseudo-scalar tests and RCP_IFLAG tests share these
checkpoints and report the first mismatching operation/input-index interval.
The original final hardware digests remain checked as well.

The fixture is 2,933 source bytes. Its rectangular array occupies 1,280 bytes
per emitted copy (704 bytes of actual checkpoint values plus zero padding).
It adds no API invocations and only one hash assertion per million results.
It localizes a mismatch to a block, not to a single lane or input. Existing
literal/modifier tests provide finer diagnostics for their covered cases.

The checkpoints were generated directly from rocjitsu, not from GoC. The offline
`tools/generate_unary_checkpoints.cpp` calls rocjitsu's unchanged C++20 headers
without linking GoC. It verifies all five resulting full-domain hashes against
the retained hardware hashes **before emitting anything**. This preserves the
established model/capture relationship instead of blessing a current GoC result.
It performs no hardware discovery or numerical model changes.

Source: `TheRock/rocm-systems`, commit
`ffc144c564c450b7847ae0528410dc63af9725a8`, with the five source headers clean.
All paths below are under `emulation/rocjitsu/lib/util/include/util/`:

| Header | SHA-256 |
| --- | --- |
| `amdgpu_rcp.h` | `5538062de610792c017cef63c093ef488b0bedeee4dd676e823e7241c81522fd` |
| `amdgpu_sqrt.h` | `050dada1c764eb70533f1eb25ca3e0eaa623c09ef04840575bb59d1707d4c7d9` |
| `amdgpu_rsq.h` | `378d98edb7ee4292c4875b302eb650e123276bc819aee5be5c22d82e1155d28b` |
| `amdgpu_exp.h` | `ee4434584bc5579f1403722acb6b2311b0973263ddb04d1ea928b2a165d739d9` |
| `amdgpu_log.h` | `74c0052079da6dd20c5e0a7aed4dbeff81ec98f377a9c9ba9924c505e68977fe` |

To reproduce from the GoC source directory, set `ROCJITSU_UTIL_INCLUDE` to that
checkout's `emulation/rocjitsu/lib/util/include` directory, then:

```sh
clang++ -std=c++20 -O2 -I tests -I "$ROCJITSU_UTIL_INCLUDE" \
  tools/generate_unary_checkpoints.cpp -o /tmp/goc-generate-unary-checkpoints
/tmp/goc-generate-unary-checkpoints > /tmp/goc-unary-checkpoints.h
clang-format -i /tmp/goc-unary-checkpoints.h
cmp /tmp/goc-unary-checkpoints.h tests/unary_checkpoints.h
```

C++20 is required only for this optional upstream-facing generator. GoC and its
normal tests remain C++17 and require neither rocjitsu nor a GPU.

Validation: all seven affected tests pass with Clang and GCC. An out-of-tree
fault-injection copy changed the third RCP checkpoint by one bit; the test failed
at the expected input-index interval `[2097152, 3145728)`. This verifies the
localization path without changing production code or committed fixture values.
Artifacts: `goc-build/coverage-work/testing-checkpoints/`; GCC results:
`goc-build-gcc/coverage-work/testing-checkpoints-tests.log`.

## Remaining review

Other digest assertions often already operate per opcode/modifier row, or have
per-output comparisons alongside them. Their diagnostic scope still needs a
family-by-family review with the reference-evidence inventory. These unary
checkpoints improve the large dense domains; they do not establish that every
compact corpus now has adequate localization.

## Half RCP/RSQ/SQRT

`half_sfu_checkpoints.h` contains 48 cumulative word-wise hashes (384 bytes),
one after each 4096 ascending half encodings for each operation. The independent
`tools/generate_half_sfu_checkpoints.cpp` calls unchanged rocjitsu FP32 models
and compiler `_Float16` conversions, linking no GoC implementation or reference
helper. Clang 21.1.8 and GCC 15.2.0 produce identical fixtures. Generate with C++20,
`-O2`, and `TheRock/rocm-systems/emulation/rocjitsu/lib/util/include` on the include
path, then clang-format its stdout. Ordinary tests require only C++17.

Provenance: rocjitsu's containing rocm-systems commit
`ffc144c564c450b7847ae0528410dc63af9725a8`. Header SHA-256 values:

| Header | SHA-256 |
| --- | --- |
| `amdgpu_rcp.h` | `5538062de610792c017cef63c093ef488b0bedeee4dd676e823e7241c81522fd` |
| `amdgpu_rsq.h` | `378d98edb7ee4292c4875b302eb650e123276bc819aee5be5c22d82e1155d28b` |
| `amdgpu_sqrt.h` | `050dada1c764eb70533f1eb25ca3e0eaa623c09ef04840575bb59d1707d4c7d9` |

The scope is unmodified inputs, preserved denormals, quiet NaNs and no saturation.
These are borrowed-model regression fixtures, not new hardware captures. The
upstream FP32 models' hardware validation is described in the existing unary
corpus provenance. Both architectures, widths and all CPU selectors consume the
same fixture; policy/NaN/saturation/modifier witnesses remain separate tests.
The first failed prefix reports operation, architecture, width, CPU and the
4096-encoding interval. Successful generation alone does not validate additional
flags or policies. Local generator binaries and measurements are under
`goc-build/coverage-work/testing-half-sfu/`.

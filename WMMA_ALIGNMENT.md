# RDNA WMMA alignment with rocjitsu

Target: rocjitsu branch `users/kuhar/rdna-wmma-fixes`, commit
`ad2bb973ec4a3d254e705c5a8f8cc0d17bc1d034`. GoC baseline:
`4edcaca2d37b129de2d9120c36d4c5a0d2c66bdf`. Implementation and fixture edits
are confined to GoC; the upstream checkout is unchanged.

## Changes and evidence

| Area | Result | New supported captures | Excluded raw captures |
|---|---|---:|---:|
| Dense integer, RDNA3/RDNA4 | Fixed RDNA3 saturation after IU8 pairs / IU4 quartets in scalar and SIMD; RDNA4 already aligned | 10 | 0 |
| Sparse integer, RDNA4 | Existing physical rearrangement matches all new captures | 78 | 0 |
| Dense FP8/BF8, RDNA4 | Borrowed native DOT8 exact arithmetic; accepted C ABS/NEG now ignored in both semantics | 268 | 0 |
| Sparse FP8/BF8, RDNA4 | Borrowed two contiguous native DOT8 stages | 296 | 48 |
| Sparse FP16/BF16, RDNA4 | Borrowed four DOT4 stages; packed loose scalar/AVX2/AVX-512 now narrow each stage | 488 | 24 |
| Dense FP16/BF16, RDNA3/RDNA4 | Existing arithmetic matches expanded captures, including packed halves and aliases | 1,618 | 0 |
| **Total** | 56 public entry points reviewed | **2,758** | **72** |

Supported capture counts by wave size:

| Family / architecture | Wave32 | Wave64 | Replay semantics |
|---|---:|---:|---|
| Dense integer, RDNA3 | 2 | 2 | Loose and exact |
| Dense integer, RDNA4 | 3 | 3 | Loose and exact |
| Sparse integer, RDNA4 | 30 | 48 | Loose and exact |
| Dense FP8/BF8, RDNA4 | 134 | 134 | Exact |
| Sparse FP8/BF8, RDNA4 | 144 | 152 | Exact |
| Sparse FP16/BF16, RDNA4 | 180 | 308 | Exact |
| Dense FP16/BF16, RDNA3 | 487 | 487 | Exact |
| Dense FP16/BF16, RDNA4 | 322 | 322 | Exact |

Counts are distinct stored matrix/operand cases, not output words, CPU-selector
repetitions, host configurations or aliases. Integer captures run in both
semantics. New floating captures run in strict empirical exact semantics;
existing independent fuzzy tests cover loose arithmetic. Source and destination
storage is compared, including guards and untouched halves. Existing captures
were retained; no conflicting golden was replaced with a model-generated answer.

Sparse exclusions are unsupported raw sign/C-modifier/clamp encodings. Decoder
aliases for integer selector bits beyond GoC's legal key range are also not
replayed; none of the 78 stored sparse integer cases is excluded. GoC has no
instruction decoder or inline-selector API; tests materialize inline constants
in the appropriate format. Explicit failure-preservation tests retain rejection
of unsupported flags and reserved semantics.

## Arithmetic and state contracts

Exact models are ports of the four pinned upstream commits, not newly
reverse-engineered numerical models. Native FP8/BF8 alignment differs from
widening to FP16; named regressions show native subnormal input handling,
ignored C modifiers, and Wave64 infinity/NaN stage precedence. Integer saturation
and packed sparse midpoint/overflow tests distinguish the corrected behavior
from the former final-only implementations.

Floating matrix APIs accept/ignore guest denormal and IEEE flags in both
semantics. Packed FP16 still honors finite-overflow saturation. The saved MODE
values have equal F32 and F16/F64 denormal policies, checked by the adapter.
Rounding selectors have no API counterpart and are ignored by the borrowed
matrix model. All supported floating captures are replayed under four host
rounding modes, FTZ/DAZ off/on where available, and every available CPU selector;
results and host control bits are checked. IEEE-on is an additional model
invariance test, not a new hardware observation. Production does not explicitly
write the CPU FP environment; ordinary sticky exception flags remain allowed.

No EXEC or implicit-state output parameters were added. Shared DOT2/DOT4
interfaces and their existing optional exception-reporting contracts remain.
Loose SIMD dispatch retains supported modifiers and keys. The packed loose
reference now models FP32 accumulation and independent mathematical narrowing
at each DOT4 boundary. Primary and holdout seeds pass unchanged tolerances on
Clang and GCC; packed results have zero measured error in that bounded corpus.
See [calibration details](tests/LOOSE_TOLERANCES.md).

## Plan completion evidence

| Plan item | Implementation / qualification | Tests and records |
|---|---|---|
| 1. Pin and import evidence | AMD/MIT fixtures imported from `ad2bb973ec4`; explicit local input reconstruction | New `rdna_*_cases.h`, captured matrix/operand tests; no upstream runtime dependency |
| 2. RDNA3 integer saturation | `e2da519`; scalar performance follow-up `1a61ec6` specializes clamp outside the reduction | Format-group, signedness, wrap, alias and captured saturation tests |
| 3. RDNA4 integer qualification | `fc7c1bd`; no production correction needed | 78 sparse and six RDNA4 dense captures, existing independent corpora retained |
| 4. Native dense FP8/BF8 | `4f11187`, readable witnesses in `e7de3d8` | 268 captured cases, NaN/infinity/Wave64 and native-subnormal/C-alias witnesses |
| 5. Sparse floating and packed loose | `78a206e`, `57bdeae`, `cde58d8` | 784 supported captured cases; staged numeric reference, holdout, midpoint and overflow tests |
| 6. Dense half / guest state | `c9f8578` | 1,618 captured cases, packed halves/aliases, host controls and translated guest flags |
| 7. Documentation and final validation | This report, public comments, `PLAN.md`, architecture/reference reviews, tolerance and testing audits | Validation and performance results below; compiled API inventory |

## Validation and performance

All four final configurations pass **1,224 CTests each**: Clang 21.1.8 Release,
GCC 15.2 Release, scalar-only Clang Release, and Clang Debug with ASan/UBSan.
The suites include benchmark smoke/output and embedding/export checks; all
standalone public C/C++ header compile checks also pass. Final GCC, scalar-only
and sanitizer runs took 21.64 s, 23.93 s and 151.93 s, respectively.
Fourteen Python tooling tests pass. The rebuilt compiled-reference inventory
contains **1,791 instructions, 163 active test objects, zero missing family
references, and zero unreviewed family files**. This inventory establishes
linkage/review records; the captures establish the numerical evidence.

Performance comparison uses Clang Release (`-O3`, C++17), Ryzen 7950X3D,
CPU affinity 8, and the existing matrix benchmark bodies extracted into small
standalone drivers. Baseline and final production revisions are `4edcaca` and
`1a61ec6`. Each row takes the median of seven samples of at least 10 ms, with
three alternating before/after process pairs; reported ratios use each row's
median across those three runs. Compilation and tests were idle during timing.
The sibling core was not reserved; these are local observations, not universal
performance guarantees. The sampled set has 109 loose rows and is not an
exhaustive performance sweep of every wave size, modifier or input distribution.

| Sampled loose rows | Count | Median time change | Mean time change |
|---|---:|---:|---:|
| All | 109 | −0.15% | +3.15% |
| SIMD only | 61 | −1.88% | −1.21% |
| Packed sparse scalar | 4 | +47.18% | +46.05% |
| Packed sparse SIMD | 8 | +2.40% | +3.42% |

Positive means slower. Representative corrected packed paths:

| Instruction / path | Flags | Before ns/wave | After ns/wave | Change |
|---|---|---:|---:|---:|
| Sparse FP16 / scalar | none | 8,116.0 | 12,247.8 | +50.9% |
| Sparse FP16 / AVX2 | none | 1,706.8 | 1,885.5 | +10.5% |
| Sparse FP16 / AVX-512 | none | 1,682.1 | 1,697.3 | +0.9% |
| Sparse BF16 / scalar | none | 7,030.2 | 9,758.2 | +38.8% |
| Sparse BF16 / AVX2 | none | 619.3 | 635.2 | +2.6% |
| Sparse BF16 / AVX-512 | none | 602.6 | 584.4 | −3.0% |

Supported NEG/key variants retain SIMD; their costs are comparable (BF16
AVX-512 varies from −3.0% to +2.9% across the two sampled flag settings).
Packed paths now perform required intermediate narrowing rather than the old
final-only approximation. The extra scalar cost is disclosed, not hidden by
switching to exact execution. Dense FP8 C-flag rows likewise compare the old
transformed-C behavior with the corrected ignored aliases.

The first run exposed a 3.4× unintended slowdown of unclamped scalar RDNA3 IU4.
`1a61ec6` restores the old vectorizable reduction by specializing saturation
outside the loop; Wave64 returned to approximately 267 ns/wave. Smaller scalar
fluctuations need caution: a sampled RDNA3 FP16 row varied by about 17% despite
its object disassembly being identical except for the accepted-flag immediate.
Do not interpret the small aggregate improvements as established speedups.

Local evidence is under the existing build directories: `alignment-*-ctest.log`,
`alignment-reference-inventory.json`, `alignment-final-perf-{before,fixed}-*.csv`,
and `alignment-performance-summary.json`. The generated comparison sources are
`goc-build/alignment-{before,after}-benchmark.cpp`; the baseline source archive
and build remain under `goc-build/alignment-baseline/`. No benchmark-only edits
were made to either source revision.

## Limits

Fixtures retain upstream device/code/MODE/seed/expected-word metadata and AMD
copyright/MIT provenance. Inputs use explicitly defined unsigned xorshift
reconstruction; tests have no upstream runtime dependency. Compact captures
came from Radeon Pro W7900 (`gfx1100`) and Radeon AI Pro R9700 (`gfx1201`)
on 2026-10-10 using TheRock `10.1.0a20260822`, as recorded in the pinned
upstream `tests/fixtures/float_dot/README.md`. We did not replay the unavailable larger archive,
claim its billion-word totals, validate physical RDNA3.5 hardware, or establish
universal correctness outside the borrowed empirical models' tested domains.

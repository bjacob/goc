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

## Validation and performance

Final cross-build validation and before/after performance measurements are in
progress. Do not treat this section as a completed acceptance gate yet.

## Limits

Fixtures retain upstream device/code/MODE/seed/expected-word metadata and AMD
copyright/MIT provenance. Inputs use explicitly defined unsigned xorshift
reconstruction; tests have no upstream runtime dependency. Compact captures
came from Radeon Pro W7900 (`gfx1100`) and Radeon AI Pro R9700 (`gfx1201`)
on 2026-10-10 using TheRock `10.1.0a20260822`, as recorded in the pinned
upstream `tests/fixtures/float_dot/README.md`. We did not replay the unavailable larger archive,
claim its billion-word totals, validate physical RDNA3.5 hardware, or establish
universal correctness outside the borrowed empirical models' tested domains.

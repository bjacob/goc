# RDNA3/RDNA4 coverage worklist

The goal is complete arithmetic/register-operation coverage for both architectures,
with shared APIs where semantics agree and `_rdna4` variants where they differ.
The [generated inventory](ISA_INVENTORY.md) lists every canonical mnemonic in AMD's
RDNA3 and RDNA4 XML, including excluded operations and unresolved scope boundaries.
Regenerate it with `python3 tools/isa_inventory.py <machine-readable-isa/isa>`.

An API declaration is not a completed audit. Existing RDNA4 tests do not prove
RDNA3 compatibility. Equal XML descriptions do not prove identical arithmetic.
A family is complete only after reviewing its instruction results, modifiers,
wave widths, aliasing, implicit register outputs, and numerical semantics against
both targets. Exact support is limited to ports of existing rocjitsu implementations; new hardware
reverse-engineering is out of scope. Unsupported exact/global-state requests must
fail honestly under the documented contracts. Loose SIMD
paths must retain their existing performance and may skip optional state outputs.

The [cross-architecture review queue](ARCHITECTURE_AUDIT.md) now identifies a
confirmed broad gap: ordinary RDNA3 OMOD/IEEE policies differ from the RDNA4
policies currently used by many unsuffixed floating APIs. Those names must not
be treated as verified shared semantics. Width completeness does not close this gap.

## Evidence and starting point

- Start: `80470bd`, 682 instruction APIs, 674 tests passing with Clang and GCC.
- ISA and rocjitsu source: local TheRock `rocm-systems` revision `ffc144c564c`.
  XML content fingerprints are recorded in the inventory.
- Borrowed hardware captures establish only the targets and cases they capture.
  New RX 9070 Wave32/Wave64 DS_SWIZZLE captures are recorded below.
- Inventory initially finds 804 in-scope canonical mnemonics in the union:
  664 have matching API names; 140 lack them. Another 14 control/register
  boundaries require explicit review. 578 memory/control/scheduling mnemonics
  are excluded. These counts are neither coverage percentages nor correctness claims.

## Cross-cutting completion gates

- [x] Enumerate both XML instruction sets and identify matching public API names.
- [x] Resolve the initial boundary rows: relative addressing and hardware-register
  access remain in scope; pipeline no-ops/flushes are scheduling exclusions.
- [ ] Map XML aliases and dual-issue operations; distinguish assembly aliases
  from missing semantics and define simultaneous-read behavior for paired outputs.
- [ ] Review every shared mnemonic for architectural differences, including
  special values, rounding, denormals, exceptions, modifiers, and implicit state.
- [ ] Fill every in-scope missing API or record a concrete implementation blocker.
- [ ] Borrow all applicable rocjitsu empirical models and captures; document any
  unsupported exact behavior and optional global-state reporting.
- [ ] Audit Wave32/Wave64 contracts, especially comparisons, lane routing, and EXEC.
- [ ] Preserve existing SIMD dispatch; compare performance for affected hot paths.
- [ ] Validate complete Clang/GCC suites and targeted ASan/UBSan tests.
- [ ] Final requirement-by-requirement audit; unresolved items prevent completion.

## Family checklist

| Family | Current evidence | Remaining work |
|---|---|---|
| RDNA3 WMMA | All six mnemonics, both widths, exact rocjitsu GFX11 model; `wmma_replicated_exact_test.cpp`; 103 DOT2 captures and 28 packed matrices | Scalar implementations; verify inventory integration as other APIs evolve |
| RDNA4 WMMA/SWMMAC | Existing layouts, SIMD paths, exact FP16/BF16 tests | Audit current rocjitsu against existing implementation; FP8 exact availability and state |
| DOT2 and DOT2ACC | FP32 DOT2 split into unsuffixed RDNA3 / `_rdna4`; shared modifiers/DPP/loose SIMD; GFX1100 captures, aliases and host-state tests | DOT2ACC and dual forms reuse the same models and SIMD; RDNA3 reporting remains unsupported; ordinary forms support both wave widths |
| FP32 unary | Ten floating-result operations plus FREXP_EXP retain rocjitsu-derived numerical models; FRACT is loose-only. Architecture policies are split; new inferred reporting was removed | Finish modifier review and upstream-port availability review for FP16/FP64; unavailable exact reporting is an explicit limitation, not a reverse-engineering task |
| Floating min/max/median | FP32 ternary family reviewed: separate RDNA3 IEEE/OMOD rules and RDNA4 NUM/propagating forms, all guest policies/modifiers and both widths on scalar/AVX2 | Review remaining binary, FP16/FP64 and packed contracts; strict exact and optional reporting remain unsupported |
| Scalar integer arithmetic | All 29 canonical names in `scalar_integer.cpp` reviewed across ISA descriptions, rocjitsu handlers and existing fixtures; shared results/SCC, seven renamed aliases, RDNA4-only 64-bit forms; carry/borrow-chain and immediate tests | Scalar bitwise/field, comparisons and register/control families have separate review records or remain pending |
| Scalar/register moves | MOV/CMOV, MOVK/CMOVK, SAVEEXEC/WREXEC, relative moves, GETPC and round/denorm MODE updates; scalar-relative RX 9070 captures | General GETREG/SETREG access and continued semantic audit |
| Vector/register routing | Lane transfers, MOV/MOVREL, SWAP/SWAPREL, PERMLANE64 and DS permutations at both wave sizes; RX 9070 half-move, SWAPREL and DS_SWIZZLE captures | Broader cross-width/modifier audit; RDNA3 hardware confirmation |
| Comparisons | Existing RDNA4 results/SCC/EXEC and exception tests; integer relational predicates now have Wave64 scalar/SIMD paths; RDNA3 true/false results implemented at both wave widths | RDNA3 floating constant exception reporting and architecture-specific floating-policy audit |
| Basic floating arithmetic, FMA, conversions | Existing RDNA4 implementations and captures; confirmed OMOD/IEEE architecture differences | Split differing APIs and implement RDNA3 guest-mode policies; see ARCHITECTURE_AUDIT.md |
| Transcendentals, reciprocal, division | FP32 RCP/SQRT/RSQ/EXP/LOG exact captured mappings on both architectures; other partial exact coverage | Review newer rocjitsu models/captures, differing semantics and state outputs |
| Integer/vector arithmetic and bit operations | Existing broad RDNA4 coverage; bit-count family has exact scalar/SIMD paths and DPP at both wave sizes | Check RDNA3 spellings, modifier encodings, widths, carry outputs and all aliases |
| Interpolation | All six APIs at both wave widths; shared quad routing checked against both manuals; RDNA4 captures and independent arithmetic tests | Compare remaining RDNA3 numerical contracts and empirical behavior |
| Dual-issue VOPD | All 19 canonical operations; shared ordinary kernels, architecture-split floating arithmetic/DOT2ACC; Wave32 only | Direct VOPD hardware captures; inherited exact/global-state gaps |

Keep this checklist conservative: unchecked review work is not made complete by
a green test suite that exercises only the current implementation.

## Validated increments

- DOT2 architectural split: all 677 tests pass in Clang 21 and GCC 15 Release
  builds; all 10 DOT2 tests pass under Clang ASan/UBSan. The new API tests cover
  GFX1100 result captures, all sign/half selectors with DPP8/DPP16, EXEC masks,
  aliases and guards, all host rounding modes, and an explicit RDNA3/RDNA4
  result difference. RDNA3 optional exact exception reporting remains unsupported.
  Existing RDNA4 tests and benchmark rows now use the suffixed names. The loose
  SIMD kernel and its dispatch conditions are unchanged; no performance gain is
  claimed for this increment.
- Scalar RDNA3 spellings/immediates: 19 APIs added, reusing the existing scalar
  integer and comparison implementations. RDNA4 XML explicitly aliases all seven
  arithmetic spellings; rocjitsu's SOPK implementation confirms signed versus
  unsigned immediate extension for the twelve comparisons. Tests exhaust every
  16-bit immediate against boundary/equal operands and independently check carry,
  borrow, overflow, aliases and error atomicity. All 680 tests pass with Clang
  and GCC; the three new tests pass under ASan/UBSan. Missing in-scope API names
  are now 121; semantic audits remain distinct from name coverage.
- Basic scalar moves: six shared RDNA3/RDNA4 entry points added from rocjitsu's
  register-transfer contracts. Tests exhaust all signed immediates, every raw
  bit position in B32/B64, conditional preservation and aliases, invalid flags,
  and host FP-state preservation. All 683 tests pass with Clang and GCC; all
  three new move tests pass under ASan/UBSan. Missing in-scope API names: 115.
- Lane transfers: READLANE, WRITELANE and READFIRSTLANE implemented for both wave
  widths, using shared RDNA3/RDNA4 contracts from rocjitsu execution and register
  access. Explicit indices wrap; READFIRSTLANE selects lane 0 for zero EXEC.
  Tests cover every lane, wrapped/high-bit indices, single-bit and suffix EXEC
  masks, scalar output overlap and invalid flags. All 686 tests pass on Clang
  and GCC; the three new tests pass under ASan/UBSan. Missing API names: 112.
- Scalar EXEC operations: all 24 canonical SAVEEXEC/WREXEC B32/B64 APIs added,
  sharing validation and stores while retaining explicit Boolean expressions.
  The contracts follow rocjitsu, including B64 access to raw EXEC_HI independently
  of active wave width. Independent per-bit truth tables verify all operations,
  old/new destination distinction, zero EXEC, optional EXEC reporting and output
  alias priority. All 688 tests pass on Clang/GCC; both new tests pass under
  ASan/UBSan. Missing in-scope API names: 88.
- Constant integer comparisons: sixteen RDNA3 CMP/CMPX F/T forms added for
  I32/U32/I64/U64 operands in Wave32. Predicate evaluation does not read operands;
  true results still apply DPP output filtering. Independent DPP references check
  all new forms, CPU selections, semantics, masks and validation. All 689 tests
  pass on Clang/GCC; the new test passes under ASan/UBSan. Missing API names: 72.
  Floating F/T exception behavior and Wave64 comparison APIs remain open.

## Scope review after initial implementation increments

The functional-group labels in AMD's XML are not sufficient to decide scope.
The generator now honors explicit branch/termination flags even within SALU.
`S_CALL_B64`, `S_SETPC_B64`, `S_SWAPPC_B64`, and `S_CODE_END` are excluded as
control flow/traps. RDNA3 `LDS_DIRECT_LOAD` and `LDS_PARAM_LOAD` are labeled VALU
but their descriptions explicitly read LDS memory; they are excluded as memory
operations. `S_ALLOC_VGPR` allocates wave resources, `S_SLEEP_VAR` schedules sleep,
and `V_NOP`/`V_PIPEFLUSH` concern pipeline scheduling; these are excluded.

All twelve initially ambiguous relative-register and GETREG/SETREG operations
remain **in scope**. So do `S_GETPC_B64` (reads PC without branching),
`S_ROUND_MODE`, and `S_DENORM_MODE` (modify guest FP state, never host FP state).
This is not permission to implement guest mode updates with host fenv mutations.

All 1,396 canonical names remain listed. The revised scope contains 808 names:
732 have APIs and **76 are missing**. The 588 exclusions are explicit. The rise
from 72 missing names reflects twelve register operations admitted from the
review list and eight former in-scope exclusions; two other exclusions came
from the review list. No implementation was removed and no unresolved register
operation was silently excluded. Scope checks verified each reclassification.

## Floating min/max review in progress

At rocjitsu revision `ffc144c564c`, generated FP32 MIN/MAX, MIN3/MAX3 and
MINMAX/MAXMIN execution bodies for RDNA3 spellings and RDNA4 NUM spellings are
identical after whitespace normalization. MED3's wrappers differ in SIMD
availability; FP64 MIN/MAX have different first available encodings, requiring
like-for-like VOP3 review. FP16 and packed forms use other generated execution
paths and were not covered by that text comparison.

This supports sharing **loose** arithmetic, not a bit-exact equivalence claim.
Before adding these names, review NaNs (especially signaling NaNs), zero ties,
operand ordering in median networks, guest FP flags, output modifiers and
exception updates. Existing GoC loose/SIMD tests must remain valid, but cannot
substitute for RDNA3 hardware evidence. No min/max API was added by this audit.
- RDNA3 DOT2ACC: the VOP2 accumulator form now reuses the GFX11 exact DOT2 model
  and existing loose SIMD implementation, with D as the read/write accumulator.
  Only DPP routing is accepted; VOP3P sign/half controls are rejected as absent
  from its encoding. Captured results, DPP aliases, masks, loose SIMD and optional
  reporting contracts are tested. All 691 tests pass on Clang/GCC and all five
  architectural DOT2 tests pass under ASan/UBSan. Missing API names: 75.

Relative VGPR move range rules were checked against section 3.3.2.2 and the
VGPR-indexing tables of AMD's [RDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
and [RDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).
Both specify VGPR0 substitution for out-of-range VALU sources and discarded
out-of-range destination writes. Both specify an index limit of 255; rocjitsu
currently checks 1023. GoC follows the published 255 rule and records this
upstream discrepancy rather than treating the upstream code as hardware evidence.
The split instruction still extracts the documented 10-bit M0 fields before
range checking. No new hardware capture establishes behavior in that disputed
range. API views begin at VGPR0 with explicit base indices and bounds so fallback
reads are possible; 64-bit intermediate sums prevent accidental index wrapping.

- Wave32 vector-relative moves: four shared APIs implemented with bounded full
  register-file views, explicit base indices, VGPR0 fallback and DPP routing.
  Tests cover aliases, masks, empty ranges, split offsets, the documented index
  limit and widened address calculations. All 694 tests pass on Clang/GCC;
  all three relative-move tests also pass under ASan/UBSan. Missing API names: 71.
  Scalar-relative operations and Wave64 vector-relative variants remain open.

- RDNA3 DX9 FMAC: `goc_v_fmac_dx9_zero_f32` reuses DX9 FMA's scalar,
  x86-64-v3 and x86-64-v4 paths with D as the accumulator. rocjitsu's VOP2/VOP3
  execution at `ffc144c564c` confirms the accumulator is unmodified by source
  modifiers; A/B ABS/NEG, OMOD, CLAMP and DPP are supported. C modifiers are
  rejected. Tests cover all valid modifier combinations with DPP, aliases and
  masks, plus zero/NaN/infinity/subnormal cases and error preservation. The
  existing FMA hardware corpus validates the reused arithmetic on RDNA4;
  this is not a new RDNA3 FMAC hardware capture or a bit-exactness claim.
  Strict exact semantics remain unsupported. All 696 tests pass on Clang/GCC;
  the two new tests pass under ASan/UBSan. Missing API names: 70.

## Wide integer MAD architecture review

AMD's XML identifies RDNA3 `v_mad_u64_u32` as the alias of RDNA4
`v_mad_co_u64_u32`, with the same opcode, operands and unsigned carry contract.
rocjitsu at `ffc144c564c` has equivalent scalar arithmetic and saturation for
both. The new RDNA3 entry point shares GoC's implementation, including v3/v4
SIMD, exact integer arithmetic and full input/output alias handling. All six
wide-MAD tests exercise the new spelling where applicable, including an
independent 128-bit reference and the existing **RDNA4** hardware corpus.
This reuse does not turn that corpus into an RDNA3 hardware capture.

The initial signed RDNA3 review found a discrepancy (resolved by the full ISA
pseudocode below). rocjitsu's RDNA3 and RDNA4 handlers
both report signed overflow, whereas GoC's RDNA4 hardware captures establish
that the scalar output is bit 64 of the full mathematical sum (its extended
sign). These differ even without overflow: `0 * 0 + (-1)` has scalar bit 1
in the RDNA4 capture model but bit 0 in rocjitsu. XML's prose says
"overflow/carryout" and does not resolve the discrepancy. The subsequent RDNA3 pseudocode review below supplies the missing evidence;
rocjitsu's overflow result was not copied.

All 696 tests pass on Clang/GCC; all six wide-MAD tests pass under ASan/UBSan.
Missing canonical API names: 69. Wave64 MAD variants remain open.

## Register swaps and half-wave routing

`v_swap_b32` and `v_permlane64_b32` now have shared RDNA3/RDNA4 implementations
for Wave32 and Wave64. rocjitsu's two VOP1 execution files have identical swap
bodies and call the same PERMLANE64 handler. AMD's RDNA4 ISA section 16.8 confirms
that PERMLANE64 selects lane XOR 32 using destination EXEC, regardless of source
EXEC, and is a no-op in Wave32; both XML descriptions also specify that no-op.
Accordingly the Wave32 API omits EXEC and permits null operands. The Wave64
implementation snapshots both halves before any stores. Both swap operands are
read/write pointers, and whole-register aliases are supported.

Tests cover both wave sizes, full/empty/partial EXEC, lane 31/32 boundaries,
inactive sources, in-place routing, guard words and rejection of every modifier
bit. These are specification/reference tests, not new GPU captures. Scalar loops
provide exact bit operations; dedicated SIMD implementations remain open.
SWAPREL remains open because its relative-range rules need separate auditing.
The later half-register follow-up below implements SWAP_B16. Missing canonical
API names at this stage: 67.

Validation for these additions: all 700 tests pass with Clang and GCC; the four
swap/routing tests also pass under ASan/UBSan.

## DS register permutations and resolved signed MAD

Implemented DS_PERMUTE/BPERMUTE for shared Wave32, RDNA3 Wave64 and RDNA4
Wave64. AMD's RDNA3 manual section 12.5.2 explicitly restricts routing to each
32-lane half, whereas RDNA4 section 12.5.2 permits full-wave routing. The RDNA4
Wave64 APIs therefore use `_rdna4_wave64`. The RDNA3 detailed pseudocode omits
the half-wave base, but its explicit prose and rocjitsu agree on independent
halves; the implementation follows those. Both masks apply to reads and writes.
Scatter selects the highest active source on collisions, following rocjitsu and
the pseudocode, while documenting that hardware prose does not guarantee a winner.
Inputs and addresses are consumed before output stores for alias safety.

The scope audit also corrected `ds_bpermute_fi_b32`: despite its XML VMEM label,
it only routes registers and belongs in scope. Its new Wave32/Wave64 APIs read
inactive source lanes. Total in-scope canonical names are now 809, with 64 missing.
The DS_SWIZZLE follow-up below addresses rotate/FFT modes missing from
rocjitsu's simple quad/bit implementation.

The previously open signed MAD question is resolved by RDNA3 section 16.12's
explicit 65-bit concatenation: the scalar output is bit 64 of the full signed
sum, matching GoC's existing model rather than rocjitsu's signed-overflow flag.
`goc_v_mad_i64_i32` now shares the existing scalar/v3/v4 implementation. Both
signed spellings run the extended-sign witnesses, wide-reference random cases,
mask/alias tests and host-FP checks. Captures remain RDNA4-only.

Sources: AMD [RDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
(downloaded from a [mirror](https://llm-tracker.info/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
when AMD's endpoint returned 401), AMD RDNA4 ISA sections 12.5.2/16.15, and
rocjitsu `ffc144c564c` shared DS execution handlers. No new hardware captures.

Validation: all 705 tests pass with Clang and GCC. All five DS permutation tests
and all six wide-MAD tests pass under ASan/UBSan. The DS tests use an independent
destination-centric reference, explicit cross-half witnesses, byte-offset overflow,
collisions, masks, inactive reads, aliases and guard words. Dedicated DS SIMD
paths are not yet implemented; existing MAD SIMD implementations are unchanged.


## DS swizzle: all four encoding classes

Added shared Wave32/Wave64 DS_SWIZZLE_B32, including bit-mask, quad, masked
rotate and FFT modes. Basic selection follows rocjitsu; unlike its current
handler, inactive source lanes supply zero. Rotate and FFT follow the matching
pseudocode in both ISA manuals, with rotate arithmetic confined to each 32-lane
row. Source snapshots preserve in-place operations. No dedicated SIMD path yet.

The RDNA3 manual's masked-rotate examples disagree with its own pseudocode;
RDNA4's examples and pseudocode agree. This was checked on the local RX 9070
(gfx1201): all 2,048 rotate encodings, 32 FFT masks, 256 quad selectors and 32
bit-mode examples, each under eight EXEC masks in both wave sizes. All
1,818,624 output words match. This validates RDNA4, **not** RDNA3 hardware;
RDNA3 uses its explicit pseudocode pending hardware confirmation of the
inconsistent examples. No architectural difference was established that would
justify separate entry points.

`tests/capture_ds_swizzle.py` reproduces the captures, and
`tests/ds_swizzle_hardware.h` records word-wise hashes and raw capture SHA-256s.
Capture compiler: ROCm clang 23; Wave64 explicitly uses `-mwavefrontsize64`.
Additional tests cover all 65,536 offset encodings for both wave sizes against
an independent bit-by-bit/table reference, in-place routing, partial masks,
guard words, invalid modifiers and empty EXEC. Missing canonical API names: 63.

Validation: all 708 tests pass with Clang and GCC; all three DS_SWIZZLE tests
pass under ASan/UBSan. The final capture generator reproduces the Wave64 binary
capture byte-for-byte; regenerated Wave32 instruction bodies match the original
probe. Existing arithmetic/SIMD implementations were not changed.

## Half-register moves and swaps

Added shared `v_mov_b16` (Wave32, including DPP) and `v_swap_b16` (Wave32/Wave64).
MOV applies ABS then NEG directly to the selected half's sign bit. It preserves
NaN payloads and subnormals without FP conversion. RX 9070 raw-encoding probes
cover all ABS/NEG/OMOD/CLAMP combinations and half selections: ABS/NEG work,
while OMOD/CLAMP are ignored. LLVM rejects these modifier spellings even though
the hardware supports the input bits, and rocjitsu currently omits those input
modifiers. The implementation follows the captured behavior and the manual's
statement that MOV supports negation/absolute value. RDNA3 shares that documented
behavior, but the new capture is RDNA4-only.

SWAP accepts only HIGH_A/HIGH_D; both operands are read/write. Selected halves
use original values, and each write preserves the other half, including the
first write when exchanging halves of one VGPR. Tests cover all half selections,
whole-register aliases, both wave widths, masks, guard words, and invalid flags.
MOV tests also exhaust all 65,536 half bit patterns with every sign/half choice,
replay 4,096 captured output words, and cover DPP source filtering and aliases.
`capture_half_move.py` reproduces the raw VOP3 probe; the fixture includes its
SHA-256 and word-wise digest. The regenerated probe source matches the captured
source byte-for-byte.

Wave64 MOV, dedicated half-move/swap SIMD paths, and RDNA3 hardware validation
remain open. Missing canonical API names: 61.

Validation: all 712 tests pass on Clang/GCC; all ten tests in the move/swap
binaries pass under ASan/UBSan. Existing SIMD implementations are unchanged.

## Guest MODE updates and GETPC

Added shared S_ROUND_MODE/S_DENORM_MODE with an optional trailing `mode`
register pointer. Both manuals' scalar-control table specifies immediate[3:0];
the MODE layout puts round control at [3:0] and denormal control at [7:4].
The implementation updates only those bits in the supplied guest register,
never the host environment. Null opts out after validation. Both loose and
exact modes perform the inexpensive update when requested. The rocjitsu
handlers at the audited revision are empty, so no behavior was borrowed from
those stubs.

S_GETPC_B64 now returns PC+4 in the ordinary scalar destination, following both
manuals and rocjitsu's next-instruction address calculation. It has no EXEC
parameter and no implicit PC output because it does not write PC. Integer
addition and the existing scalar-move implementation preserve host FP state.

Tests exhaust all 65,536 immediates in both semantics modes, check preservation
of unrelated bits, null opt-out, validation and error preservation, host rounding,
sticky exceptions and x86 MXCSR. GETPC covers ordinary addresses, carry/wrap,
and input/output aliasing. No new GPU capture was required for these integer
register transformations. Missing canonical API names: 58.

Validation: all 716 tests pass with Clang/GCC; all four guest-state tests pass
under ASan/UBSan. Public standalone-header and exhaustive instruction-flag checks
include the new signatures. Existing numerical/SIMD implementations are unchanged.

## RDNA3 floating min/max spellings

Added the 18 RDNA3 binary, ternary and packed floating min/max/median names
listed as aliases of the RDNA4 `*_num_*` names in the machine-readable ISA.
Thin entry points forward to the existing implementations, including their
CPU dispatch, modifiers, DPP support, masked stores and operand alias handling.
No numerical algorithm or SIMD loop was duplicated or changed.

These paths remain loose-only. The RDNA3 manual distinguishes IEEE-mode
signaling-NaN propagation from default number selection; the new names do not
claim to implement that exact behavior. Strict exact requests fail, and exact
requests for exception reporting return `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE`.
Loose calls leave the optional exception register untouched. No new RDNA3
hardware observations are claimed. Wave64 and exact variants remain open.

New tests exercise independently specified finite results, signed zeros,
NaNs, infinities and subnormals, all available CPU dispatch levels, masks,
binary destination aliasing, ternary operand order and modifiers, and unchanged
outputs on exact/reporting errors. Existing numerical and instruction-flag
suites continue to exercise the shared implementations. The inventory now has
769 canonical in-scope names with APIs and 40 absent; names alone do not establish
complete architectural coverage.

Validation: all 720 tests pass with Clang and GCC; all four new tests pass under
ASan/UBSan. Public standalone-header checks and exhaustive instruction-flag
checks include the new entry points.

## Scalar relative-register moves

Implemented S_MOVRELS_B32/B64, S_MOVRELD_B32/B64 and S_MOVRELSD_2_B32,
shared by RDNA3 and RDNA4. These are raw integer transfers in both semantics,
independent of EXEC and the host FP environment. B64 uses two consecutive
32-bit registers, with even base/index preconditions from the ISA.

Both manuals' S_MOVREL pseudocode adds DWORD offsets without scaling; RELSD_2
extracts M0[9:0] and M0[25:16]. The audited rocjitsu revision instead scales B64
offsets and extracts different fields, so those particular handlers were not
copied. A new RX 9070/gfx1201 probe covers all five instructions at five offsets
and confirms the manual interpretation. Its 450 captured register words and
SHA-256 are recorded in `tests/scalar_relative_hardware.h`; the reproducible
probe generator is `tests/capture_scalar_relative.py`. The probe saves/restores
M0 and SGPR0/1; M0 is the only reserved register it temporarily uses. No RDNA3 hardware capture
is claimed.

Indexed arrays map SGPR0..107 (including VCC) and TTMP0..15 at indices108..123.
The manuals' region rule forbids indexing across these ranges: out-of-range
reads redirect to SGPR0; writes are discarded. Explicit counts additionally
bound available storage. Destination counts must exclude TTMPs outside the
trap handler. Non-indexed scalar inputs are passed by value and non-indexed
outputs are ordinary destination pointers, allowing the caller to decode any
valid scalar operand. Widened address arithmetic prevents host overflow from
wrapping a bad index into an accessible register. Boundary behavior follows
both manuals and is separately tested; the GPU capture covers ordinary
SGPR indexing plus offset 256, confirming fallback to SGPR0/1 on reads and
discarded writes without truncating the M0 offset.

Tests cover region/count boundaries, full-width M0, split-field reserved bits,
source/destination overlap, zero-sized arrays, and flag validation before
operand access. Canonical name gaps are now 35 (774 present of 809 in scope).

Validation: all 724 tests pass with Clang/GCC; all four new scalar-relative tests
pass under ASan/UBSan. Header standalone checks and exhaustive instruction-flag
checks cover the new APIs. Existing SIMD code is unchanged.


## Relative vector swaps

Added shared V_SWAPREL_B32 for Wave32 and Wave64. Both operands are writable
register-file views. M0[9:0] and M0[25:16] select source and destination offsets;
reserved bits are ignored. Both manuals specify that an invalid operand cancels
the entire instruction, since both operands are destinations. The implementation
uses the existing swap loop and the same bounded relative-index policy as the
MOVREL family, with widened sums and flag validation before pointer arithmetic.
It copies raw bits exactly in both semantics; EXEC controls both stores.

rocjitsu's RDNA3/RDNA4 handlers agree on the valid-index exchange and on cancelling
both writes for an invalid operand. New RX 9070/gfx1201 Wave32 and Wave64 captures
also confirm valid offsets, identical resolved registers, zero/sparse/full EXEC,
reserved M0 bits and independent source/destination offset-256 cancellation.
The 18,432 captured words are represented by per-case word-wise digests in
`tests/relative_swap_hardware.h`, with raw SHA-256 fingerprints and reproducible
`tests/capture_relative_swap.py` probes. No RDNA3 GPU capture is claimed.

CPU tests additionally cover noncontiguous/reversed register views, overlapping
storage, guard words, upper Wave64 lanes, empty allocations, widened-index
overflow and all invalid instruction-flag bits. Missing canonical names: 34;
775 of 809 in-scope names have APIs, without implying complete semantic coverage.

Validation: all 727 tests pass on Clang/GCC, and all eight swap-family tests pass
under ASan/UBSan. Existing SIMD implementations are unchanged.


## RDNA3 floating constant comparisons

Added all twelve Wave32 CMP/CMPX false/true FP16/FP32/FP64 entry points. Results
follow rocjitsu's constant predicates: false returns zero, true returns EXEC
filtered by DPP16 row/bank/boundary rules. CMPX returns the new EXEC mask as its
ordinary scalar destination. FP16/FP32 accept DPP8/DPP16; FP64 rejects DPP.
ABS/NEG, half selectors, CLAMP and supported flushing settings cannot change
the constant result. Existing integer and new floating forms share the short
predicate/mask implementation; no SIMD numerical loop is necessary.

The RDNA3 manual explicitly says these predicates still signal invalid for
sNaNs and, with CLAMP, qNaNs. rocjitsu's shared result handlers do not model
those exceptions. The existing GoC comparison exception classifier is backed
by RDNA4 captures, which do not establish the behavior of these RDNA3-only
opcodes. Accordingly result masks are exact, but non-null reporting in
non-loose semantics returns `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE` before any
output write. Loose mode leaves the optional register unchanged. No exception
support or new GPU observation is claimed for this slice.

The input VGPR pointers may be null because successful calls need only EXEC
and the modifiers. Tests cover all twelve entry points, all available CPU
levels, strict exact and loose semantics, EXEC/DPP filtering, accepted and
rejected modifiers, unchanged optional state in loose mode, and preservation
of both outputs on reporting errors. The generic reporting-contract test now
includes these APIs instead of treating them as implemented classifiers.

There are now 787 in-scope canonical names with APIs and 22 absent. The missing
names are VOPD operations and general scalar hardware-register access; Wave64,
exception models, and broader semantic audits remain separate open work.

Validation: all 729 tests pass with Clang/GCC. All eight constant-comparison and
exception-register contract tests pass under ASan/UBSan. Public standalone
headers and exhaustive high instruction-flag checks include the new entry points.


## Wave64 register moves and relative moves

Added Wave64 MOV_B32, MOV_B16 and all four relative MOVREL forms. All currently
accepted DPP controls stay within 16-lane rows. The wider moves therefore reuse
the existing Wave32 implementation for each half, remapping DPP16 row-enable
bits for rows 2/3. This preserves the existing SIMD source-routing paths and
keeps half/sign selection, modifier validation and raw-bit behavior in one place.
Validation runs before operand access or either half's stores. Whole-VGPR aliases
are safe because supported controls cannot read across the 32-lane boundary.
The relative-index calculation and allocation/offset checks are shared between
both wave sizes; no second indexing algorithm was introduced.

Independent routing references now accept 64-bit EXEC and directly calculate
all 64 source/destination lanes. Tests cover every row-enable combination,
representative controls from each accepted DPP class, bank filtering, FI/BOUND,
upper-lane and alternating EXEC, in-place routing, half/sign modifiers, guard
words, relative indexing and invalid descriptors. Existing Wave32 reference
users remain covered by the full suite.

The raw MOV_B16 probe now supports Wave64. Its new 8,192 RX 9070/gfx1201 outputs
match the bit-transfer/sign-modifier model, including NaNs and subnormals.
The SHA-256 and word-wise digest are in `tests/half_move_hardware.h`; the existing
hardware-corpus test now checks both wave sizes. This capture verifies raw
half-move modifiers, not Wave64 DPP routing or RDNA3 hardware behavior.

Six Wave64 APIs were added. The canonical-name count remains 787 present and
22 absent, illustrating why that count alone is not a completion measure.

Validation: all 732 tests pass on Clang/GCC, and all eleven move/relative-move
tests pass under ASan/UBSan. Standalone-header and exhaustive instruction-flag
checks include all six new entry points.


## Wave64 integer relational comparisons

Added the 72 Wave64 signed/unsigned LT/EQ/LE/GT/NE/GE CMP/CMPX entry points
for 16-, 32- and 64-bit elements. Each returns a native 64-bit scalar mask;
inactive bits are zero. A shared implementation invokes the existing comparison
kernel for both halves, remaps DPP16's upper row enables, and publishes the
combined mask only after all source reads. This last ordering is necessary when
the scalar output overlaps an upper-half source. A byte-copy publishes the mask
without introducing a typed 64-bit aliasing access to the VGPR words.

The scalar ordering logic borrowed from rocjitsu and the existing x86-64-v3/v4
kernels are reused unchanged. FP16-style half selectors apply to 16-bit integer
operands; 16/32-bit comparisons retain DPP8/DPP16, and 64-bit element comparisons
continue to reject DPP. Flags are validated before accessing any source or
writing the real output, including empty EXEC. Both semantics are exact.

Tests independently compute full 64-lane masks for all predicates, signedness,
element widths, selectors, edge values and random patterns at every available
CPU level. They cover all four upper/lower row enables, DPP filtering, empty and
upper-lane EXEC, masks overlapping each input register, guard words and error
preservation. Existing host rounding/sticky-flag/MXCSR checks now exercise both
wave sizes. No new GPU comparison capture is claimed; the existing Wave32
hardware corpus remains unchanged. Wave64 constant and floating comparisons
remain open. Canonical name counts remain 787 present and 22 absent.

The high-flag audit also exposed an existing cross-width limitation: Wave64
MBCNT_LO/HI reject DPP although Wave32 supports it. This was open at that checkpoint (resolved below); tests
must not infer modifier parity merely by stripping every `_wave64` suffix.
The shared-name check is restricted to the audited comparison/move families.

Validation: all 735 tests pass with Clang/GCC. The three new Wave64 comparison
tests, the combined-width host-FP-state test and exhaustive instruction-flag
test all pass under ASan/UBSan. Existing numerical SIMD kernels are unchanged.


## Wave64 MBCNT DPP and exact bit counts

Closed the Wave64 MBCNT_LO/HI DPP gap. The implementation stages all 64 source
lanes using existing DPP/SIMD routing, including upper row enables, then runs the
existing 64-lane counting kernel. It deliberately does not split the arithmetic
into Wave32 calls: MBCNT's prefix mask depends on the physical destination lane,
not the permuted source lane or lane number modulo 32. Whole-register aliases
remain safe, and invalid flags are rejected before source access.

The bit-count family (CLZ, CTZ, CLS, BCNT and MBCNT) now accepts strict exact
semantics. These kernels use integer bit scans/population counts and wrapping
addition, without approximation. The underlying scalar and SIMD numerical
implementations were not changed. Existing exhaustive half-word/bit-pair tests
and the Wave32 DPP GPU corpus now explicitly request strict exact semantics.

A new RX 9070/gfx1201 Wave64 capture contains 3,584 MBCNT outputs: both
instructions, seven DPP8/DPP16 descriptors and four EXEC masks, covering
FI/BOUND, row/bank selection, upper-lane behavior and accumulator wrap.
`tests/capture_mbcnt_dpp.py` reproduces the probe; per-case digests and the raw
SHA-256 are in `tests/mbcnt_dpp_hardware.h`. Every available CPU path matches.
Independent tests additionally cover all sixteen row-enable masks and aliases
with A or B. No RDNA3 hardware observation is claimed.

No canonical names or new entry points were added. Wave64 CLZ/CTZ/CLS/BCNT APIs
remain open; this slice completes the DPP support of the already existing
Wave64 MBCNT APIs and enables exact semantics for the existing bit-count family.

Validation: all 737 tests pass on Clang/GCC. All fourteen bit-count, integer DPP
and instruction-flag tests pass under ASan/UBSan. Both existing Wave32 and new
Wave64 DPP captures pass in strict exact semantics for the bit-count operations.


## Remaining Wave64 bit-count entry points

Added Wave64 CLZ, CTZ, CLS and BCNT, completing both wave sizes for the currently
exposed bit-count family. The public entry points instantiate the existing
64-lane executor and x86-64-v3/v4 templates; no numerical or routing algorithm
was duplicated. Both semantics are exact, DPP8/DPP16 are supported, and inactive
lanes/unselected storage are preserved. The existing sentinel and wrapping rules
are unchanged.

Tests cover all 65,536 half-word patterns placed low, high and repeated into both
halves of each input word, including zero/all-one sentinels. They check every
available CPU path against the independent bit-by-bit reference, along with
DPP, upper-lane masks, aliases with A or B, guards, invalid descriptors and host
rounding/exception/MXCSR preservation. The existing Wave32 bit-count and Wave64
MBCNT hardware corpora still cover the shared numerical/routing components;
no new direct GPU captures of these four Wave64 opcodes are claimed.

Four APIs were added. Canonical name counts remain 787 present and 22 absent.

Validation: all 740 tests pass on Clang/GCC. All thirteen bit-count and
instruction-flag tests pass under ASan/UBSan. Existing numerical/routing bodies
are unchanged; the new SIMD specializations vary only the lane count.


## VOPD decoded operations

Added the nineteen canonical VOPD operation names (21 APIs including the two
RDNA4 DOT2ACC variants). Each forwards to the corresponding ordinary operation;
DOT2ACC supplies the old destination as the addend. Existing scalar/SIMD dispatch
and exact models are reused without a second arithmetic implementation.

Section 7.8 of the RDNA4 ISA permits Y to overwrite an X source, but forbids X
from overwriting a Y source. Consequently, legal pairs can execute X then Y.
The caller validates slot, bank, destination parity and literal restrictions and
resolves encoded operands, including scalar DOT2 replication and out-of-range
VGPR addressing. VOPD is Wave32-only and encodes no arithmetic modifiers or DPP;
all nonzero instruction flags are rejected. LLVM's gfx1201 assembler confirms
that the ordinary ADD form is accepted and a negated VOPD operand is rejected.

The RDNA3 and RDNA4 rocjitsu VOPD handlers select different GFX11/GFX12 DOT2
models. Both FP16 and BF16 therefore have unsuffixed RDNA3 and `_rdna4` APIs.
Tests reuse the borrowed GFX1100 ordinary DOT2 captures and check an explicit
architecture-difference witness. These are not new VOPD hardware captures.
Additional tests independently check each operation's finite result, masks,
accumulator/literal/condition operands, every instruction-flag bit, state
preservation on errors, all available CPU levels and a legal reverse dependency.
Exact support and optional exception reporting retain the ordinary APIs' limits.

Canonical-name coverage is now 806 present and three absent (general scalar
GETREG/SETREG). This is not semantic completeness: the family checklist and
cross-cutting gates above remain open.

Validation: all 744 tests pass with Clang 21 and GCC 15 Release builds. All ten
VOPD, exception-register and instruction-flag tests pass under ASan/UBSan.


## Hardware-register access audit (implementation still open)

See [the detailed HWREG audit](HWREG_AUDIT.md) for the complete XML register-ID
inventory, privilege differences, snapshot-read side effects, clock-input needs
and implementation gates. The three APIs remain missing.

The three remaining canonical names, `s_getreg_b32`, `s_setreg_b32` and
`s_setreg_imm32_b32`, cannot be completed by adding a generic bitfield helper.
RDNA3 ISA section 3.4 and SOPK definitions specify a six-bit register ID,
five-bit bit offset and five-bit size-minus-one. SETREG also applies
register-specific writable bits and privilege and can trigger side effects.
The printed SETREG pseudocode's mask does not include the offset shift; it
must not be copied without checking hardware or the independent model.

rocjitsu `vm/amdgpu/hwreg.cpp` provides useful field decoding/insertion and
architecture tables, but many entries explicitly return Unsupported. Its RDNA3
STATUS descriptor is read-only even though the manual allows some privileged
writes. It is evidence for partial behavior, not an exhaustive reference.

| Register ID | RDNA3 | RDNA4 |
|---|---|---|
| 1 | MODE | WAVE_MODE |
| 2 | STATUS (includes privileged writable fields) | WAVE_STATUS (separate privileged state) |
| 3 / 4 | TRAPSTS at 3 | WAVE_STATE_PRIV at 4 |
| 17 | SHADER_TBA_HI | WAVE_EXCP_FLAG_PRIV |
| 18 | PERF_SNAPSHOT_PC_LO | WAVE_EXCP_FLAG_USER |
| 19 | PERF_SNAPSHOT_PC_HI | WAVE_TRAP_CTRL |
| 20 / 21 | SHADER_FLAT_SCRATCH_LO/HI | WAVE_SCRATCH_BASE_LO/HI |
| 29 / 30 | SHADER_CYCLES at 29 | SHADER_CYCLES_LO/HI |

A raw-encoding API therefore needs `_rdna4` variants. Implementing it requires
explicit input snapshots for readable registers and one trailing optional output
per writable architectural register, together with the existing exact/loose
state contract. Scheduler/cache actions such as FLUSH_IB must be separated from
register arithmetic rather than silently fabricated. Architectural register
write masks, privilege-dependent writes and side effects remain unchecked work.
This does not change the family's in-scope status or claim those three APIs.

## Wave64 constant comparisons

Added the 28 Wave64 F/T integer and floating CMP/CMPX entry points. One shared
mask-width template retains the Wave32 validation and result rules. DPP uses
independent 16-lane rows; the Wave64 implementation remaps the row-enable bits
for each half and combines both filtered masks before writing the result.
No operand loads or floating arithmetic are needed for these predicates.

Tests independently check both semantics, every available CPU level, masks
including lane 63, DPP routing/filtering, modifiers, and error preservation.
Floating exact non-null exception reporting still fails explicitly; loose mode
leaves the supplied state unchanged. No new hardware captures are claimed.
Canonical-name counts remain 806 present and three absent.

Validation: all 747 tests pass with Clang 21 and GCC 15 Release. All twelve
constant-comparison, exception-register and instruction-flag tests pass under
ASan/UBSan. No existing arithmetic or SIMD kernel changed.


## Wave64 floating relational comparisons

Added all 84 relational FP16/FP32/FP64 CMP/CMPX Wave64 APIs. The existing
32-lane numerical and x86-64-v3/v4 kernels remain unchanged. A Wave64 executor
validates once before reading operands, remaps DPP row enables for each half,
and stages both comparison masks and accumulated exceptions before writing
outputs. Exact non-null reporting selects the existing integer classifier;
loose reporting stays off the SIMD path's arithmetic and leaves state unchanged.

Independent tests cover predicates, signed zeros/NaNs/denormals and other edge
inputs, all sign/half selectors, flushing, every DPP row-enable mask, lane 63,
and a result overlapping the upper source half. Exception tests place the
existing RX 9070 captured operand pairs in lanes 0, 31, 32 and 63 and check masked
accumulation, signaling comparisons, input flushing and loose opt-out behavior.
These reuse the Wave32 numerical observations; no new Wave64 GPU capture or
RDNA3 floating exception validation is claimed. CLASS predicates and the
cross-architecture floating-policy audit remain open.

Validation: all 750 tests pass with Clang 21 and GCC 15 Release builds. All nine
Wave64 floating-comparison, exception-register and instruction-flag tests pass
under ASan/UBSan. Canonical counts remain 806 present and three absent.


## Wave64 CLASS and explicit width backlog

Added the six FP16/FP32/FP64 CLASS CMP/CMPX Wave64 APIs. Each reuses the existing
scalar and x86-64-v3/v4 classification kernels, with row-enable remapping for
DPP and delayed 64-bit result stores. Raw denormal classification is independent
of guest flushing flags. CLASS does not generate floating exceptions.

Tests exhaust every FP16 encoding against each of the ten class bits on every
available CPU path. Further tests cover both semantics, all three formats,
sign/half modifiers, all sixteen DPP row masks, lane 63, result/source overlap,
invalid calls and zero EXEC. All four host rounding modes and preexisting sticky
flags are preserved. Existing Wave32 classification captures continue testing
the shared kernel; no new Wave64 hardware observations are claimed.

The generated inventory now separately lists APIs taking a 32-bit EXEC mask
without a Wave64 counterpart, grouped by implementation file. There are 336
candidates after this slice, excluding architecturally Wave32-only VOPD.
This is a review queue, not a legality claim: each encoding's width must be
checked, then implemented or explicitly restricted. Scalar/pseudo-scalar/matrix
APIs are outside this signature screen and still require their own audit.
The three absent canonical names therefore substantially understate remaining
coverage work. All currently exposed comparison families now have both widths,
but the cross-architecture semantics/reporting audit remains open.

Validation: all 752 tests pass with Clang 21 and GCC 15 Release. All ten CLASS
and instruction-flag tests pass under ASan/UBSan. Sanitizer validation caught
and verified the fix for an initial FP64 adapter overread: the class-selection
operand occupies one VGPR even though the classified value occupies two.


## Wave64 Boolean operations and conditional selects

Added nine Boolean and two CNDMASK Wave64 entry points. They reuse the existing
portable and x86-64-v4 kernels, remapping DPP row-enable bits per half; CNDMASK
also consumes the corresponding half of its 64-bit condition. Whole-VGPR source
aliases remain safe because the supported DPP controls do not cross 16-lane
rows. Inactive lanes and the unwritten half of 16-bit destinations are preserved.
Existing v3 policy is unchanged: prior measurements favored the portable paths.

These operations are pure raw-bit logic, so strict exact semantics is now
accepted at both widths. No arithmetic or SIMD kernel changed. The former
unsupported-exact validation tests now exercise a genuinely reserved semantics
value instead. New independent bit-by-bit tests check all eleven operations,
both semantics, all CPU paths, all half/sign modifiers, DPP8/DPP16, all row
enables, different upper/lower condition bits, whole-source aliases, masks and
guards. Existing Wave32 hardware corpora continue to exercise shared kernels;
no new Wave64 capture or throughput measurement is claimed.

The wave-width inventory now has 325 unresolved candidates. Canonical-name
counts are unchanged at 806 present and three absent.

Validation: all 755 tests pass on Clang 21 and GCC 15 Release builds. All 24
Boolean, conditional-selection, integer DPP and instruction-flag tests pass
under ASan/UBSan, including the existing exhaustive routing suites.


## Wave64 carry/borrow arithmetic

Added the six ADD/SUB/SUBREV CO and CO_CI Wave64 APIs, with 64-bit EXEC,
input-carry and scalar output masks. Both semantics reuse the existing scalar,
x86-64-v3 and x86-64-v4 kernels. DPP row enables are remapped per half; SUBREV
continues to route B, while ADD/SUB route A. Carry-in bits retain their original
lanes. The scalar output is staged until after both halves, preserving source
aliasing, and zero EXEC writes a zero carry mask without touching VGPR pointers.

Tests independently evaluate all boundary operand pairs, carry/borrow patterns
with different upper/lower halves, saturation, both semantics and every CPU
path. They cover DPP8/DPP16 and all row masks, whole-VGPR aliases, scalar carry
output overlapping the upper source half, guards, error preservation and null
VGPRs at zero EXEC. Existing Wave32 hardware captures still validate the shared
arithmetic; no new direct Wave64 GPU capture or performance measurement is
claimed. The width-review queue now has 319 candidates.

Validation: all 758 tests pass with Clang 21 and GCC 15 Release builds. All
twelve carry/borrow and instruction-flag tests pass under ASan/UBSan.


## Wave64 shifts and bitfields

Added six reverse-shift and eight bitfield APIs. The 64-bit shift adapter takes
one count VGPR and two value/destination VGPRs, and rejects DPP. The 32-bit
shifts and all bitfield operations retain DPP on A with four Wave64 row-enable
bits. Each executes the existing kernels for both halves; whole-VGPR aliases
are safe because DPP stays within 16-lane rows. Scalar/v3/v4 dispatch policies
and numerical kernels are unchanged, including BFI's portable fallback on v3.

Strict exact semantics is now accepted at both widths: these operations use
integer bit manipulation only. Tests check shifts at every bit position and
count, sign extension, wrapped counts, bitfield offsets/widths across bit 31,
byte selectors, DPP8/DPP16, all row masks, both semantics, every CPU path,
whole-source aliases, guards and invalid/zero-EXEC calls. Existing Wave32
hardware captures continue to validate shared kernels; no new Wave64 capture
or throughput measurement is claimed. The width-review queue now contains
305 candidates; canonical-name counts remain 806 present and three absent.

Validation: all 761 tests pass on Clang 21 and GCC 15 Release. All 23 shift,
bitfield, Wave64, DPP and instruction-flag tests pass under ASan/UBSan.


## Wave64 non-carry integer addition and multiplication

Added six ADD/SUB/SUBREV/ADD3 and seven MUL Wave64 APIs, reusing existing
portable, x86-64-v3 and x86-64-v4 kernels and dispatch policy. Ordinary 32-bit
multiply still rejects DPP; 24-bit multiply supports it, and only low-product
24-bit forms accept saturation. ADD3 still wraps and rejects CLAMP; SUBREV's
DPP source remains B. Whole-VGPR aliases are preserved with row-local DPP.

Both wave sizes now accept strict exact semantics for these integer kernels.
Tests independently cover signed/unsigned 24/32-bit boundary pairs, random
words with discarded upper bytes, high/low products, saturation, ADD3 overflow,
all CPU paths, both semantics, DPP8/DPP16 and every row mask, different masks
across halves, whole-source aliases, guards and validation before operand
access. Existing Wave32 captures continue validating shared numerical/routing
logic; no new Wave64 GPU capture or throughput measurement is claimed.
The width-review queue now has 292 candidates.

Validation: all 763 tests pass with Clang 21 and GCC 15 Release. All nineteen
integer arithmetic, Wave64, DPP, instruction-flag and zero-EXEC tests pass under
ASan/UBSan. The zero-EXEC support table now records the expanded exact support.


## Wave64 integer selection and ternary operations

Added fourteen signed/unsigned min/max/median and eight ternary Wave64 APIs.
Both widths now accept strict exact semantics; existing numerical and SIMD
kernels are unchanged. Binary min/max does not access C. DPP routes A within
16-lane rows and retains all four row enables. Mixed min/max still combines
A/B before C, and byte LERP uses independent byte sums and rounding bits.

Independent tests cover all triples from signed/unsigned boundary sets, random
words, wrapping shifts/addition, byte rounding, all CPU paths, both semantics,
DPP8/DPP16 with every row-enable mask, different EXEC patterns across halves,
all whole-source destination aliases and guards. Validation checks zero EXEC
with null pointers and all unsupported low instruction-flag bits. Existing
Wave32 hardware captures remain the direct hardware evidence for the shared
kernels; no new Wave64 captures or throughput measurements are claimed.
The width-review queue now contains 270 candidates.

Validation: all 765 tests pass with Clang 21 and GCC 15 Release. All 24
selection, ternary, Wave64, DPP, zero-EXEC and instruction-flag tests pass under
ASan/UBSan. The zero-EXEC table reflects the expanded exact support.


## Wave64 ordinary and packed 16-bit integer arithmetic

Added 34 APIs: twelve ordinary binary/shift operations, eight ordinary ternary
operations, twelve packed binary/shift operations and two packed MAD forms.
The existing shared scalar/v3 kernels serve both widths. Ordinary forms preserve
the unselected D half and permit DPP on A before half selection; packed forms
compute both halves from original sources and reject DPP. Both widths now
accept strict exact semantics for this integer arithmetic.

New independent tests cover every supported half-selector and saturation
combination, signed/unsigned boundaries and random words, both semantics and
all CPU paths, whole-source aliases, masks/guards, and all ordinary DPP row
masks plus DPP8. They check packed DPP rejection and validation before operand
access. Existing exhaustive Wave32 operand tests and hardware captures remain
in place for the shared kernels; no new Wave64 captures or throughput numbers
are claimed. The width-review queue is now 236 candidates.

Validation: all 767 tests pass on Clang 21 and GCC 15 Release. All 31 ordinary
and packed integer16, Wave64, DPP, zero-EXEC and instruction-flag tests pass
under ASan/UBSan, including both exhaustive ordinary DPP suites.


## Wave64 narrow-factor and wide-result integer MAD

Added four 16/24-bit-factor MAD and four wide-result MAD APIs (including RDNA3
spellings). Narrow-factor forms now accept strict exact semantics at both
widths and reuse existing scalar/v3 kernels, DPP and selectors. Wide forms
reuse scalar/v3/v4 kernels, require one VGPR per factor and two for C/D, reject
DPP and produce a 64-bit carry mask. Their signed carry is the extended sum's
bit 64 before saturation. The scalar output is committed after all source
reads and vector writes, including zero for zero EXEC.

Independent tests cover boundary factors/addends, saturation, both semantics,
every CPU path, DPP row masks and selectors where supported, differing EXEC
halves, whole-source aliases, guards and invalid/empty calls. Wide results and
carry use the independent 128-bit reference, including negative sums and signed
overflow. A scalar-output alias in the upper source half checks delayed carry
stores against separately captured vector results. Existing Wave32 hardware
corpora remain the direct numerical evidence; no new Wave64 capture or speedup
claim is made. The width-review queue is now 228 candidates.

Validation: all 770 tests pass on Clang 21 and GCC 15 Release. All twenty
integer/wide MAD, Wave64, DPP, zero-EXEC and instruction-flag tests pass under
ASan/UBSan.

## Wave64 SAD and exact integer semantics

Added all eight SAD/MSAD/QSAD/MQSAD Wave64 APIs. Both wave widths now accept
strict exact semantics, using the existing integer arithmetic and scalar/v3
kernels. Ordinary forms route A through DPP; quad forms reject DPP, read two
A VGPRs and one B VGPR, and produce either two packed or four full-width
accumulator VGPRs. The Wave64 wrapper remaps row enables for each half without
changing the arithmetic kernels or adding sparse-mask paths.

Independent reference tests cover sliding byte windows, unsigned extremes,
zero-byte suppression, independent packed/full-width saturation, all CPU paths
and both semantics, upper/lower EXEC bits, DPP row masks, source/output aliases
and guard words. Invalid and zero-EXEC calls are tested without operands.
Existing Wave32 exhaustive arithmetic tests and the RX 9070 DPP capture remain
numerical evidence for the shared kernels; no new Wave64 hardware capture or
throughput measurement is claimed. The width-review queue is now 220 candidates.

Validation: all 772 tests pass on Clang 21 and GCC 15 Release. All thirteen
SAD, Wave64 SAD and instruction-flag tests pass under ASan/UBSan.

## Wave64 integer DOT4/DOT8

Added the four signed/mixed and unsigned byte/nibble dot-product Wave64 APIs.
RDNA3 and RDNA4 use the same packed operands and signedness controls; the
entry points remain architecture-neutral. Each operand is one VGPR, DPP is
rejected, and all modifier combinations reuse the existing scalar/v3 kernels.
Saturation remains after the whole dot plus accumulator, without intermediate
byte-pair saturation. Both widths support exact integer semantics.

The Wave64 tests use the independent radix-based reference for every
signedness/CLAMP combination, both semantics and all CPU paths, boundary and
random inputs, upper/lower masks, aliases, and guard words. Zero-EXEC and invalid
calls require no operand access. Existing cancellation and host-environment
tests continue to exercise the shared kernels. No new hardware capture or
performance measurement is claimed. The width-review queue is now 216 APIs.

Validation: all 773 tests pass on Clang 21 and GCC 15 Release. All five
integer DOT and instruction-flag tests pass under ASan/UBSan.

## Wave64 saturation and FP16 packing

Added Wave64 SAT_PK_U8_I16 and PACK_B32_F16, sharing their scalar/v3/v4
kernels and DPP routing. Both widths now accept strict exact result semantics.
SAT preserves the unselected destination half; PACK preserves payloads and
subnormals while quieting signaling NaNs. Exact PACK with non-null optional
exception output remains explicitly unsupported; loose mode skips reporting.

The Wave64 exact tests reproduce the existing RX 9070 capture digests for all
65536 encodings and 66 modifier combinations, regrouped into 64-lane calls.
This validates result bits against existing captures, not new native Wave64
hardware execution. Independent DPP tests cover every row mask, all modifiers,
CPU paths, upper EXEC bits, aliases and guards. Empty/invalid calls and optional
state opt-out/rejection are covered. The width-review queue is now 214 APIs.

Validation: all 775 tests pass on Clang 21 and GCC 15 Release. All ten packing
and instruction-flag tests pass under ASan/UBSan.

## Wave64 integer widening and saturating conversion

Added all four signed/unsigned integer widening and packed narrowing Wave64
APIs, using the existing scalar/v4 kernels and v3 packed conversion paths.
Both widths now support strict exact integer semantics. The unary widening
forms read only A; packed forms read A/B. DPP routes A before half selection
or saturation and remaps all four row-enable bits across the two halves.

Independent tests exhaust all 65536 source halves and signed/unsigned
saturation boundaries, then test whole-word operands with DPP, every row mask,
upper/lower EXEC bits, source/destination aliases and guard words on every CPU
path. Empty and invalid calls are checked before operand access. Existing
Wave32 hardware/DPP and host-FP-state tests remain in place. No new native
Wave64 capture or performance claim is made. The width-review queue is 210 APIs.

Validation: all 776 tests pass on Clang 21 and GCC 15 Release. All ten integer
conversion and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP32/FP64 exponent extraction

Added both FREXP_EXP_I32 Wave64 APIs with the existing scalar/v3 execution
paths. FP32 accepts DPP and remaps all row enables; FP64 rejects DPP and reads
two source VGPRs. Both write one integer-result VGPR. Loose optional state
remains unchanged; exact non-null reporting is rejected before operand access.

The exact-semantics gate is deliberately unchanged. FP64 SIMD normalizes
subnormals through host floating-point scaling and requires host denormal
inputs enabled. A guest/host denormal-state audit is still necessary before
claiming exact support. No production FP-environment mutation was added.

Independent integer-reference tests cover every exponent field, subnormal
leading bits, all modifiers and CPU levels, DPP row masks, upper/lower EXEC,
aliases with either source half, guards, empty/invalid calls and optional state.
The existing FP32 hardware corpus remains in place; this adds no new Wave64
hardware capture. The width-review queue is 208 APIs.

Validation: all 777 tests pass on Clang 21 and GCC 15 Release. All nine
exponent-extraction and instruction-flag tests pass under ASan/UBSan.

## Host-independent FP64 exponent extraction

Removed the FP64 host-denormal dependency identified above. Scalar extraction
now uses integer exponent fields and the existing leading-zero helper. AVX2
constructs 2^52 plus the integer significand and subtracts 2^52 exactly; it never
consumes a subnormal floating-point input. Normal and exceptional results keep
their previous rules. No host control-register writes or mask-density dispatch
were added. Exact guest semantics/reporting remain a separate open gate.

Tests cross every subnormal leading-bit position, boundary significands,
normals, signed zeros, infinities and signaling NaNs with all host rounding
modes and DAZ/FTZ combinations on every CPU path through Wave64. The results
match an independent integer reference and preserve the complete MXCSR and
preexisting exception flags. Existing Wave32 and DPP tests continue to pass.

A pinned thread-CPU-time microbenchmark used five samples of four million
Wave32 calls per case, with identical saved-before and rebuilt-after harnesses.
After the test workloads finished, median ns/wave were:

| Inputs | Path | Before | After |
|---|---|---:|---:|
| Normal | Scalar | 80.141 | 33.025 |
| Normal | AVX2 | 20.530 | 20.702 |
| Subnormal | Scalar | 98.198 | 40.705 |
| Subnormal | AVX2 | 33.294 | 20.699 |

These are local focused measurements, not a whole-library performance claim.
The normal SIMD difference is below 1%; subnormal SIMD avoids the previous
floating-point denormal cost. Logs and harness are in the build directory's
`coverage-work/frexp-speed-*` files.

Validation: all 778 tests pass on Clang 21 and GCC 15 Release. All nine
exponent-extraction tests pass under ASan/UBSan.

## Wave64 byte/nibble conversion and byte packing

Added six Wave64 APIs covering four unsigned-byte-to-FP32 selections,
signed-nibble offsets, and nearest-even FP32-to-byte packing. The five
integer-to-float forms now accept strict exact semantics at both widths: their
finite values and OMOD scaling are exactly representable. Byte packing retains
loose semantics and rejects exact non-null optional exception reporting.
All forms reuse their existing scalar/v3/v4 kernels and DPP routing. Packing
routes A only, keeping byte selector B and preserved word C in their own lanes.

Independent references test all byte/nibble values and output modifiers,
rounding ties, saturation, signed infinities/NaNs, byte selections, all DPP row
masks, upper/lower EXEC, every source alias and guard words on all CPU levels.
Loose optional state remains unchanged, and exact reporting rejects before
operand access. Existing hardware captures and exhaustive Wave32 tests remain
in place; no new Wave64 capture or throughput measurement is claimed.
The width-review queue is now 202 APIs.

Validation: all 780 tests pass on Clang 21 and GCC 15 Release. All twenty
byte conversion, byte packing and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP8/BF8 widening and raw NaN correction

Added all four single/packed FP8/BF8 widening Wave64 APIs and enabled exact
semantics at both widths. The existing scalar/v3/v4 algorithms are shared;
single-result forms support DPP, packed forms reject it and write two VGPRs.
Destination aliasing retains the second-result-wins rule.

Review found that prior hardware tests canonicalized NaNs before comparing,
so they could not justify an exact NaN-bit claim. A new RX 9070 gfx1201 probe
captured all 256 codes for both single instructions and both packed result
pairs. Hardware maps every NaN, regardless of input sign, to `0xffc00000`.
The former sign-preserving conversion behavior was corrected in scalar and
SIMD kernels. The shared SIMD decoder has an explicit constant NaN-policy
parameter used by conversions; existing dot/matrix callers retain their prior
policy. This changes the NaN constants in the same SIMD instruction sequence,
without adding a reporting pass or changing finite arithmetic.

`tests/capture_fp8_widen.cpp` reproduces the 1536 raw captured result words.
The regression hashes raw words without NaN normalization and checks both
wave widths on all CPU paths; Wave64 regroups the Wave32 hardware corpus.
Additional independent tests exhaust all packed byte pairs, selectors,
upper/lower masks and aliases, including overlapping packed outputs, and
exercise every DPP row-enable combination. The width-review queue is 198 APIs.

Validation: all 782 tests pass on Clang 21 and GCC 15 Release. All twelve FP8
widening and instruction-flag tests pass under ASan/UBSan. The raw HIP probe
completed successfully on the local RX 9070.

## Wave64 FP8/BF8 nearest-even and stochastic narrowing

Added all four Wave64 narrowing APIs and enabled strict exact semantics at
both widths. They share the existing integer scalar/v3/v4 models, including
captured canonical NaNs, infinity behavior, finite overflow saturation and
stochastic significand alignment. Packed forms preserve the unselected half;
stochastic forms preserve three bytes and use B's seed without modification.
DPP routes only A and remaps row enables across both halves.

The existing raw 20-column hardware fixture now runs in strict exact mode at
Wave32 and through the new Wave64 calls. The Wave64 fixture repeats captured
per-lane inputs; it is not a new native Wave64 capture. Independent numeric
references additionally cover random FP32 words/seeds, every selector and
source modifier, both saturation modes, DPP row masks, upper/lower EXEC,
whole-source aliases and guard words. Invalid and empty calls avoid operand
access. No arithmetic kernels or SIMD dispatch decisions changed.
The width-review queue is 194 APIs.

Validation: all 784 tests pass on Clang 21 and GCC 15 Release. All twelve FP8
narrowing and instruction-flag tests pass under ASan/UBSan.

## Wave64 normalized conversions

Added all six signed/unsigned normalized conversions for FP32 packed, FP16
packed and FP16 unary forms. Wave64 shares the existing scalar/v3/v4 numeric
kernels and loose semantics. DPP routes A before modifiers/selection, B remains
in its own lane, and unary forms preserve the unselected destination half.
Only packed-half APIs have optional exception outputs; loose calls skip them
and exact non-null reporting rejects before operand access.

New tests replay the raw twenty-column RX 9070 fixture across 64 lanes and
compare random inputs against the independent integer significand reference.
They cross all modifiers, CPU levels, DPP row enables, masks with both EXEC
halves, source aliases, guards and optional-state behavior. Repeating a Wave32
hardware fixture across 64 lanes is not a new native Wave64 capture. Exact
semantics and guest FP-state behavior remain open; no SIMD kernel changed.
The width-review queue is 188 APIs.

Validation: all 786 tests pass on Clang 21 and GCC 15 Release. All fourteen
normalized-conversion and instruction-flag tests pass under ASan/UBSan.

## Wave64 packed FP32 conversion

Added Wave64 FP16 RTZ and signed/unsigned integer16 packed conversions. Each
uses one VGPR per operand, routes A through DPP before modifiers, and packs A/B
into the low/high destination halves. Existing scalar/v3/v4 rounding, overflow
and NaN behavior is unchanged, as are loose semantics and optional-state rules.
Exact non-null reporting still rejects before operand access.

Tests repeat the existing nine-column raw hardware fixture across 64 lanes,
including both FP16_OVFL settings, and independently compare every modifier
combination with random FP32 inputs, all DPP row masks, upper/lower EXEC,
source aliases and guard words. Loose non-null state remains unchanged;
invalid and empty calls are checked without operands. This reuses Wave32
hardware evidence rather than claiming a new Wave64 capture. Exact semantics
and faithful optional reporting remain open. The width-review queue is 185 APIs.

Validation: all 788 tests pass on Clang 21 and GCC 15 Release. All thirteen
packed-conversion and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP32/integer conversion

Added all six FP32/integer conversion Wave64 APIs. Integer-to-float, truncation,
nearest and floor reuse the existing scalar/v3/v4 dispatch choices, including
the deliberate scalar fallback for truncating v3 calls. DPP routes A with all
four row enables. Loose semantics and optional-state behavior are unchanged;
only the two truncating float-to-integer APIs have optional exception outputs.

Independent integer-bit references check signed/unsigned rounding boundaries,
positive/negative ties, saturation, infinities and NaNs, all modifiers/CPU
levels, full and upper-only masks, all DPP row enables, random source words,
aliases and guards. Loose non-null state remains unchanged and exact reporting
rejects without operand access. Exact guest rounding/denormal semantics and
faithful exception reporting remain open. No new hardware capture or SIMD
kernel change is claimed. The width-review queue is 179 APIs.

Validation: all 790 tests pass on Clang 21 and GCC 15 Release. All twelve
FP32/integer-conversion and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP64 conversions

Added all six FP64/integer/FP32 conversion Wave64 APIs. FP64 operands retain
their low/high VGPR layout; integer and FP32 operands use one VGPR. The wrapper
uses the correct source/destination arity for each direction, rejects DPP and
retains existing scalar/v3/v4 dispatch, loose semantics and optional-state rules.

Independent integer-bit references check rounding and saturation witnesses,
NaNs/infinities and subnormal boundaries across every modifier and CPU path.
A separate finite-input test crosses all sixteen destination-word layouts,
source aliases, full/upper-only/sparse EXEC masks and guards, including the
high-word-wins rule when output halves alias. Unused pointer slots are null.
Loose optional state remains unchanged; exact reporting and invalid flags
reject before operand access. No new exactness, capture or throughput claim
is made. The width-review queue is 173 APIs.

Validation: all 792 tests pass on Clang 21 and GCC 15 Release. All ten FP64
conversion and instruction-flag tests pass under ASan/UBSan, including the
existing exhaustive alias suite. The new references explicitly discard unused
upper storage words for 32-bit source operands.

## Wave64 FP16 conversions

Added all six FP16/integer16/FP32 conversion Wave64 APIs, preserving the
one-VGPR operand layout, source/destination half selectors, untouched destination
half and existing overflow/output-scaling rules. DPP routes A before selection
and modifiers. Existing scalar/v3/v4 kernels and loose semantics are unchanged.
All optional exception pointers retain loose opt-out behavior and exact
non-null rejection before operand access.

The tests repeat captured FP16 rounding/overflow witnesses across both wave
halves and compare random input words against independent references for every
modifier, half selector, CPU level and overflow mode. DPP row enables, masks
with upper EXEC bits, aliases, guard words and unselected destination halves
are checked. NaN comparisons retain the loose payload policy. Existing
exhaustive Wave32 encoding and DPP hardware tests remain in place; this is not
a new native Wave64 capture or an exactness claim. The width-review queue is
167 APIs.

Validation: all 794 tests pass on Clang 21 and GCC 15 Release. All fourteen
FP16 conversion and instruction-flag tests pass under ASan/UBSan.

## Wave64 row permutations

Added Wave64 APIs for PERMLANE16, PERMLANEX16 and both variable-selector
forms. The ISA defines X row pairing as 0/1 and 2/3, so each half reuses
the existing scalar/AVX2/AVX-512 kernel without changing its numerical or
inactive-source rules. Both semantics remain supported. Whole-register
source/index/destination aliases are safe because no source crosses halves.

Independent row/index references check all four operations, FI/BOUND_CTRL
combinations, both semantics, all CPU paths, upper/lower-only and sparse EXEC,
source/index aliases and guard words. Invalid flags reject before operand
access; zero EXEC accepts null operands. Existing Wave32 raw hardware and host
FP-state tests remain in place. This slice adds no native Wave64 capture or
new throughput claim. The width-review queue is now 163 APIs.

Validation: all 796 tests pass on Clang 21 and GCC 15 Release. All six
PERMLANE and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP16 exponent operations and width-inventory correction

Added Wave64 LDEXP_F16 and FREXP_EXP_I16_F16 APIs using the existing scalar
and AVX2 kernels. Half selectors, signed exponents, overflow control, DPP row
filtering and whole-register aliases retain their Wave32 behavior. Loose
semantics and exact optional-state rejection are unchanged.

Independent references check exponent boundaries, FP16 special values, both
overflow modes, each DPP routing form and all sixteen row enables, source and
exponent aliases, upper-only/sparse EXEC, guards and preserved destination
halves. Loose exception outputs remain unchanged. Validation checks null
operands on zero EXEC and on rejected modes/reporting. Existing exhaustive
Wave32 encoding, exponent, hardware-hash and host-environment tests remain.

The inventory previously misclassified twelve scalar SAVEEXEC/WREXEC APIs
as missing Wave64 forms because their explicit input data is named exec_mask.
They already have distinct `_b32`/`_b64` mnemonics and implementations; they
belong in the scalar audit, not the vector-EXEC suffix screen. Correcting
that screen and adding these two vector APIs leaves 149 width candidates.
This is not a new exactness, native Wave64 capture or performance claim.

Validation: all 799 tests pass on Clang 21 and GCC 15 Release. All fifteen
FP16 exponent and instruction-flag tests pass under ASan/UBSan. Inventory
regeneration is deterministic, and all twelve scalar exclusions have `_b64`
API counterparts.

## Wave64 FP32 FMA forms

Added Wave64 FMA, FMAC, their DX9-zero forms, and literal FMAMK/FMAAK.
The wrappers retain each form's source/output modifier restrictions, literal
placement, accumulator aliasing, optional-state contract and existing scalar,
AVX2 and AVX-512 dispatch. Ordinary and DX9 forms route A through DPP within
each half; literal forms reject DPP. Existing loose semantics and exact
reporting rejection are unchanged.

Tests combine special-value inputs from the DX9 hardware corpus with independent
source-modifier, fused arithmetic and output-scaling references. They cover all
six forms, source/destination aliases, upper EXEC bits, every DPP descriptor
and row enable, literal placement, guards and unchanged loose exception state.
Invalid modes and exact requests reject before accessing null operands. Existing
Wave32 hardware tests remain; no native Wave64 capture or exactness claim is
added. The vector width-review queue is now 143 APIs.

Validation: all 801 tests pass on Clang 21 and GCC 15 Release. All nineteen
arithmetic and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP32 binary arithmetic and min/max

Added all eleven FP32 binary APIs at Wave64: ADD, SUB, SUBREV, MUL, DX9-zero
MUL, four min/max variants and the legacy MIN/MAX aliases. The existing
scalar/AVX2 operation kernels, loose semantics and optional-state rejection
are unchanged. DPP routes A and applies each half's row enables; ordinary
source/output modifiers retain their Wave32 meaning.

Tests cover every supported low modifier combination, all DPP forms plus
unpermuted execution, all row enables, upper EXEC bits, source/destination
aliases and guards. Independent references include signed zeros, subnormals,
infinities, quiet/signaling NaNs and DX9 zero-product cases. NaN payload
comparison retains the loose policy. Loose exception state stays untouched;
invalid flags and unsupported exact requests reject before operand access.
No new exactness or hardware capture claim is made. The width queue is 132.

Validation: all 803 tests pass on Clang 21 and GCC 15 Release. All eight
binary arithmetic and instruction-flag tests pass under ASan/UBSan.

## Wave64 exact FP16 FMA and optional exceptions

Added Wave64 FMA, FMAC and both literal forms using the existing exact scalar
and loose AVX2 kernels. Source/destination half selectors, accumulator selection,
DPP and overflow controls retain their Wave32 rules. Exact reporting accumulates
into a local register across both halves, then ORs the caller's EXCP_FLAG_USER
after every source read and destination write. Loose execution skips reporting.

The existing exception hardware corpus now also runs in lane 63; raw flag
hashes match, including equivalent accumulator/literal cases. Independent
integer references check all forms, exact/loose results, modifiers, half
selectors, DPP row filtering, aliases, masks and preserved halves. Directed
tests combine invalid and overflow/inexact in different halves and alias the
exception output with an unread upper source. DPP-filtered exception tests
verify host rounding and sticky-flag preservation in every rounding mode, plus
loose opt-out behavior. No new native Wave64 capture is claimed.

Corrected two stale README descriptions of FP16 FMA saving/restoring host FP
state: the implementation uses an integer intermediate and never changes host
controls. The vector width-review queue is 128 APIs.

Validation: all 807 tests pass on Clang 21 and GCC 15 Release. All fourteen
FP16 FMA and instruction-flag tests pass under ASan/UBSan, as do all five
generic exception-register tests. The generated unsupported-reporting test
list now explicitly excludes these four supported Wave64 APIs.

## Wave64 packed FP16 FMA

Added packed FP16 FMA/FMAC Wave64 forms, retaining exact scalar and loose AVX2
execution. FMA preserves all 8,192 packed selector/negation/clamp combinations;
FMAC retains its zero-instruction-flags contract. Neither form accepts DPP.
Exceptions accumulate from both packed components across both wave halves,
with the caller's register updated only after all reads and destination writes.

Tests exercise every modifier with both overflow policies and both semantics,
all CPU paths, upper EXEC bits, masks, aliases and guard words. The existing
packed exception hardware corpus also runs in lane 63. Directed witnesses
combine lower-component invalid and upper-component overflow/inexact in
different wave halves, with reporting aliased to an upper input. Rocjitsu
result witnesses check both components across the whole wave and preserve host
rounding/exception state in all four rounding modes. Loose reporting remains
untouched; validation precedes operand access.

No native Wave64 capture or new throughput claim is added. The vector width
queue is now 126 APIs.

Also corrected a stale interpolation README paragraph: RTZ P10 uses the
integer FMA intermediate instead of changing host rounding controls. Removed
the obsolete RTZ throughput claim that described the earlier implementation.

Validation: all 811 tests pass on Clang 21 and GCC 15 Release. All seventeen
packed FMA, instruction-flag and exception-register tests pass under ASan/UBSan.

## Wave64 mixed-precision FMA

Added MIX_F32, MIXLO_F16 and MIXHI_F16 Wave64 APIs using the existing kernels.
The FP32 result remains loose-only; FP16 results retain empirical exact
semantics and the measured no-exception behavior. Optional state is untouched
for supported reporting, while FP32 exact reporting remains explicitly rejected.
Source formats, half selectors, modifiers, DPP row enables, preserved destination
halves and scalar/AVX2 dispatch remain unchanged.

Independent integer references cover DPP forms, all row-enable patterns,
upper-only/sparse EXEC, shared source registers, destination aliases and guards.
The no-exception hardware corpus also runs in lane 63 across all eight source
format combinations. Exact FP16-result tests check every host rounding mode
and DAZ/FTZ setting without changing host controls or sticky exceptions.
No new native Wave64 capture is claimed. The vector width queue is 123 APIs.

Validation: all 814 tests pass on Clang 21 and GCC 15 Release. Twelve targeted
mixed-FMA hardware/host-state/Wave64, instruction-flag and exception-register
tests pass under ASan/UBSan.

## Wave64 ordinary and packed FP16 binary arithmetic

Added eighteen Wave64 APIs: ordinary ADD/SUB/SUBREV/MUL and min/max forms,
packed ADD/MUL and min/max forms, plus both sets of legacy MIN/MAX aliases.
The family-local wrapper reuses the existing scalar/AVX2 kernels. Ordinary
forms route A through DPP and preserve the unselected destination half;
packed forms read original components for both results and reject DPP.
Loose semantics and unsupported exact reporting are unchanged.

Independent references check all 1,024 ordinary modifier/half-selection
combinations with every DPP form, all row-enable patterns, both overflow
policies, all CPU paths, aliases, upper EXEC bits and guards. Packed tests
cover every valid packed modifier combination under the same overflow/CPU
choices. NaN comparisons retain loose payload semantics. All eighteen APIs
verify empty EXEC, invalid modes and unchanged optional state, including
rejection of unsupported exact requests before operand access. No new
exactness, capture or throughput claim is made. The width queue is 105 APIs.

Validation: all 818 tests pass on Clang 21 and GCC 15 Release. All five
new Wave64 ordinary/packed binary and instruction-flag tests pass under
ASan/UBSan, including the complete modifier sweeps.

## Wave64 three-input FP16/FP32 min/max

Added all 28 three-input min/max/median Wave64 APIs, including legacy
spellings. Each family reuses its existing scalar/AVX2 selection algorithm,
retaining A/B-first mixed ordering, NaN and signed-zero behavior, output
modifiers, DPP routing and (for FP16) independently selected source/destination
halves. Loose semantics and unsupported exact reporting are unchanged.

Independent references sweep every FP32 low modifier and every FP16
modifier/half-selector combination while varying DPP descriptor, row enables,
overflow policy and destination alias. Tests cover all CPU paths, upper EXEC
bits, preserved destination halves, guards and unchanged loose optional state.
Validation checks empty EXEC and rejected modes/exact requests before operand
access. Existing Wave32 numerical and hardware tests remain. This adds no new
exactness, hardware capture or throughput claim. The width queue is 77 APIs.

Validation: all 822 tests pass on Clang 21 and GCC 15 Release. The five
new Wave64 min/max and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP64 arithmetic and unary operations

Added eighteen FP64 Wave64 APIs: ADD/MUL/FMA, min/max and aliases, rounding,
FRACT, FREXP mantissa, SQRT/RCP/RSQ. Each operand retains its two-VGPR low/high
layout. The family wrapper respects unary/binary/ternary arity and reuses
existing scalar/AVX2 kernels, loose semantics, modifier rules and exact-reporting
rejection. DPP remains unsupported.

Tests compare independent finite arithmetic and higher-precision unary
references across every modifier, full/upper-only/sparse EXEC, crossed source
and destination words, shared destination words and guards. The high result
word wins when output VGPRs share storage. Upper unary inputs differ from
lower inputs, and unused ternary operands are null for binary operations.
Literal rounding/exceptional witnesses are repeated across all 64 lanes;
reciprocal/square-root comparisons retain existing loose ULP tolerances.
Validation rejects before operand access, and loose exception state remains
unchanged. No new exactness or native hardware claim is made. The width queue
is 59 APIs.

Validation: all 827 tests pass on Clang 21 and GCC 15 Release. All six new
Wave64 FP64 arithmetic/unary and instruction-flag tests pass under ASan/UBSan.


## Wave64 FP16 and FP32 unary operations

Added all eleven FP16 and eleven FP32 unary Wave64 APIs. Family-local wrappers
reuse existing numerical kernels and scalar/AVX2 dispatch, including scalar
FP32 EXP/LOG. Four independent DPP rows, source/output modifiers and FP16
half selection retain their existing contracts. Exact semantics and faithful
optional exception reporting remain unsupported; loose reporting adds no work.

Independent references cover every modifier/selector combination, plain and
DPP routing, all row-enable patterns, upper EXEC bits, aliases, guards and
preserved destination halves. FP32 tests also replay the existing hardware
OMOD/CLAMP/NEG corpus across 64 lanes, checking mandatory denormal flushing,
zeros and infinities while retaining loose finite SFU tolerances. This is
reuse of existing captures, not a new native Wave64 capture. Validation rejects
unsupported modes before operand access and preserves optional output state.
The vector width queue is 37 APIs. No new throughput claim is made.

Validation: all 832 tests pass on Clang 21 and GCC 15 Release. All six new
Wave64 unary and instruction-flag tests pass under ASan/UBSan.


## Wave64 empirical SIN/COS and exceptions

Added FP16 and FP32 SIN/COS Wave64 APIs, reusing the existing rocjitsu-derived
empirical model and loose AVX2 paths. Family-local wrappers remap DPP row enables
and stage exception accumulation until all source reads and destination writes
finish. This preserves reporting-pointer aliases with upper source lanes.
Loose semantics still skip reporting and retain SIMD dispatch.

Tests replay all existing exception hardware corpora with only lane 63 active,
repeat the FP32 raw result captures with every modifier across 64 lanes, and
check every FP16 encoding against the independent rocjitsu result digests.
DPP tests vary every modifier/half selector and row-enable pattern, upper-only
and sparse EXEC, whole-register aliases and exception accumulation. Additional
tests check aliased reporting, loose opt-out, error atomicity and exact FP32
results under host rounding/denormal controls without mutating those controls.
These reuse existing hardware observations; no new native Wave64 capture or
throughput claim is made. The vector width queue is 33 APIs.

Validation: all 839 tests pass on Clang 21 and GCC 15 Release. All eight
Wave64 trig and instruction-flag tests pass under ASan/UBSan, including the
complete exception corpus and exhaustive FP16 input sweep.


## Wave64 FP32/FP64 LDEXP

Added both Wave64 LDEXP APIs using the existing scalar/v3/v4 kernels and loose
numerical semantics. FP64 A/D retain low/high VGPR pairs while B remains one
signed integer VGPR. FP32 routes A through four independent DPP rows without
permuting B; FP64 continues to reject DPP. Exact reporting remains unsupported,
and loose optional state remains unchanged.

Independent higher-precision references cover all modifiers, upper-only/sparse
EXEC, crossed source/destination words, shared destination words and guards.
Literal rounding witnesses check subnormal ties, minimum-normal boundaries and
overflow. FP32 DPP tests cover all row-enable patterns, random source words,
integer exponent extremes, source sharing and destination aliases. Validation
checks errors, empty EXEC and optional-state contracts. No new exactness,
native Wave64 capture or performance claim is made. The width queue is 31 APIs.

Validation: all 843 tests pass on Clang 21 and GCC 15 Release. All five
Wave64 LDEXP and instruction-flag tests pass under ASan/UBSan.


## Wave64 exact DIV_FMAS and condition masks

Added FP32 and FP64 DIV_FMAS Wave64 APIs. Their condition input is a 64-bit
per-lane mask, split together with EXEC when reusing the existing exact
scalar/v3/v4 kernels. Optional exception flags accumulate only after both
halves consume all operands. Loose mode skips reporting; exact null reporting
retains SIMD dispatch. FP64 operands retain two-VGPR low/high layouts.

Tests replay existing Cartesian/random result digests across 64 lanes and
exception digests with only lane 63 active. Captured rounding-boundary results
check mixed lower/upper condition bits independently. Other tests cover crossed
and shared destination words, guards, upper-only/sparse EXEC, shared inputs,
all modifiers under host rounding modes, loose opt-out and a reporting pointer
aliased with an upper input. This reuses existing hardware evidence rather
than claiming native Wave64 captures or new throughput. The width queue is
29 APIs.

Validation: all 850 tests pass on Clang 21 and GCC 15 Release. All eight
Wave64 DIV_FMAS and instruction-flag tests pass under ASan/UBSan, including
the complete result/exception corpora and reporting-alias witness.


## Wave64 exact DIV_FIXUP

Added FP16/FP32/FP64 DIV_FIXUP Wave64 APIs with existing exact scalar/v3/v4
kernels, format-specific layouts and optional exception accumulation. FP16
supports four independent DPP rows and preserves the unselected destination
half. FP32/FP64 reject DPP. A family-local wrapper consumes both halves before
writing optional state, preserving aliases with upper-lane source operands.
Loose reporting remains skipped; exact opt-out retains SIMD dispatch.

Tests replay existing Cartesian result captures across 64 lanes and exception
captures with only lane 63 active. Independent references cover every modifier
and half selector, overflow policies, random inputs, whole-register/cross-word
aliases, shared sources, guards, sparse/upper-only EXEC and host rounding modes.
FP16 DPP tests cover all row enables, source routing and masked exception
accumulation. A separate zero/zero witness checks delayed reporting with an
upper input alias. This adds no native Wave64 capture or throughput claim.
The vector width queue is 26 APIs.

Validation: all 859 tests pass on Clang 21 and GCC 15 Release. All ten
Wave64 DIV_FIXUP and instruction-flag tests pass under ASan/UBSan, including
the exhaustive modifier/FP16 DPP sweeps and result/exception corpora.


## Wave64 DIV_SCALE and scalar condition output

Added FP32/FP64 DIV_SCALE Wave64 APIs using the existing exact scalar/v3/v4
kernels. Their mandatory condition output is 64 bits, with inactive bits
cleared and zero written for empty EXEC. Both halves are staged before the
condition store, which follows all source reads and VGPR writes and supports
unaligned aliases. Errors leave both vector and condition outputs unchanged.

Tests replay the existing captured value/condition digests across 64 lanes,
check all modifiers and host rounding modes, full/upper-only/sparse EXEC,
FP64 cross-word destination aliases and zero-EXEC/error behavior. Dedicated
byte-offset aliases place the condition output in each source or destination
register, in either wave half, with guard words checked. This reuses existing
hardware observations; no native Wave64 capture or new throughput is claimed.
The vector width queue is 24 APIs.

Validation: all 864 tests pass on Clang 21 and GCC 15 Release. All six
Wave64 DIV_SCALE and instruction-flag tests pass under ASan/UBSan, including
unaligned scalar-output aliases and captured value/condition digests.


## Wave64 RCP_IFLAG

Added RCP_IFLAG Wave64 using its existing loose scalar/v3/v4 kernels and
inexpensive INT_DIV0 classification. This preserves the existing reporting
behavior even in loose mode; it does not introduce exact reciprocal support.
Four DPP rows are independent. The wrapper snapshots the prior exception
register and commits accumulated flags after all source reads and destination
writes, including when the register aliases an upper input or destination.

Independent references cover every modifier, all row-enable patterns,
full/upper-only/sparse EXEC, preserved guards, input/output/status aliases and
null reporting. Existing hardware status digests are replayed in lanes 32–63;
inactive lower zeros must not contribute. Invalid modes and strict exact
requests preserve state. No new native Wave64 capture or throughput is claimed.
The width queue is 23 APIs.

Validation: all 869 tests pass on Clang 21 and GCC 15 Release. All six
Wave64 RCP_IFLAG and instruction-flag tests pass under ASan/UBSan, including
the full upper-half hardware status corpus.


## Wave64 cube-map operations

Added CUBEID/CUBESC/CUBETC/CUBEMA Wave64 APIs with existing exact scalar/v3/v4
kernels. A family-local wrapper handles lane offsets and four DPP row enables;
only X is permuted, while Y and Z retain their lanes. Numeric selection,
modifier rules and whole-register alias support are unchanged.

Tests replay the existing raw hardware Cartesian digests across 64 lanes,
check literal axis ties at every masked lane, and compare independent references
for all modifiers, special/random words, destination/shared-source aliases,
upper-only/sparse EXEC and guards. DPP tests vary all row enables and modifier
combinations. Host tests check every rounding mode and preserved sticky flags.
Validation covers empty EXEC and error atomicity. No new native Wave64 capture
or throughput claim is made. The vector width queue is 19 APIs.

Validation: all 876 tests pass on Clang 21 and GCC 15 Release. All eight
Wave64 cube and instruction-flag tests pass under ASan/UBSan, including
the complete modifier/DPP sweeps and captured Cartesian result digests.


## Wave64 MULLIT and TRIG_PREOP

Added both Wave64 APIs using existing scalar/v3/v4 kernels. MULLIT retains loose
semantics and four independent DPP rows. TRIG_PREOP retains exact table-based
results, two destination VGPRs, A's high-word-only exponent read and one selector
VGPR; it rejects DPP. Optional exact exception reporting remains unsupported
for both, while loose pointers remain unchanged.

Tests replay MULLIT's existing hardware Cartesian digests and TRIG_PREOP's
complete exponent/selector/modifier digests across 64 lanes. Independent MULLIT
references check all modifiers, DPP row enables, source sharing, destination
aliases, upper-only/sparse EXEC and guards. TRIG_PREOP checks captured samples
with crossed/shared destination words, upper-half input rotation and all host
rounding modes. A null low-word pointer verifies its unused input, and optional
state/error tests preserve outputs. No native Wave64 capture or new throughput
claim is made. The vector width queue is 17 APIs.

Validation: all 885 tests pass on Clang 21 and GCC 15 Release. All ten
Wave64 MULLIT/TRIG_PREOP and instruction-flag tests pass under ASan/UBSan,
including the full hardware digest replays and modifier/DPP sweeps.


## Wave64 interpolation

Added all six FP32/mixed-FP16 interpolation Wave64 APIs, including RTZ forms.
Each family reuses its existing scalar/SIMD numerical kernels. Quad broadcasts
remain within each 32-lane half and read inactive source lanes; wait fields,
NEG/CLAMP, half selectors, overflow policy and destination-half preservation
retain their contracts. Numerical semantics remain loose; strict exact requests
continue to fail rather than claiming new arithmetic fidelity.

Reviewed both manuals' interpolation descriptions: RDNA3 section 12.3 gives the
same fixed DPP8 broadcasts and fetch-invalid behavior as RDNA4 sections 12.3
and 16.13. This establishes the routing shared here, not untested numerical
identity between GPUs.

Tests replay existing FP16 result digests across 64 lanes and FP32 captures in
both halves. Upper-only/single-lane capture tests verify inactive source reads.
Independent numerical/integer references cover modifiers, wait counts, half
selection, saturation, random/special values, shared inputs, destination aliases,
guards and sparse EXEC. RTZ tests preserve host rounding and pre-existing sticky
flags. No new native Wave64 capture or throughput claim is made. The vector
width queue is 11 APIs, all dot products.

Validation: all 895 tests pass on Clang 21 and GCC 15 Release. All eleven
Wave64 interpolation and instruction-flag tests pass under ASan/UBSan,
including captured digests, RTZ rounding and inactive-source reads.


## Wave64 true16 DOT2 and FP8/BF8 DOT4

Added both FP16/BF16-result DOT2 APIs and all four FP8/BF8 DOT4 combinations
at Wave64. Family-local wrappers reuse existing scalar/AVX2 numerical kernels.
True16 DOT2 retains C/D half selectors, destination-half preservation and four
independent DPP rows; DOT4 retains its C modifiers and rejects DPP. Both families
remain loose-only, and true16 exact exception reporting remains unsupported.

Independent references check all 65,536 byte pairs for each DOT4 format and
modifier, every true16 accumulator encoding, literal rounding/denormal/overflow
boundaries, mixed packed elements, all modifiers/selectors, upper-only/sparse
EXEC, destination aliases and guards. DPP tests cover every row-enable pattern
and true16 modifier combination. Validation preserves rejected outputs. No new
exactness, native Wave64 capture or throughput is claimed. The vector width
queue is five FP32-result DOT2 APIs, including architectural variants.

Validation: all 904 tests pass on Clang 21 and GCC 15 Release. All ten
Wave64 narrow-dot and instruction-flag tests pass under ASan/UBSan.

## Wave64 architecture-specific FP32-result DOT2

Added the five remaining ordinary vector width variants: FP16/BF16 DOT2 for
RDNA3 and RDNA4, plus RDNA3 DOT2ACC. A family-local two-half executor preserves
the existing GFX11/GFX12 exact models, loose AVX2 kernels, DPP row enables and
whole-VGPR aliasing. RDNA4 reporting leaves EXCP_FLAG_USER unchanged, as in the
existing capture-backed Wave32 implementation; RDNA3 exact reporting remains
explicitly unsupported. No new numerical or native Wave64 hardware claim is made.

Tests replay both GPUs' raw result fixtures, including a small finite example
that distinguishes their result bits; verify modifiers, half selectors, DPP,
upper-only/sparse EXEC, old-D accumulation and aliases; and preserve host rounding
and sticky flags. RDNA4 no-exception corpora are replayed at lane 63. Independent
integer references check the shared loose/scalar/SIMD paths and four-row routing.

The vector-EXEC signature screen now has zero missing Wave64 counterparts. This
does not close the separate scalar/pseudo-scalar/matrix width audit, encoding
legality review, architecture semantics audit or three hardware-register APIs.

Validation: all 914 tests pass with Clang 21 and GCC 15 Release. All eleven
Wave64 DOT2 and instruction-flag tests pass under ASan/UBSan.

## Matrix width audit and Wave64 FP8/BF8 WMMA

Extended the generated inventory to cover matrix APIs, which omit EXEC and
therefore escaped the ordinary vector-width screen. RDNA4 manual sections
7.12.2/7.12.3 specify Wave64 layouts for these operations. After adding all four
FP8/BF8 dense WMMA combinations, fourteen matrix width gaps remain: three integer
WMMA and eleven SWMMAC APIs. Their arithmetic stage order and sparse selectors
need separate review; a dense layout conversion alone does not prove correctness.

FP8/BF8 Wave64 uses pointer views of 32-lane register sections to reuse the
existing scalar/AVX2 algorithms, with no matrix copies or new dispatch machinery.
A/B each use one VGPR, C/D four. All inputs precede destination writes. C modifiers,
loose fallback and strict-exact rejection are unchanged. Tests cover dense
mathematical goldens, every byte encoding, all four format pairs, modifiers, CPU
levels, aliases, guards and validation.

A new native RX 9070 gfx1201 Wave64 capture,
`tests/capture_fp8_wmma_wave64.cpp`, produces the word-FNV digest
`cebccc43bff33725` for each of the four formats (256 output words each). Its
exactly representable finite sums are replayed on every CPU path and with aliases.
This confirms register use and results for that corpus, not general bit-exactness.
Rocjitsu's `gfx12_wmma_input_loc` orders FP8 lane groups differently from the
manual. Our source mapping follows the manual; a common permutation of A/B's K
axis leaves these exact finite sums invariant, so the capture does not resolve
that discrepancy or establish the GPU's floating accumulation order.

Capture source SHA-256: `e25e3e098a27660b344f341e1cb242dc772cffb0af23cf1ac7678758b6ba6967`.
Validation: all 918 tests pass with Clang 21 and GCC 15 Release. All five
Wave64 FP8 WMMA and instruction-flag tests pass under ASan/UBSan.

## Wave64 integer WMMA with native saturation evidence

Added IU8 K=16, IU4 K=16 and IU4 K=32 Wave64 APIs, preserving architectural
`_rdna4_wave64` suffixes for the two shared RDNA3/RDNA4 mnemonics. Pointer views
of 32-lane register sections reuse existing scalar, AVX2 and Zen4 VNNI kernels.
A/B each use one VGPR, C/D four; IU4 K=16 ignores upper source lanes. Validation
precedes operand access, and the existing result snapshot preserves aliases.

A native RX 9070 gfx1201 capture covers 16 randomized near-limit input sets,
all eight signedness/CLAMP modes and all three shapes: 384 digests, 98,304
output words. `capture_integer_wmma_wave64.cpp` generates the recorded
`integer_wmma_wave64_hardware.h` fixtures. Source SHA-256:
`62052c723576344a55c7c20d936fb02c8bc2e1ef1c8e7cae743fd4878911f387`.

The captures distinguish the two saturation stages (source lane groups 0+2,
then 1+3) from one final clamp and, for IU8/K=32 IU4, four clamps. All captures
match the reused kernels. Independent logical-matrix tests cover extreme
factors, cancellation and overflowing accumulators using the manual's Wave64
layout. Native digests are replayed across CPU levels and destination aliases.
Invalid flags preserve outputs. The matrix width backlog is now eleven SWMMAC
APIs; existing dense WMMA APIs all have both widths, with numerical and
architecture audits still tracked separately.

Validation: all 921 tests pass on Clang 21 and GCC 15 Release. Five targeted
integer WMMA and instruction-flag tests pass under ASan/UBSan, including the
existing Wave32 capture corpus after making its input generator width-aware.

## Wave64 FP8/BF8 SWMMAC and four index sets

Added all four FP8/BF8 sparse matrix formats at Wave64. The family decoder
now handles both register layouts and feeds the existing scalar, AVX2 and
AVX-512 accumulation kernels. Wave64 uses one A VGPR, two B VGPRs, four
in/out D VGPRs and one index VGPR. Its four 8-bit metadata sets use both
selector bits; `GOC_SWMMAC_INDEX_KEY_2` exposes the second bit. Wave32 retains
its two 16-bit sets and rejects the new bit. Arithmetic remains loose-only.

Native RX 9070 gfx1201 captures cover four format pairs, sixteen input sets
and all four keys: 256 word-FNV digests over 65,536 output words. Every digest
matches all CPU paths. The generator `capture_swmmac8_wave64.cpp` has SHA-256
`9a691f2cfd9d341b6f2e0abcbb7994bbee8d8e3562faa4a5adc9b6f43e7c2b54`.
Fixtures are in `swmmac8_wave64_hardware.h`. Exactly representable input sums
validate the routing and selectors without claiming general exact arithmetic.
The lane routing follows rocjitsu's Wave64 sparse layout; four-byte selector
handling follows the ISA and native capture rather than its 16-entry helper.

Tests additionally cover every byte encoding, source/index/destination aliases,
duplicate destinations and rejected flags. The existing Wave32 captures remain
in the suite. Seven matrix width gaps remain: FP16/BF16 and integer SWMMAC.

Validation: all 925 tests pass with Clang 21 and GCC 15 Release. All ten
SWMMAC8 and instruction-flag tests pass under ASan/UBSan.

## Wave64 integer SWMMAC with shape-specific index selection

Added all three integer SWMMAC Wave64 forms, sharing decoded matrices and
existing scalar, AVX2 and AVX-512 accumulation kernels. IU8 accepts four 8-bit
index sets; both IU4 shapes accept two 16-bit sets. Wave64 K=64 therefore permits
KEY_1, unlike Wave32 K=64. A uses one VGPR, D four, and B two (one for K=32 IU4).
K=32 IU4 ignores upper A/index lanes. Inputs remain snapshotted before stores.

Wave64 K=64 saturation groups are consecutive halves of compressed K, unlike
Wave32's interleaved halves. The decoder places these products in the existing
kernels' two-stage order. Native RX 9070 gfx1201 captures cover 1,024 combinations
of shape, randomized near-limit inputs, sign flags, CLAMP and every legal key:
262,144 result words. All recorded digests match every CPU path. Capture source:
`tests/capture_swmmac_integer_wave64.cpp`, SHA-256
`cb3a44d8c2831748ba6373d03806e72acffa2750d08a012da8fb1e13abc87208`.
Fixtures live in `tests/swmmac_integer_wave64_hardware.h`.

Additional tests cover literal overflow/cancellation that distinguishes staged
saturation from final saturation, aliases with sources and metadata, duplicate
destinations, and invalid flag output preservation. Wave32 hardware fixtures
remain unchanged. Four matrix width gaps remain, all FP16/BF16 SWMMAC.

Validation: all 929 tests pass with Clang 21 and GCC 15 Release. All eleven
integer SWMMAC and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP16/BF16 SWMMAC completes the matrix width inventory

Added all four FP16/BF16 sparse forms at Wave64. Width-aware decoding handles
two A VGPRs, four B VGPRs, four FP32 or two packed D VGPRs, and four metadata
bytes per lane. The existing scalar, AVX2 and AVX-512 kernels retain pair
negation, FP32 accumulation and packed narrowing. Strict exact requests remain
unsupported; this change makes no new general numerical-exactness claim.

Native RX 9070 gfx1201 captures cover all 64 combinations of four negation bits
and two index bits for each format/output type: 256 digests over 49,152 result
words. Every CPU path matches these exactly representable sums. Generator:
`tests/capture_swmmac16_wave64.cpp`, SHA-256
`023bff6696b8698779e8ad6ca1df3bb9c7085ba9edc6e54c204e232577744f41`;
fixtures: `tests/swmmac16_wave64_hardware.h`. Additional tests cover packed
nearest-even ties, finite saturation versus infinity, literal pair negation,
aliases with inputs/metadata, duplicate destinations and validation.

The ordinary-vector and matrix API width inventories now both have zero missing
counterparts. Scalar/pseudo-scalar width contracts, encoding legality, complete
architecture comparison, hardware-register access and remaining numerical/state
gaps are separate gates and are not closed by this milestone.

Validation: all 935 tests pass with Clang 21 and GCC 15 Release. All fourteen
FP16/BF16 SWMMAC and instruction-flag tests pass under ASan/UBSan.

## Cross-architecture policy audit after width completion

Added `tools/architecture_audit.py` and the generated `ARCHITECTURE_AUDIT.md`
review queue. Every one of the 587 common in-scope canonical mnemonics is listed;
89 have direct architecture/IEEE/OMOD/NaN policy references in rocjitsu's shared
execution helpers. The report explicitly leaves transitive helpers, SIMD macros,
architecture-specific overrides and all semantic proof open. The other 222
architecture-exclusive canonical mnemonics remain covered by the ISA inventory.

A concrete finite-result incompatibility is confirmed by both manuals and
rocjitsu's `fp_mode::effective_omod`: RDNA3 ignores OMOD with output denormals
enabled or IEEE mode set, while RDNA4 applies it. At `227ced2`, a GoC scalar
probe of `goc_v_add_f32(0, 1, GOC_ALU_OMOD_2, ...)` on 1 and 2 returns success
and `0x40c00000` (6). RDNA3 with output denormals enabled requires 3. This is
a specification/model comparison, not native RDNA3 evidence. Existing RDNA4
behavior must acquire the suffix as each affected family gains RDNA3 policy;
adding a guest IEEE-mode flag is also required. Packed results and instructions
that force flushing require individual review rather than bulk renaming.

Revisited HWREG prose: RDNA4 explicitly includes the shifted SETREG mask,
whereas RDNA3's printed pseudocode omits the shift. That discrepancy must not
be mistaken for hardware semantics. Register IDs, privilege masks and side
effects still require architecture-specific API/state contracts; the three
missing HWREG APIs remain open.

Validation: four Python extraction tests cover nested scopes, comments and
strings, missing/truncated input, and architecture-exclusive inventory rows.
The report regenerates deterministically from the recorded source fingerprint.
The unchanged 935-test suites also pass on Clang 21 and GCC 15 Release;
those passing tests do not cover or resolve the newly identified RDNA3 gap.

## FP32 binary architecture-policy split

- ADD/SUB/SUBREV/MUL retain RDNA4 behavior under `_rdna4`, before `_wave64`.
  Unsuffixed APIs implement RDNA3's conditional OMOD rule. Shared arithmetic
  kernels receive the effective modifier; no duplicate SIMD implementation.
- Guest input/output denormal flushing and MODE.IEEE are accepted by these four
  families. Both scalar and eight-lane v3 kernels flush values explicitly; no
  host FP-control writes. Other families still reject the new IEEE flag.
- Tests preserve the existing RDNA4 modifier/special-value corpus and add both
  architectures, both wave sizes, guest flag combinations, OMOD, DPP, masks,
  aliases, literal signed denormal cases and unsupported exact/state contracts.
- These APIs remain loose-only. Exact NaN quieting/payloads, guest rounding modes
  and exception reporting remain unsupported; this closes an OMOD policy gap,
  not the entire arithmetic semantic audit.
- Validation: all 937 tests pass under Clang 21 and GCC 15 Release; all 10
  Binary/InstructionFlags tests pass under ASan/UBSan; four audit-tool tests pass.
  Logs: `../goc-build/coverage-work/binary-policy-*-tests.log`.

## FP32 binary selection and DX9 policy follow-up

- RDNA3 MIN/MAX now have their own explicit selection rules: in IEEE mode an
  sNaN wins over a numeric value (and is quieted); outside IEEE mode a lone NaN
  is ignored and two NaNs select A unchanged. RDNA4 MIN_NUM/MAX_NUM keep number
  preference. The manuals' V_MIN_F32/V_MAX_F32 pseudocode is stronger evidence
  here than rocjitsu's shared host `fmin`/`fmax` implementation.
- Both architectures' min/max apply input flushing before selection and ignore
  output flushing except through OMOD. This follows the instruction-specific
  denormal notes, not the different memory-atomic rules. RDNA3 additionally gates
  OMOD on guest IEEE/output-denormal mode.
- DX9 multiply is split into unsuffixed RDNA3 and `_rdna4`, in both wave widths.
  It shares the existing arithmetic kernels and adds guest denormal/IEEE flags.
- Tests extend the architecture-policy matrix to MIN/MAX and DX9 multiply, and
  use literal NaN/sign/denormal expectations across CPUs, guest flags, widths,
  masks and aliases. Existing RDNA4 tests/benchmark calls retain their behavior
  under the suffixed DX9 names. Exact arithmetic/state requests remain unsupported.
- Validation: all 939 tests pass under Clang 21 and GCC 15 Release, and all 16
  Binary/MinmaxSpellings/InstructionFlags tests pass under ASan/UBSan. Four audit
  tool tests pass. Logs: `../goc-build/coverage-work/binary-selection-*-tests.log`.

## Ordinary FP16 binary architecture policies

- ADD/SUB/SUBREV/MUL now have unsuffixed RDNA3 and `_rdna4` entry points in
  Wave32 and Wave64. RDNA3 disables OMOD with preserved output denormals or IEEE;
  RDNA4 applies OMOD regardless, matching the manuals and rocjitsu's
  `fp_mode::effective_f16_omod` policy.
- Both variants accept guest input/output denormal controls. Scalar and AVX2
  kernels flush selected input halves before arithmetic and encoded subnormal
  results after narrowing, retaining signed zero and the unselected destination
  half. Existing FP16 overflow saturation and DPP remain supported.
- Existing RDNA4 tests and benchmarks now call suffixed APIs. The new policy
  matrix spans both architectures/widths, all guest flag combinations, selectors,
  scaling factors, overflow saturation, DPP, masks and destination/source aliases.
- Packed FP16 and FP16 min/max retain their previous flags/contracts pending their
  own architecture audit; packed arithmetic does not inherit ordinary OMOD rules.
  These changes do not add exact arithmetic or faithful exception reporting.
- Validation: all 941 tests pass with Clang 21 and GCC 15 Release; all 11
  HalfBinary/InstructionFlags tests pass with ASan/UBSan. Four audit-tool tests
  pass. Logs: `../goc-build/coverage-work/half-policy-*-tests.log`.

## FP64 ADD/MUL/FMA architecture policies

- ADD/MUL/FMA now expose unsuffixed RDNA3 and `_rdna4` APIs in both wave sizes.
  RDNA3 gates OMOD on output-denormal/IEEE state; RDNA4 applies it regardless.
  This follows the manuals and rocjitsu's shared FP64 effective-OMOD policy.
- Guest input/output denormal controls are value transforms on scalar and
  four-lane AVX2 paths. FMA flushes all three selected operands before its fused
  operation. Staged low/high destination writes preserve cross-half aliases.
- Existing RDNA4 modifier, special-value and output tests retain their behavior
  under suffixed APIs. New tests combine both architectures/widths, guest flags,
  scaling/source modifiers/CLAMP, denormals and special values, EXEC masks, cross-
  half aliases and duplicate destination buffers.
- Other FP64 families still reject these guest flags pending policy review.
  Exact arithmetic, guest rounding modes and faithful optional exception reporting
  remain unsupported; this slice does not claim those numerical guarantees.
- Validation: all 942 tests pass with Clang 21 and GCC 15 Release; all 13
  Fp64/Fp64Output/InstructionFlags tests pass with ASan/UBSan. The new policy test
  also passes after the final GCC assertion-bracing correction. Four audit-tool
  tests pass. Logs: `../goc-build/coverage-work/fp64-policy-*-tests.log`.

## FP64 min/max architecture policies

- RDNA3 MIN/MAX now apply IEEE-mode signaling-NaN priority, preserve A unchanged
  when both inputs are NaNs outside IEEE mode, and gate OMOD on guest IEEE/output
  denormal state. RDNA4 MIN_NUM/MAX_NUM keep number preference and unconditional
  OMOD. The RDNA3 V_MIN_F64/V_MAX_F64 pseudocode establishes these differences.
- All six FP64 binary min/max APIs accept guest input flushing. Output flushing
  is ignored except as required by OMOD, following the instruction-specific notes.
  Both wave sizes retain four-lane AVX2 and staged cross-half alias handling.
- The RDNA3 scalar NaN rule now lives once in `minmax.h`, shared by FP32 and FP64
  using the existing FloatBits masks. Explicit vector overloads live beside the
  existing SIMD min/max functions; no policy classes, adapters or new files.
  Maintainers change the scalar selection rule in one place; SIMD intrinsics and
  the instruction-level OMOD/denormal decisions remain directly visible.
- Tests extend the FP64 policy/mask/alias matrix to min/max and add literal NaN
  payload/quieting cases, signed zero, and input/output denormal combinations.
  Existing FP32 policy tests exercise the extracted shared helper. Strict exact
  arithmetic and faithful optional exception reporting remain unsupported.
- Validation: all 943 tests pass under Clang 21 and GCC 15 Release; all 27
  Binary/Fp64/MinmaxSpellings/InstructionFlags tests pass under ASan/UBSan. Four
  audit-tool tests pass. Logs: `../goc-build/coverage-work/fp64-selection-*-tests.log`.
- A pinned-CPU-2 AVX2 MIN_NUM_F64 comparison against parent `c2a1c7f` used nine
  alternating samples with at least 50 ms thread CPU time per case. Median time
  was 40.718 -> 40.770 ns unmodified and 45.214 -> 45.645 ns with ABS/NEG/OMOD/CLAMP
  (about +0.1% and +1.0%, respectively). This limited check found no material
  regression; it does not establish performance for every instruction or host.
  Scratch driver, parent objects and raw samples are in
  `../goc-build/coverage-work/fp64-selection-*`, including `fp64-selection-timing.json`.

## Ordinary and packed FP16 binary min/max policies

- RDNA3 MIN/MAX and PK_MIN/PK_MAX now use IEEE-mode signaling-NaN priority and
  preserve A unchanged when both inputs are NaNs outside IEEE mode. Ordinary
  forms gate OMOD; packed forms do not interpret packed modifier bits as OMOD.
  All binary min/max forms accept guest input flushing and ignore output flushing
  except through ordinary OMOD, in both wave widths.
- Existing scalar and SIMD RDNA3 selection helpers now serve FP16 as well as
  FP32/FP64. FP16 SIMD conversion gains an opt-in NaN-bit-preserving path for
  legacy and propagating selection; narrowing preserves signaling bits only for
  the legacy forms. Arithmetic and RDNA4 number-selection conversion paths retain
  their existing defaults. This also fixes RDNA4 propagating FP16 min/max's sNaN
  priority when a quiet NaN competes with a signaling NaN.
- Literal tests distinguish all three selection policies using raw payloads,
  signs and quiet bits, ordinary/packed forms, both widths, guest flag combinations,
  half selectors, independent negations, clamp, masks and aliases. The ordinary
  architecture-policy matrix now includes min/max scaling/denormal combinations.
- Exact semantics and faithful exception reporting remain unsupported for these
  APIs. Ternary FP16 min/max/median are a separate policy-audit item.
- An additional all-65,536-encodings MIN/MAX test verifies that IEEE-disabled
  selection against a quiet NaN preserves every FP16 sign, payload and signaling
  bit, including through the F16C path. Conversion repair restores only the quiet
  bit changed by F16C; it does not reconstruct the entire NaN encoding.
- Validation: all 945 tests pass under Clang 21 and GCC 15 Release; all 24
  HalfBinary/PackedBinary/MinmaxSpellings/InstructionFlags tests pass under
  ASan/UBSan. Four audit-tool tests pass. Logs:
  `../goc-build/coverage-work/half-selection-*-tests.log`.
- Performance check against parent `23bbf06`: eleven alternating samples pinned
  to CPU 2, each at least 50 ms thread CPU time. AVX2 MIN_NUM_F16 medians changed
  24.607 -> 23.117 ns unmodified and 26.173 -> 25.274 ns with ABS/NEG/OMOD/CLAMP.
  MINIMUM_F16 changed 26.815 -> 26.664 ns unmodified and 28.627 -> 29.798 ns with
  modifiers (about +4.1% for that case). The measured cost buys correct sNaN
  priority; this limited finite-input check is not a whole-library performance
  result. Driver, parent objects and raw samples are in
  `../goc-build/coverage-work/half-selection-*`, including `half-selection-timing.json`.

## Architectural exception-register naming

- RDNA3 ISA section 3.4 names TRAPSTS.EXCP as the sticky exception register;
  RDNA4 uses EXCP_FLAG_USER. Arithmetic bits 0–6 agree, as also reflected by
  rocjitsu's gfx12_excp_flag_user_raw mapping. Other register fields differ.
- Renamed the final pointer to `trapsts` in 94 explicitly RDNA3 public APIs and
  their definitions: RDNA3-only canonical instructions and unsuffixed instructions
  with an explicit RDNA4 counterpart. Parameter types/order, ABI, optionality,
  accumulation/error contracts and numerical behavior are unchanged.
- Generated runtime reporting-contract tests now discover both register names.
  New `tools/global_state_audit.py` / `GLOBAL_STATE_AUDIT.md` classify every existing
  exception-output signature, with tests for architecture resolution and the real
  public declarations. Pending shared APIs are explicitly unresolved.
- This naming audit does not establish whether each instruction should expose an
  exception output at all, whether any other writable state is missing, or whether
  reporting is numerically faithful. Those remain separate completion gates.
- Validation: all 945 tests pass under both Clang 21 and GCC 15 Release; all 15
  ExceptionRegisters/Dot2Architecture tests pass under ASan/UBSan. Three new
  global-state audit tests and four architecture audit tests pass. Generated
  reporting-contract coverage remains 366 entries after accepting both parameter
  names. Logs: `../goc-build/coverage-work/register-names-*-tests.log`.

## RDNA3 FP32 ternary selection policies

- Replaced RDNA3 MIN3/MAX3/MINMAX/MAXMIN/MED3 forwarding to RDNA4 number-selection
  APIs with RDNA3 binary-policy composition, in both wave widths. RDNA3 ISA
  section 16.12 explicitly defines these nested operations; MED3 uses MIN3 for
  any NaN and otherwise removes the first maximum. IEEE mode therefore affects
  signaling NaNs at each binary selection, rather than globally propagating any
  signaling input to the final result.
- Scalar and eight-lane AVX2 paths reuse their respective binary RDNA3 selectors.
  Added input-denormal flushing and guest IEEE/output-denormal flag acceptance
  throughout FP32 ternary selection. Output denormals survive selection; RDNA3
  output scaling is disabled when output denormals are preserved or IEEE mode
  is enabled. RDNA4 scaling remains independent of those guest settings.
- Literal tests cover signaling payloads in different operand positions, all-NaN
  inputs, both IEEE settings, wave widths and optional-state preservation. Existing
  Wave64 all-modifier/DPP/mask/alias tests now use RDNA3 expectations for its APIs.
  Additional guest-mode tests distinguish input flushing, retained output
  subnormals and architecture-dependent output scaling.
- Strict exact semantics and faithful exception reporting remain unsupported for
  these APIs; loose execution still skips optional reporting. FP16 ternary
  selection remains a separate policy-audit item.
- Rocjitsu's shared MIN3/MED3 helpers confirm conditional OMOD through
  `fp_mode::effective_omod`, but use host `fmin`/`fmax` for selection. Their host
  selection is not evidence for guest IEEE-mode NaN priority; that correction
  follows the manual's nested binary pseudocode.
- Performance: eleven alternating before/after samples against `db9a8c3`, pinned
  to CPU 2, each at least 50 ms thread CPU time. RDNA4 AVX2 MIN3_NUM_F32 medians
  changed 20.384 -> 21.506 ns unmodified (+5.5%) and 24.070 -> 25.203 ns with
  ABS/NEG/OMOD/CLAMP (+4.7%). MED3_NUM_F32 changed 32.819 -> 32.883 ns and
  34.875 -> 35.102 ns (both within 1%). Input-denormal policy dispatch occurs
  outside the SIMD loop; this reduced an initial roughly 10% MIN3 regression.
  These finite-input measurements are not a whole-library performance result.
  Driver, parent objects and raw samples: `../goc-build/coverage-work/ternary-*`,
  including `ternary-timing.json`.
- Final validation: all 947 tests pass with Clang 21 and GCC 15 Release; all nine
  Minmax3 tests pass under ASan/UBSan. Seven architecture/global-state audit
  tests pass. Logs: `../goc-build/coverage-work/ternary-*-tests.log`.

## RDNA3 FP16 ternary selection policies

- Replaced RDNA3 FP16 MIN3/MAX3/MINMAX/MAXMIN/MED3 forwarding with the same
  RDNA3 nested binary selection rule used by FP32. Both wave widths use guest
  IEEE-mode signaling-NaN priority, input-denormal flushing and conditional OMOD.
  Selection preserves output subnormals except when OMOD requires flushing.
- RDNA3 MED3 follows the manual's first-maximum removal pseudocode, including
  signed-zero ties. RDNA4 MED3_NUM retains its hardware-captured ordered-zero
  median behavior. The RDNA3 result is a manual-based model, not a new native
  RDNA3 capture or a claim of empirical exact semantics.
- Reused existing scalar/SIMD RDNA3 selection and opt-in FP16 NaN-preserving
  conversion helpers. RDNA4 propagating ternary selection now also preserves
  input signaling bits until selection, fixing premature quieting by F16C.
- Literal tests check NaN signs/payloads/signaling bits, IEEE mode, first-maximum
  zero ties, all half selectors, both wave widths, masks and unchanged optional
  state. An exhaustive 65,536-encoding test checks legacy MIN3/MAX3/MED3 against
  two quiet NaNs. Existing Wave64 DPP/modifier/alias tests now use RDNA3 policy
  expectations; RDNA4 median hardware fixtures remain unchanged.
- Guest-mode tests distinguish input flush, output-subnormal preservation and
  architecture-dependent scaling. Strict exact semantics and faithful optional
  exception output remain unsupported for these selection APIs.
- Validation: all 950 tests pass under Clang 21 and GCC 15 Release; all 12
  HalfMinmax3 tests pass under ASan/UBSan. Seven architecture/global-state audit
  tests pass. Logs: `../goc-build/coverage-work/half-ternary-*-tests.log`.
- Performance against `619dd03`: eleven alternating samples pinned to CPU 2,
  each at least 50 ms thread CPU time. RDNA4 AVX2 MIN3_NUM_F16 medians changed
  29.316 -> 28.539 ns unmodified and 33.631 -> 32.500 ns with ABS/NEG/OMOD/CLAMP.
  MINIMUM3_F16 changed 33.707 -> 35.251 ns (+4.6%) and 37.016 -> 38.044 ns (+2.8%),
  including the signaling-NaN conversion correction. MED3_NUM_F16 changed
  38.987 -> 35.408 ns and 41.941 -> 39.066 ns. These finite-input timings are a
  limited regression check, not a whole-library performance claim. Artifacts:
  `../goc-build/coverage-work/half-ternary-*`, including `half-ternary-timing.json`.

## HWREG state and side-effect follow-up

- Added `HWREG_AUDIT.md`, cross-checking all 24 RDNA3 and 19 RDNA4 XML IDs plus
  the RDNA4 manual-only IB_STS2 entry. No API-presence counts changed.
- Identified RDNA4 GETREG's snapshot-PC-high read-unlock effect and sampled-wave
  visibility requirement. A read-only raw snapshot API alone cannot represent
  that complete behavior. The manual does not name a lock register or fully
  specify post-read values; the audit does not invent them.
- Recorded the RDNA3 STATUS privileged writable-bit list, scratch-write
  privilege differences from rocjitsu, exception-register replacement versus
  arithmetic accumulation, FLUSH_IB scope ambiguity, and shader-versus-host
  clock-read protocols. These are explicit implementation gates.
- Validation: extracted both OPR_HWREG maps from the machine-readable ISA XML,
  checked all table IDs/names against them, and checked the behavioral notes
  against manual sections 3.4/16.2 and rocjitsu's HWREG implementation. This was
  documentation-only; the last production gates remain 950 Clang/GCC tests and
  12 FP16 ternary sanitizer tests from `422fbfd`.

## FP32 FMA architecture and guest-mode policies

- Split FMA, FMAC and FMA_DX9_ZERO into unsuffixed RDNA3 and `_rdna4` APIs in
  both wave widths. FMAC_DX9_ZERO is RDNA3-only in the XML and keeps only its
  unsuffixed APIs. RDNA3 OMOD is disabled with preserved output denormals or
  IEEE mode; RDNA4 always applies it. This follows the manuals' OMOD rule and
  rocjitsu's effective_omod/effective_vop3_omod_f32 handling.
- Ordinary FMA/FMAC now accept guest input/output-denormal flushing and IEEE
  mode in scalar, eight-lane AVX2 and sixteen-lane AVX512 paths. DX9-zero forms
  retain their fixed input/output flushing and zero-product rule, independent
  of guest flush flags; architecture-specific OMOD gating still applies.
- Both architectures share the arithmetic kernels and DPP routing. Existing
  RDNA4 captures/tests/benchmarks now call the suffixed APIs. The RDNA3-only
  DX9 accumulator routing test explicitly enables output flushing when comparing
  against RDNA4 FMA with OMOD enabled. New tests cover every guest-mode
  combination, all four OMOD values, normal/subnormal operands through each
  input, generated subnormal outputs, masks, both widths and accumulator forms.
- Optional exception reporting remains unsupported in exact semantics; loose
  execution skips it. No host FP control writes were introduced. Literal FMA,
  FP16 FMA and dual-issue architecture policies remain separate audit items.
- Validation and DPP scratch allocation are centralized without burdening ordinary
  FMA: only the DPP helper validates routing and stages a temporary VGPR. Flags
  and low modifier bits are checked once before arithmetic, and zero-mask DPP
  calls still validate their descriptors before returning.
- Performance against `17bea5f`: eleven alternating samples pinned to CPU 2,
  each at least 50 ms thread CPU time. Ordinary RDNA4 FMA AVX2 medians changed
  17.189 -> 16.301 ns unmodified and 19.675 -> 19.250 ns with ABS/NEG/OMOD/CLAMP;
  AVX512 changed 7.848 -> 7.147 ns and 9.115 -> 8.073 ns. DX9-zero AVX2 changed
  21.242 -> 21.313 ns and 23.477 -> 23.591 ns; AVX512 changed 7.620 -> 7.701 ns
  and 10.447 -> 10.664 ns. These are limited finite-input checks, not overall
  library speedups. Raw samples and driver/parent objects are under
  `../goc-build/coverage-work/fma-architecture-*`, including
  `fma-architecture-timing.json`.
- Final validation: all 951 tests pass under Clang 21 and GCC 15 Release; all
  20 Arithmetic/InstructionFlags tests pass under ASan/UBSan. Seven audit-tool
  tests pass. Regenerated ISA/architecture/global-state inventories: the latter
  now has 616 reporting signatures, 274 resolved names, zero mismatches and
  342 unresolved shared signatures. Logs:
  `../goc-build/coverage-work/fma-architecture-*-tests.log`.

## FP16 FMA architecture and guest-mode policies

- Split FMA/FMAC/FMAMK/FMAAK into unsuffixed RDNA3 and `_rdna4` APIs, in both
  wave widths. Adapted rocjitsu's shared fp_mode::fma_f16, quiets_nan,
  effective_f16_omod and finish_fma_f16 rules: RDNA3 gates OMOD on output
  denormals/IEEE mode and preserves a selected signaling NaN when IEEE is off;
  RDNA4 quiets selected NaNs and applies OMOD independently of those settings.
- The RNE exact result model now accepts input/output-denormal flushing. Input
  flushing precedes exceptional-input classification; output flushing preserves
  zero signs and uses the pre-exponent-limit rounding boundary at the smallest
  normal. OMOD still rounds to FP16 before scaling. The existing integer
  intermediate avoids changing or depending on the host rounding controls.
- Scalar and eight-lane loose SIMD paths support the guest flush flags. Loose
  NaN payload/quiet-bit behavior remains approximate. Existing RDNA4 result and
  exception corpora were migrated to the suffixed APIs without changing their
  captured expectations. This is a rocjitsu-derived RDNA3 model, not a claim
  of a newly captured native RDNA3 hardware corpus.
- RDNA3 exception reporting and RDNA4 reporting with input/output flush enabled
  reject non-null exact requests until faithful reporting for these policies is
  verified. Null exact requests retain the result model; loose requests skip
  reporting without error. Existing RDNA4 no-flush reporting remains enabled.
- Tests cover all four operand forms, both wave widths, half selectors, all
  guest-mode combinations, all allowed OMOD settings, masks, NaN priority,
  signaling bits and host-control preservation. An independent integer oracle
  covers a Cartesian boundary corpus under both loose and exact semantics.
  Generated unsupported-reporting tests now distinguish the RDNA3 and RDNA4
  forms. Packed, mixed and dual-issue FMA policies remain separate audit items.
- Performance against `f9b7063`: eleven alternating samples pinned to CPU 2,
  each at least 50 ms thread CPU time. RDNA4 loose AVX2 FMA medians changed
  27.657 -> 28.218 ns unmodified (+2.0%) and 43.244 -> 44.717 ns with
  ABS/NEG/OMOD/CLAMP (+3.4%); literal FMAMK changed 23.228 -> 24.607 ns (+5.9%).
  Exact scalar FMA changed 792.706 -> 825.748 ns (+4.2%) and
  917.987 -> 950.263 ns (+3.5%); exact FMAMK changed 782.899 -> 803.665 ns (+2.7%).
  The shared scalar helper now carries saturation and guest policy in one flag
  word instead of adding a seventh per-lane argument. These finite-input checks
  expose modest overhead from guest controls; they are not a whole-library
  performance result. Artifacts: `../goc-build/coverage-work/half-fma-architecture-*`,
  including `half-fma-architecture-timing.json`.
- Final validation: all 954 tests pass under Clang 21 and GCC 15 Release;
  all 37 HalfFma/PackedFma/ScalarFma/ExceptionRegisters/InstructionFlags tests
  pass under ASan/UBSan. Seven audit-tool tests pass. The global-state audit
  now has 624 reporting signatures, 290 resolved names, zero mismatches and
  334 unresolved shared signatures. Logs:
  `../goc-build/coverage-work/half-fma-architecture-*-tests.log`.

## Packed FP16 FMA architecture policies

- Split `v_pk_fma_f16` and `v_pk_fmac_f16` into unsuffixed RDNA3 and
  `_rdna4` APIs for both wave sizes. rocjitsu's generated shared handlers
  (lines 17495 and 17594/17627, source hash in `ARCHITECTURE_AUDIT.md`) use
  the same `fp_mode::fma_f16` model as ordinary FP16 FMA, including
  architecture/IEEE-dependent NaN quieting. Both packed halves now use GoC's
  existing architecture-aware result helper, rather than duplicating its rules.
- Guest input/output denormal flags and IEEE mode are accepted. The existing
  eight-lane loose SIMD kernel handles flushing without falling back to scalar.
  Selected signaling NaNs remain unquieted in RDNA3 exact mode with IEEE off;
  RDNA4 quiets them. Existing RDNA4 result/exception fixtures retain their raw
  expectations and now call the suffixed APIs. No new native RDNA3 capture is
  claimed; that policy is derived from rocjitsu.
- New tests exercise both architectures, wave widths, FMA/FMAC, both semantics,
  baseline/AVX2, all eight guest-policy combinations, packed selectors,
  negation/clamp, masks and unchanged outputs on unsupported reporting errors.
  Exact RDNA3 reporting and RDNA4 reporting with flushing remain explicitly
  unsupported. Null reporting works; loose reporting is skipped. Wave64
  preflight checks these limits before any writes.
- Packed FMAC still exposes the unmodified operand form. Its VOP3 ABS/NEG/CLAMP
  support (and applicable encoding-specific behavior) remains to be implemented;
  this policy split is not a completeness claim for all encodings or FP modes.
- Against `7a33d4b`, eleven alternating CPU-2-pinned samples with at least 50 ms
  thread CPU time measured RDNA4 packed FMA loose AVX2 at 38.278 -> 39.470 ns
  (+3.1%) unmodified and 43.724 -> 44.572 ns (+1.9%) with packed selectors,
  negation and clamp. Exact scalar medians were 1633.945 -> 1642.266 ns (+0.5%)
  and 1655.012 -> 1670.885 ns (+1.0%). These finite-input measurements are not a
  whole-library performance claim. Raw samples and drivers are under
  `../goc-build/coverage-work/packed-fma-policy-*`.
- Validation: 955/955 tests pass with Clang 21 and GCC 15 Release; 17/17
  PackedFma/ExceptionRegisters tests pass with ASan/UBSan. Seven audit-tool
  tests pass. The refreshed global-state audit has 628 signatures, 298 resolved
  register names, zero mismatches and 330 still requiring architecture review.

## Packed FMAC VOP3 modifiers

- `v_pk_fmac_f16` and its RDNA4/Wave64 variants now accept ordinary ALU
  ABS/NEG for A/B and CLAMP. Each modifier applies to both packed components;
  D remains an unmodified accumulator. All OMOD values are accepted and ignored
  as specified by rocjitsu's `effective_f16_omod(..., packed_result=true, ...)`
  for GFX11/GFX12. C modifiers and half-selection flags remain invalid.
- Both manuals allow VOP2 instructions to use VOP3 controls (RDNA3 section
  16.7.1 / RDNA4 section 16.7.1). The component rules follow rocjitsu's
  `execute_v_pk_fmac_f16_vop3` (shared handler line 17627) and its FP16 model.
  This closes the VOP3 modifier gap recorded in the previous slice; DPP/other
  encoding-specific behavior is not implied by this change.
- Scalar and AVX2 paths share the existing packed FMA computation. FMA obtains
  per-component ALU modes from packed selectors; FMAC uses corresponding halves
  and applies ordinary ALU modifiers. Both compile from the same source loop,
  with a boolean template parameter distinguishing the operand forms. A single
  compiled SIMD kernel accepting arbitrary per-half modes regressed existing
  loose FMA by about 9%; the specialization removes its unused ABS checks.
  SIMD object text grows from 1,793 to 3,221 bytes. No scalar fallback is added.
- A new independent-oracle test covers all 128 modifier combinations, eight
  guest-policy combinations, both semantics/architectures/wave widths, every
  detected CPU level, masks, and destination aliases with either input.
  Validation tests still reject unsupported instruction bits, including with
  zero EXEC. Existing RDNA4 exception and result captures remain unchanged.
- A reporting test also verifies that ABS/NEG changes invalid infinity
  cancellation and that RDNA4 CLAMP suppresses its exception flags, with sticky
  state and an active uppermost lane in each wave width.
- Final performance against `f6cdd8c`: eleven alternating CPU-2-pinned samples,
  at least 50 ms thread CPU time each. Existing RDNA4 packed FMA loose AVX2
  medians were 39.549 -> 39.563 ns unmodified and 44.748 -> 44.758 ns with
  selectors/negation/clamp (both within 0.1%). Exact scalar medians were
  1645.582 -> 1651.787 ns and 1672.163 -> 1676.021 ns (both within 0.4%).
  This checks the existing kernel, not all library instructions or new FMAC
  modifier workloads. Samples/drivers: `../goc-build/coverage-work/packed-fmac-modifiers-*`.
- Validation: all 957 tests pass under Clang 21 and GCC 15 Release; all 19
  PackedFma/ExceptionRegisters tests pass under ASan/UBSan. `PLAN.md` now also
  explicitly distinguishes implemented RDNA4 no-flush FP16 reporting from
  unsupported RDNA3/flushing requests. Logs use the artifact prefix above.

## FP32 literal FMA architecture names and guest denormals

- Split FMAMK/FMAAK FP32 into unsuffixed RDNA3 and `_rdna4` entry points,
  including Wave64. Their architectural exception outputs are `trapsts` and
  `excp_flag_user`, respectively. rocjitsu's shared handlers (lines 11899 and
  12194, fingerprint in `ARCHITECTURE_AUDIT.md`) route both through its
  architecture/IEEE-aware FMA model. Existing callers now explicitly select
  the RDNA4 forms.
- Guest input/output-denormal flushing is implemented in the shared scalar,
  eight-lane AVX2 and sixteen-lane AVX-512 literal paths, including flushing
  the scalar literal before arithmetic. IEEE mode is accepted. The result
  contract remains loose: NaN payload/quieting need not match the architecture;
  both architecture APIs currently share that approximate arithmetic kernel.
  Exact arithmetic and faithful reporting remain unimplemented, rather than
  inferred from matching finite-input results. Non-null exact reporting rejects
  before writes; loose reporting leaves the state untouched.
- New tests cover all guest-policy combinations, both literal placements,
  architectures and wave sizes, every CPU level, normal/subnormal/NaN/infinity
  inputs and literals, masks, destination aliases, and unchanged results/state
  on rejected reporting. The reference flushes raw encodings around host FMA
  under the documented loose host-environment requirements. No native RDNA3
  bit-exact result claim is made by these tests.
- Performance against `c82cab2`: eleven alternating CPU-2-pinned samples,
  minimum 50 ms thread CPU time. AVX2 FMAMK/FMAAK medians changed
  13.117 -> 13.518 ns (+3.1%) and 12.602 -> 13.192 ns (+4.7%); AVX-512 changed
  4.712 -> 4.913 ns (+4.3%) and 3.615 -> 3.899 ns (+7.9%). Guest-control checks
  add a fraction of a nanosecond to these short no-modifier kernels. No additional
  duplicate fast kernel was introduced for that sub-nanosecond cost. These
  finite-input results are not a whole-library performance claim. Artifacts:
  `../goc-build/coverage-work/literal-fma32-policy-*`.
- Validation: 958/958 tests pass under Clang 21 and GCC 15 Release; 30/30
  LiteralFma/Arithmetic/ExceptionRegisters tests pass under ASan/UBSan. Seven
  audit-tool tests pass. The reporting-name audit has 632 signatures, 306
  resolved names, zero mismatches and 326 awaiting architecture review.

## Mixed FMA architecture and guest-policy audit

The denormal-policy assumptions in this initial rocjitsu-derived slice were
superseded by the native hardware correction below. In particular, MIXLO/MIXHI
do honor denormal controls, and FP16 input flushing occurs before widening in
all mixed FMA forms. The initial nondefault reporting restriction is now removed
for RDNA4 MIXLO/MIXHI following the new captures.

- `v_fma_mix_f32`, `v_fma_mixlo_f16`, and `v_fma_mixhi_f16` now distinguish
  unsuffixed RDNA3 and `_rdna4` APIs, in both wave widths. The distinction
  includes their native implicit register contracts (`trapsts` versus
  `excp_flag_user`), not just result bits. RDNA3 no longer inherits the RDNA4
  capture-derived guarantee of no MIXLO/MIXHI exception updates: its non-null
  exact reporting is explicitly unsupported.
- rocjitsu shared handlers 11750/11794/11837 (fingerprint in
  `ARCHITECTURE_AUDIT.md`) route MIX_F32 through architecture-aware FP32
  arithmetic, and MIXLO/MIXHI through `fma_f32_to_f16_nearest_environment`.
  The latter does not consume denormal or IEEE controls. GoC retains its
  integer direct-to-half result model and now accepts those ignored controls.
  Its RDNA3 results follow the shared model; no native RDNA3 corpus is claimed.
- MIX_F32 now applies guest input/output flushing in scalar and eight-lane
  AVX2 paths. FP16 inputs are widened first, so an FP16 subnormal is not an
  FP32 subnormal to be flushed. IEEE is accepted; FP32 loose NaN behavior
  remains approximate. MIXLO/MIXHI keep their four-lane loose SIMD path.
- New integer-oracle tests cover both architectures and wave widths, all eight
  source-format and guest-policy combinations, all CPU levels, half selectors,
  ABS/NEG/CLAMP, masks, both available semantics and reporting error atomicity.
  Existing RDNA4 numeric, DPP and no-exception captures keep their expectations
  and now explicitly call the suffixed APIs. DPP validation covers both names.
- RDNA4 MIXLO/MIXHI no-exception reporting remains enabled only for the captured
  default guest controls. Nondefault flags currently reject non-null exact
  reporting pending hardware verification; null reporting still computes the
  result model and loose requests always skip reporting. Existing captures do
  not by themselves establish exception behavior under every new guest setting.
- Keeping runtime guest checks in the default MIX_F32 SIMD kernel cost 7–16%
  in an initial timing check. The final implementation selects default versus
  guest-controlled SIMD before the loop. Both compile from one source template;
  the default entry retains its original six-argument interface and skips the
  new checks. SIMD object text grows from 9,499 to 10,495 bytes. The FP16-output
  kernel is unchanged.
- Final timing against `93003ea`: eleven alternating CPU-2-pinned samples with
  at least 50 ms thread CPU time. Loose AVX2 MIX_F32 medians changed
  18.725 -> 18.799 ns (+0.4%) for all-FP32 inputs and 21.191 -> 21.567 ns
  (+1.8%) for mixed inputs with NEG/CLAMP. MIXLO medians changed
  70.097 -> 70.113 ns (+0.02%) and 80.540 -> 80.897 ns (+0.4%). These are
  finite-input checks of the existing default paths, not a whole-library
  performance claim. Artifacts: `../goc-build/coverage-work/mixed-fma-policy-*`.
- Validation: all 959 tests pass on Clang 21 and GCC 15 Release; all 24
  MixedFma/ExceptionRegisters/InstructionFlags tests pass under ASan/UBSan.
  Seven audit-tool tests pass. The reporting-name audit now has 638 signatures,
  318 resolved names, zero mismatches and 320 pending architecture review.

## Native mixed-FMA denormal policies and exception reporting

Historical entry: newly inferred exact reporting was removed in the scope correction
below; FRACT exact results were also removed. See current public API comments.

- Extended `tests/capture_arithmetic_exceptions.py` so the generated probe accepts
  a runtime FP_DENORM nibble (0..15); default captures remain unchanged. Added
  the FP32 mixed-FMA family to the generator. RX 9070/gfx1201 captures exercise
  all sixteen independent FP32/FP16 input/output-denormal settings, both output
  halves, source formats/half selections, ABS/NEG/CLAMP, FP16 saturation, 4,096
  edge triples and 4,096 pseudorandom triples.
- All 16,777,216 MIXLO/MIXHI exception reads were zero. RDNA4 optional exact
  reporting therefore now accepts guest denormal flags and leaves the register
  unchanged. RDNA3 reporting remains unverified and explicitly unsupported.
- The captures contradicted rocjitsu's ignored-denormal mixed-half model. Input
  flushing uses each selected source's original FP16/FP32 type; output flushing
  uses the destination type. Corrected scalar and SIMD paths accordingly,
  including FP16 sources of MIX_F32. Output-half rounding retains the integer
  exact intermediate; the four-lane loose kernel implements the same flushing.
- `tests/mixed_fma_guest_hardware.h` stores word-wise result/exception digests
  for all 4,194,304 records with the four coupled denormal policies exposed by
  GoC. Tests reproduce those digests on exact scalar and loose SIMD paths at all
  available CPU levels. Another native probe covers both signs around the FP16
  normal boundary: flushing begins below `2^-14 - 2^-26`, rather than merely
  testing whether the final FP16 encoding is subnormal. Literal boundary tests
  and an independent integer oracle cover this distinction. Host-FP-control
  preservation tests now include guest denormal flags.
- Default SIMD kernels remain separate compiled specializations of the same
  source loops; the new checks run only for nondefault policies. Independent
  FP16 versus FP32 denormal selections are not yet exposed by the public flags.
  A further 8,388,608 MIX_F32 flag reads were zero; its general exact numerical
  and reporting contracts remain unfinished. No RDNA3 hardware claim follows
  from these RDNA4 captures.
- Raw capture summaries, generated probes and boundary values are in
  `../goc-build/coverage-work/mixed-fma-guest-captures/`; the source generator
  and compact regression fixtures are checked in. The implementation continues
  to leave host FP controls untouched; only the GPU capture probes set MODE.
- Default-path performance against `13f6238`: eleven alternating CPU-2-pinned
  samples, at least 50 ms thread CPU time. MIX_F32 AVX2 medians were
  18.688 -> 18.679 ns and 21.444 -> 21.465 ns; MIXLO medians were
  72.021 -> 71.169 ns and 80.396 -> 80.205 ns. Changes are within about 1.2%
  and do not establish a meaningful regression. These finite-input timings
  cover existing default loose paths; they are not a whole-library result.
  Timing artifacts use `../goc-build/coverage-work/mixed-guest-fix-*`.
- Validation: all 961 tests pass under Clang 21 and GCC 15 Release. All 35
  MixedFma/Interp16/ExceptionRegisters tests pass under ASan/UBSan, including
  the shared interpolation users of the modified rounding helper. Capture
  generator syntax and generated GPU probes were also exercised successfully.

## Exact MIX_F32 results and RDNA4 reporting

Historical entry: newly inferred exact reporting was removed in the scope correction
below; FRACT exact results were also removed. See current public API comments.

- Added scalar nearest-even exact MIX_F32 results using the existing integer
  fused intermediate and an integer FP64-to-FP32 rounding helper. Rounding
  handles gradual underflow, signed zeros, overflow and tininess before exponent
  packing without reading or changing host FP controls. NaN ordering follows
  rocjitsu `fp_mode::fma_f32`: invalid zero-times-infinity wins, otherwise the
  first NaN wins and architecture/IEEE controls quieting.
- Both RDNA3 and RDNA4 numeric APIs expose the model. RDNA3 is model-derived,
  not newly hardware-captured; its non-null exact global-state reporting remains
  unsupported. RDNA3 mixed CLAMP explicitly models DX10_CLAMP=1; the alternative
  NaN-preserving setting is not yet exposed. Loose paths retain SIMD dispatch.
- RDNA4 MIX_F32 exact reporting now leaves EXCP_FLAG_USER unchanged, backed by
  the preceding 8,388,608 zero flag captures across all sixteen denormal settings.
  Its 2,097,152 raw-result records for the four exposed coupled policies are
  stored as compact digests alongside the existing FP16-output fixtures. Exact
  CPU results match those digests including NaN payloads, both source-half
  selections, modifiers, formats, saturation settings and random/edge inputs.
- Expanded integer-oracle tests to exact FP32 results, RDNA3 IEEE NaN policies,
  all 8,192 modifiers and both wave sizes. Host-control tests now exercise exact
  FP32 results under every host rounding mode and FTZ/DAZ setting. Updated the
  former unsupported-reporting test to retain that assertion for RDNA3, and
  test successful no-op RDNA4 reporting with empty EXEC.
- A targeted RX 9070 capture checks FP32 tininess where ordinary subnormal
  rounding produces minimum normal but exponent-unbounded rounding remains
  tiny. Literal regression cases cover both signs and wave widths. Artifacts:
  `../goc-build/coverage-work/mixed-fma-guest-captures/float-boundaries/` and
  `float-diagonal-hashes.json` in its parent directory.
- Default loose-path timing against `dcfba15`: eleven alternating CPU-2-pinned
  samples, minimum 50 ms thread CPU time. MIX_F32 medians changed
  18.708 -> 18.808 ns and 21.779 -> 21.751 ns; MIXLO changed
  70.811 -> 71.554 ns and 80.312 -> 80.415 ns. Differences are within about
  1.1%; these existing-path checks are not an exact-path speed claim or a
  whole-library benchmark. Artifacts use
  `../goc-build/coverage-work/mixed-float-exact-*`.
- Validation: all 962 tests pass under Clang 21 and GCC 15 Release; all 42
  MixedFma/HalfFma/ExceptionRegisters tests pass under ASan/UBSan, covering the
  existing integer-FMA users as well as the new exact FP32 path.

## Exact FP32 literal FMA results

- FP32 FMAMK/FMAAK now implement exact nearest-even results in both wave sizes,
  with guest input/output denormal policies and RDNA3 versus RDNA4 NaN quieting.
  The implementation shares the integer FP32 FMA algorithm with MIX_F32:
  `fma_scalar.h` owns NaN selection and special values, while `fma_integer.h`
  owns finite fused evaluation and rounding. Instruction-specific operand
  selection and dispatch remain in their existing implementation files.
- Exact paths use scalar integer arithmetic at every CPU level; loose AVX2 and
  AVX512 paths retain their existing kernels. Neither path explicitly changes
  host FP controls. Both architectures still reject non-null exact global-state
  reporting; native captures contain nonzero flags, so reporting cannot be a
  no-op as it is for mixed FMA. No RDNA3 hardware validation is claimed.
- RX 9070 gfx1201 fixtures cover 65,536 result records across the two literal
  forms and four coupled denormal policies, with literal `0x3f800001`. Edge and
  deterministic random input records are compared as raw result digests for
  both CPU wave sizes and all available CPU levels. The checked-in capture
  generator now supports FP32 literal forms; full capture artifacts are in
  `../goc-build/coverage-work/literal-fma32-exact-captures/`.
- Independent integer-oracle tests cover other literals, random triples, NaN
  payload ordering, masks, aliases, architecture IEEE differences and guest
  denormal policies. Host-environment tests cover all rounding modes and FTZ/DAZ,
  preserving control bits and preexisting exception flags. Wave64 validation
  tests now distinguish exact-capable literal forms from loose-only ordinary FMA.
- Isolated the scalar loop in a non-inlined helper so its integer scratch frame
  does not burden SIMD dispatch. Eleven alternating CPU-2-pinned samples of
  at least 50 ms thread CPU time against `e6f0f17` gave FMAMK/FMAAK medians of
  13.544 -> 13.172 ns / 13.202 -> 12.811 ns for AVX2 and
  4.820 -> 4.672 ns / 3.920 -> 3.553 ns for AVX512. These four finite-input
  loose-path measurements show no regression; they do not characterize exact
  performance or the whole library. Artifacts use
  `../goc-build/coverage-work/literal-fma32-exact-*`.
- Validation: all 964 tests pass with Clang 21 and GCC 15 Release. All 49
  LiteralFma/MixedFma/Arithmetic/ExecMask tests pass under ASan/UBSan. Native
  capture generation, compilation and execution also succeeded.

## RDNA4 FP32 literal FMA exception reporting

Historical entry: newly inferred exact reporting was removed in the scope correction
below; FRACT exact results were also removed. See current public API comments.

- RDNA4 FMAMK/FMAAK now accumulate exact EXCP_FLAG_USER updates in both wave
  sizes for all four exposed guest denormal policies. Classification uses input
  bits and the same integer round-to-odd intermediate as exact results. It
  reports invalid, input-denormal, overflow, underflow and inexact; NaN/invalid
  priority and tininess before exponent packing follow native captures. No host
  FP control writes, exception reads or exception-clearing operations are used.
- Only active lanes contribute, and updates OR into existing state. Empty EXEC
  leaves state untouched after validation. Invalid calls preserve VGPRs and
  state. Loose semantics bypass reporting even with a non-null pointer. RDNA3
  TRAPSTS reporting remains explicitly unsupported and is not inferred from
  RDNA4 measurements.
- The ordinary edge/random corpus supplies 65,536 captured exception records.
  Another 131,072 result/flag pairs cover underflow and overflow boundaries,
  with literals immediately below and above one and all four denormal policies.
  CPU Wave32/Wave64 calls match every digest. The generator supports explicit
  literal bits and literal-FMA rounding-boundary inputs. Fixtures and generator
  are checked in; raw captures and manifests remain under
  `../goc-build/coverage-work/literal-fma32-exact-captures/`.
- Tests exercise lane-by-lane flags, accumulation across both Wave64 halves,
  inactive signaling NaNs, source/destination aliases, every CPU level, both
  exact strictness settings, error preservation, and all host rounding/FTZ/DAZ
  settings. Loose calls explicitly preserve sentinel global state. Existing
  mixed FMA callers omit reporting and retain their no-op exception contract.
- VOPD literal wrappers continue to reject non-null exact reporting explicitly;
  they delegate numerical results but their architecture/state audit is still
  pending. Ordinary-VOP2 captures do not establish VOPD exception behavior.
- Validation: all 966 tests pass under Clang 21 and GCC 15 Release; all 60
  LiteralFma/MixedFma/Arithmetic/ExecMask/ExceptionRegisters/Dual tests pass under
  ASan/UBSan. The native capture generator compiled and ran successfully.
- Eleven alternating CPU-2-pinned samples, at least 50 ms thread CPU time,
  compare default loose calls against `deaa3cc`, with both null and non-null
  state pointers. FMAMK AVX2 medians were 13.261 -> 13.515 ns (null) and
  13.024 -> 13.262 ns (non-null); FMAAK AVX2 stayed within 0.1%. AVX512 changes
  ranged from -0.1% to -8.0%, all below 0.4 ns absolute. The largest slowdown
  was 1.9%; these measurements support preserving the loose path, not a general
  speedup claim. Artifacts use `../goc-build/coverage-work/literal-fma32-reporting-*`.

## Exact ordinary FP32 FMA/FMAC results

- Ordinary FP32 FMA/FMAC now expose exact nearest-even results for RDNA3 and
  RDNA4, Wave32 and Wave64. A scalar integer path reuses `fma_value_f32` for
  fused arithmetic and NaN priority, then applies output scaling and CLAMP.
  Source modifiers, DPP8/DPP16 routing, aliases and masks retain their existing
  decoding/staging paths. The exact scalar frame is isolated from loose SIMD
  dispatch; existing AVX2/AVX512 arithmetic kernels are unchanged.
- Output scaling follows rocjitsu `fp_mode::apply_omod_f32`: flush tiny unscaled
  intermediates, adjust normal exponents, flush newly tiny halves and apply
  CLAMP last. The integer FMA model detects tininess before exponent packing.
  RDNA3 gates OMOD on output flushing and IEEE being clear, preserving selected
  signaling NaNs when IEEE is clear; RDNA4 always applies OMOD and quiets NaNs.
  RDNA3 CLAMP models DX10_CLAMP=1. No new RDNA3 hardware validation is claimed.
- RX 9070 gfx1201 captures cover 2,097,152 result records: FMA/FMAC, edge/random
  and rounding-boundary corpora, sixteen modifier configurations and four
  coupled denormal policies. Exact CPU results match raw-result digests at all
  available CPU levels and both wave sizes, including NaN payloads. Checked-in
  capture generation now accepts FP32 ordinary FMA/FMAC. Raw probes, records
  and manifests are in `../goc-build/coverage-work/fma32-exact-captures/`.
- Independent integer-oracle tests cover every source/output modifier, both
  architectures, aliases and masks. Separate DPP tests vary host rounding and
  FTZ/DAZ while checking exact results, preserved controls and sticky flags.
  C99 API tests now explicitly exercise exact ordinary FMA. Previous tests
  that assumed exact requests were rejected now test reserved semantics instead
  or verify successful exact empty-EXEC calls.
- Optional exact exception reporting for ordinary FMA/FMAC remains explicitly
  unsupported for both architectures. DX9 forms remain loose-only. Enabling
  ordinary exact results does not claim either of these unfinished behaviors.
- Validation: all 970 tests pass under Clang 21 and GCC 15 Release. All 37
  FmaExact/Arithmetic/Fmac/Dpp8/Dpp16/ExecMask/Api tests pass under ASan/UBSan.
- Eleven alternating CPU-2-pinned samples, minimum 50 ms thread CPU time,
  compare loose paths against `9d24d32`. FMA AVX2 medians were
  16.310 -> 16.374 ns (default) and 19.390 -> 19.441 ns (modifiers); AVX512
  medians were 7.161 -> 7.358 ns and 8.032 -> 8.340 ns. The DX9 control cases
  changed by less than 1%. All eight measured changes were under 4%; this is
  a dispatch-regression check, not an exact-path or whole-library speed claim.
  Artifacts use `../goc-build/coverage-work/fma32-exact-*`.

## RDNA4 ordinary FP32 FMA/FMAC exception reporting

Historical entry: newly inferred exact reporting was removed in the scope correction
below; FRACT exact results were also removed. See current public API comments.

- Exact RDNA4 FMA/FMAC now accumulate EXCP_FLAG_USER for all four exposed
  denormal policies, both wave sizes, source/output modifiers and DPP routing.
  The shared integer FMA classifier supplies invalid, input-denormal, overflow,
  underflow and inexact. OMOD suppresses underflow/inexact and reports newly
  scaled overflow; CLAMP suppresses every exception. Only participating lanes
  after DPP routing contribute. Updates OR into existing wave-wide state.
- Native flag records from the preceding 2,097,152-result RX 9070 capture corpus
  now have their own regression digests. API tests compare every flag record
  with a single participating lane, rotating its position and CPU level and
  varying strictness. Inactive lanes contain signaling NaNs to expose accidental
  contributions. Both Wave32 and Wave64 match all native digests.
- DPP/mask/alias tests independently stage each lane to check accumulated state
  against unrouted calls, in addition to the independent result oracle. These
  run with every host rounding/FTZ/DAZ setting and preserve host controls and
  sticky flags. Validation tests preserve every output on invalid calls and
  verify empty EXEC and loose non-null reporting behavior.
- Loose dispatch still bypasses reporting; RDNA3 reporting remains unsupported.
  VOPD FMAC keeps its explicit unsupported-reporting guard pending separate
  architectural validation, as the literal VOPD wrappers already do. DX9 exact
  numerical semantics remain unfinished.
- No new native capture was needed: flag digests were derived from the saved
  records in `../goc-build/coverage-work/fma32-exact-captures/`. The checked-in
  capture generator reproduces both result and exception records; the new local
  manifest is `exception-hashes.json` alongside the original result manifest.
- Validation: all 972 tests pass under Clang 21 and GCC 15 Release; all 48
  FmaExact/Arithmetic/Fmac/Dpp8/Dpp16/ExecMask/Api/ExceptionRegisters/Dual tests
  pass under ASan/UBSan. The final DPP test additionally seeds low exception
  bits to verify that reporting, including CLAMP suppression, never clears
  existing flags; it passes again in all three builds.
- Eleven alternating CPU-2-pinned samples of at least 50 ms thread CPU time
  compare loose FMA against `201f8b1`, using default and nondefault modifiers,
  AVX2/AVX512 and null/non-null state pointers. The eight FMA changes ranged
  from -0.4% to +4.3%, at most 0.5 ns absolute; default AVX512 was
  7.369 -> 7.480 ns (null) and 7.340 -> 7.656 ns (non-null). DX9 controls changed
  by less than 1.6%. No reporting classifier runs in these loose paths. These
  measurements cover dispatch overhead, not exact-reporting performance or the
  whole library. Artifacts use `../goc-build/coverage-work/fma32-reporting-*`.

## Exact DX9-zero FP32 fused arithmetic

- Added exact RNE results for RDNA3/RDNA4 FMA_DX9_ZERO and RDNA3's FMAC_DX9_ZERO,
  in both wave sizes. These reuse the ordinary integer FMA implementation with
  the rocjitsu DX9 preprocessing rules: force input/output flushing and replace
  the product by positive zero if either modified, flushed factor is zero.
  NaN selection, fused rounding, OMOD and CLAMP then use the same shared code.
  RDNA3 OMOD gating uses the original guest settings, before forced flushing.
- RX 9070 gfx1201 raw-result fixtures cover 1,048,576 records across edge/random
  and underflow/overflow-boundary corpora, sixteen modifier configurations and
  four coupled denormal modes. All modes produced identical results, and both
  CPU wave sizes match at every available CPU level, including NaN payloads.
  The 2,097,152 accompanying exception records, including repeated FP16_OVFL
  settings, were all zero. DX9 APIs retain their no-global-state-output contract.
- The shared exact tests now cover DX9 source/output modifiers, guest policies,
  architecture NaN/OMOD distinctions, masks, aliases and DPP8/DPP16. Host rounding
  and FTZ/DAZ variations preserve results, controls and sticky flags. Literal
  regression cases cover zero times signaling NaN/infinity, positive-zero product
  addition, retained C payloads, signed flushed zeros and a fused-rounding witness.
- RDNA3 FMAC uses the same model and has direct tests; these RDNA4 captures do
  not constitute new RDNA3 hardware evidence. DX10_CLAMP=1 remains the modeled
  RDNA3 setting. Loose scalar/AVX2/AVX512 kernels remain unchanged.
- Capture generation now accepts `fma_dx9_zero 32`, with optional rounding
  boundaries. Native artifacts and manifests are in
  `../goc-build/coverage-work/dx9-exact-captures/`; source and compact digests
  are checked in. Validation tests formerly expecting exact rejection now test
  reserved semantics or successful exact empty-EXEC behavior.
- Validation: all 973 tests pass under Clang 21 and GCC 15 Release. All 35
  FmaExact/Arithmetic/Dpp8/Dpp16/ExecMask/Api tests pass under ASan/UBSan.
- Eleven alternating CPU-2-pinned samples, minimum 50 ms thread CPU time,
  compare loose paths against `fc3e22e`. Default DX9 AVX2 medians were
  21.176 -> 21.150 ns and AVX512 7.616 -> 7.652 ns; with source/output
  modifiers they were 23.649 -> 23.604 ns and 10.661 -> 10.700 ns. All sixteen
  DX9/ordinary-FMA control cases, including ordinary null/non-null state
  pointers, changed by less than 1%. This does not characterize exact-path or
  whole-library performance. Artifacts use `../goc-build/coverage-work/dx9-exact-*`.

## VOPD architecture split and exact fused reporting

Historical entry: newly inferred exact reporting was removed in the scope correction
below; FRACT exact results were also removed. See current public API comments.

- Split eight floating VOPD families: FMAC, FMAAK, FMAMK, MUL, ADD, SUB,
  SUBREV and MUL_DX9_ZERO. Unsuffixed APIs now use RDNA3 ordinary implementations
  and TRAPSTS; `_rdna4` APIs use RDNA4 implementations and EXCP_FLAG_USER.
  These are explicit wrappers over existing arithmetic, with no new execution
  abstraction. The three fused families demonstrate architecture-specific NaN
  quieting; the other families inherit their ordinary guest-policy contracts.
- Captured 196,608 RX 9070 gfx1201 fused result/exception pairs across all three
  fused forms, both X/Y slots and four denormal policies. Each matches its ordinary
  instruction's capture, including nonzero exception flags. Enabled exact RDNA4
  reporting by delegation, superseding the earlier temporary VOPD reporting
  guards. RDNA3 non-null exact reporting remains explicitly unsupported.
- Added native digest tests plus architecture/IEEE, guest-policy, CPU-level,
  alias, sparse-EXEC and output-preservation checks. Loose non-null reporting
  leaves state untouched. The capture generator accepts `--vopd` and `--vopd-y`;
  artifacts are in `../goc-build/coverage-work/vopd-fma-captures/`.
- The global-state audit now covers 646 signatures: 334 resolved matches,
  zero mismatches and 312 still requiring architecture review. The generated
  helper audit does not inspect VOPD execution bodies; this slice inspected
  rocjitsu's generated RDNA3/RDNA4 `vopd_exec.cpp` directly. API presence
  remains 806 of 809 in-scope mnemonics and is not a completeness claim.
- Validation: all 976 tests pass with Clang 21 and GCC 15 Release; all 19
  Dual/ExceptionRegisters/FmaExact tests pass under ASan/UBSan. The seven audit
  tool tests also pass. RDNA3 behavior is model-backed, not newly hardware-captured.
- Eleven alternating CPU-2-pinned samples, minimum 50 ms thread CPU time,
  compare old unsuffixed RDNA4 wrappers against the new `_rdna4` wrappers.
  Twenty loose cases cover FMAC/FMAMK/FMAAK/ADD/MUL, AVX2/AVX512 and null/non-null
  reporting. Median changes range from -14.9% to +3.44%; the largest slowdown
  is 0.532 ns for MUL. Fused wrappers generally improve after removing their
  reporting guards; their underlying SIMD kernels are unchanged. This measures
  these wrappers only, not whole-library performance. Raw samples are in
  `../goc-build/coverage-work/vopd-timing-samples.json`.

## RCP_IFLAG architecture-specific output scaling

- Split `goc_v_rcp_iflag_f32` and its Wave64 variant into unsuffixed RDNA3
  APIs with TRAPSTS and `_rdna4` APIs with EXCP_FLAG_USER. RDNA3 OMOD is disabled
  when guest output denormals are preserved or IEEE mode is set. This follows
  rocjitsu's `execute_v_rcp_iflag_f32_vop3` and `fp_mode::effective_omod`;
  forced flushing by the reciprocal operation does not alter guest-mode gating.
- Both variants share one implementation with explicit architecture branches,
  and the existing SIMD kernels remain unchanged. Existing RDNA4 captures and
  benchmarks now call `_rdna4`. RDNA3 loose calls skip state reporting; exact
  non-null reporting requests explicitly fail before operand access. Numerical
  strict-exact reciprocal is still unsupported on both architectures.
- New tests cover all source/output modifiers, all eight guest IEEE/denormal
  policies, both architectures/wave widths, DPP descriptors, sparse EXEC, aliases
  and CPU tiers. Powers-of-two inputs give directly checkable exact results
  within loose semantics. Additional error tests verify state/output preservation
  and optional reporting. No new RDNA3 hardware evidence is claimed.
- The architectural-state naming audit now has 648 signatures, 338 resolved
  matches, zero mismatches and 310 pending architecture review.
- Validation: all 978 tests pass under Clang 21 and GCC 15 Release, and all
  21 RcpIflag/InstructionFlags/ExceptionRegisters tests pass under ASan/UBSan.
  Seven audit-tool tests pass. The final test-only warning/style adjustments
  were rechecked in the Clang and sanitizer builds as well.
- Eleven alternating CPU-2-pinned timing samples, minimum 50 ms thread CPU time,
  compare the old RDNA4 entry point against `_rdna4`: eight loose cases cover
  AVX2/AVX512, default/combined modifiers and null/non-null reporting. Changes
  range from -0.97% to +4.8%, at most 0.309 ns slower. Default AVX512 medians
  were 6.364 -> 6.561 ns without reporting and 6.439 -> 6.748 ns with reporting.
  These are dispatch-level measurements, not whole-library results. Artifacts
  use `../goc-build/coverage-work/rcp-arch-*`.

## FP32 transcendental architecture policies

- Split SQRT/RCP/RSQ/EXP/LOG in both wave sizes into unsuffixed RDNA3 APIs
  with TRAPSTS and `_rdna4` APIs with EXCP_FLAG_USER. Both accept IEEE and
  denormal flags; these instructions force input/output flushing. RDNA3 IEEE
  mode disables OMOD, while the guest denormal flags do not gate it. This is
  the explicit `effective_omod(arch, 0, ieee, omod)` policy in rocjitsu's five
  generated VOP3 helpers, unlike RCP_IFLAG's guest-denormal argument. The audit
  generator now records this distinction to avoid overgeneralizing the rule.
- RDNA3 applies its OMOD policy then enters the same unary execution body as
  RDNA4; source routing, scalar arithmetic and SIMD kernels remain shared. Other unary
  operations retain their existing contracts and still require architecture
  review. Old RDNA4 hardware fixtures, tests and benchmarks use `_rdna4`.
- New policy tests cover all five operations, all eight guest IEEE/denormal
  combinations, every source/output modifier, both wave sizes, DPP, EXEC masks,
  aliases and CPU levels. A literal reciprocal witness distinguishes RDNA3
  IEEE-off (0.5) from IEEE-on (0.25), with RDNA4 producing 0.5 in both cases.
  These are model-backed RDNA3 tests, not new RDNA3 hardware captures.
- Exact numerical semantics and non-null exact exception reporting remain
  explicitly unsupported; loose calls leave optional state untouched. No host
  FP-environment mutation was introduced. The state-name audit now has 658
  signatures, 358 resolved matches, zero mismatches and 300 pending review.
- Validation: all 980 tests pass with Clang 21 and GCC 15 Release; all 36
  Unary/Arithmetic/ExecMask/InstructionFlags/ExceptionRegisters tests pass under
  ASan/UBSan. After consolidating the scalar execution bodies, those 36 tests
  pass again on both Release builds and sanitizers. Seven audit-tool tests pass.
- Eleven alternating CPU-2-pinned samples, minimum 50 ms thread CPU time, compare
  twenty loose cases with `4ae9212`: five operations, default/combined modifiers,
  null/non-null reporting and the AVX2 CPU tier. SQRT/RCP/RSQ SIMD changes are
  within 0.9%. Scalar LOG changes are within 1.6%; scalar EXP default is within
  0.4%, but combined modifiers measure 107.027 -> 122.531 ns (14.5% slower).
  The latter remains a measured limitation, not a claimed performance win.
  Disassembly of the final EXP execution body matches the baseline instruction
  for instruction except the accepted-flags immediate; its linked address moved
  by 256 bytes with the same alignment. This suggests layout sensitivity, but
  does not establish its cause. No speculative padding or alignment tuning was
  added. Raw samples use `../goc-build/coverage-work/unary-arch-*`; measurements
  cover these calls only, not whole-library performance.

## Captured exact FP32 reciprocal results

- Borrowed rocjitsu `util/amdgpu_rcp.h` at revision `ffc144c564c`, preserving
  its normalized cubic mapping and sparse residual tables. GoC adapts it to raw
  integer bits, C++17 and the existing portable `Uint128` helper. The source
  records exhaustive normalized gfx1100/gfx1201 fitting; no model coefficients
  were independently reinvented here.
- Enabled exact RCP results on RDNA3/RDNA4 in both wave sizes, including source
  modifiers, OMOD, CLAMP, DPP and aliases. The exact kernel uses integer bits
  throughout. RDNA3 IEEE-off preserves signaling NaNs; IEEE-on quiets them and
  disables OMOD. RDNA4 always quiets and applies OMOD. RDNA3 CLAMP models the
  existing DX10_CLAMP=1 contract. Non-null exact reporting still fails explicitly;
  this slice does not infer flags from the newly exact numerical result.
- Fresh RX 9070 gfx1201 captures cover every one of the 8,388,608 normalized
  mantissas and 1,048,576 edge/random/exponent/modifier records across four guest
  denormal policies. All raw result hashes match. The normalized API test rotates
  architectures, CPU levels and wave sizes; modifier captures are checked in both
  wave sizes. These fresh captures establish RDNA4 evidence; RDNA3 also relies
  on rocjitsu's original model provenance and direct policy witnesses.
- DPP/EXEC/alias tests run under all four host rounding modes and both FTZ/DAZ
  settings, checking unchanged controls and pre-existing sticky flags. Validation
  tests retain explicit rejection of unimplemented reporting and reserved semantics.
  Capture generation is checked in as `tests/capture_unary_exact.py`; native binaries
  and raw records are in `../goc-build/coverage-work/rcp-exact-captures/`.
- Validation: all 985 tests pass under Clang 21 and GCC 15 Release; all 25
  RcpExact/Unary/DppUnary/ExecMask/InstructionFlags/ExceptionRegisters tests
  pass under ASan/UBSan. The final EXEC-variable naming cleanup also passes
  the five exact reciprocal tests again in Clang.
- Eleven alternating CPU-2-pinned samples, minimum 50 ms thread CPU time,
  compare loose RDNA4 reciprocal against `8a67f39`: AVX2, default/combined
  modifiers and null/non-null state pointers. Median slowdowns are 0.59–1.47%,
  at most 0.284 ns. Default medians are 14.974 -> 15.062 ns without reporting
  and 15.162 -> 15.269 ns with reporting. Exact arithmetic is dispatched out of
  line; the loose SIMD kernel is unchanged. Artifacts use
  `../goc-build/coverage-work/rcp-exact-*`; these measurements do not characterize
  exact-mode throughput or whole-library performance.

## Captured exact FP32 square root and reciprocal square root

- Borrowed rocjitsu `util/amdgpu_sqrt.h` and `util/amdgpu_rsq.h` at revision
  `ffc144c564c`. Their fitted integer mappings and sparse residual tables are
  retained, with raw-bit C++17 interfaces and GoC's portable `Uint128` where
  needed. Both source models record exhaustive gfx1100/gfx1201 provenance.
- SQRT and RSQ now implement exact results on RDNA3/RDNA4 and both wave sizes.
  They use the same exact unary execution helper as RCP for source modifiers,
  integer OMOD/CLAMP, DPP, masks and aliases. Only the core mapping differs;
  there is no separate modifier or routing implementation for each new operation.
  RDNA3 IEEE/NaN policy remains explicit. Requested exact reporting is still
  rejected; numerical exactness alone does not certify exception classification.
- Fresh RX 9070 captures cover 16,777,216 normalized inputs per operation (all
  mantissas, both exponent parities), plus 1,048,576 modifier/special/exponent
  records per operation over four denormal policies. All 35,651,584 new raw
  result records match through the public APIs. RDNA3 additionally relies on
  the borrowed model's provenance; no new RDNA3 hardware capture is claimed.
- Renamed the reciprocal-only harness/generator to `unary_exact_test.cpp` and
  `capture_unary_exact.py`. The five shared tests are parameterized over RCP,
  SQRT and RSQ, retaining full normalized coverage, NaN/IEEE witnesses, DPP,
  aliases and unchanged host controls/sticky flags under all rounding/FTZ modes.
  Artifacts are in `../goc-build/coverage-work/{sqrt,rsq}-exact-captures/`.
- Validation: all 995 tests pass under Clang 21 and GCC 15 Release; all 35
  UnaryExact/Unary/DppUnary/ExecMask/InstructionFlags/ExceptionRegisters tests
  pass under ASan/UBSan. The renamed parameterized binary runs each of its five
  tests independently for all three operations.
- Eleven alternating CPU-2-pinned samples, minimum 50 ms thread CPU time,
  compare loose SQRT/RSQ and reciprocal controls against `221b29d`. Twelve AVX2
  cases cover default/combined modifiers and null/non-null state pointers.
  Median changes range from -1.51% to +2.74%, at most 0.531 ns slower. Default
  SQRT changes 15.046 -> 14.960 ns (null), and default RSQ 16.729 -> 16.785 ns.
  SIMD kernels remain unchanged. Artifacts use
  `../goc-build/coverage-work/roots-exact-*`; these do not measure exact throughput
  or whole-library performance.

## Captured exact FP32 exponential and logarithm

- Borrowed rocjitsu `util/amdgpu_exp.h` and `util/amdgpu_log.h` at revision
  `ffc144c564c`, retaining their integer reduction, coefficients and staged
  rounding. LOG's C++20 bit-width operations use GoC's existing portable zero
  counter through a small C++17 equivalent. No host floating arithmetic or
  environment manipulation is used by the exact path.
- Enabled exact EXP/LOG on both architectures and wave sizes through the same
  unary execution helper as RCP/SQRT/RSQ. All five share source modifiers,
  integer OMOD/CLAMP, forced output flushing, DPP and alias handling. Only their
  core mappings differ. RDNA3 IEEE-off signaling-NaN preservation and IEEE-on
  OMOD suppression remain explicit. Exact exception reporting remains unsupported.
- New RX 9070 captures cover every positive and negative FP32 input with magnitude
  in [0.5,2) for EXP (33,554,432), and every input in [0.5,2) for LOG (16,777,216),
  including both cancellation branches near one. Each also has 1,048,576
  modifier/guest-policy records; EXP inputs add overflow, underflow, tiny-input
  and finite fractional-domain boundaries. All 52,428,800 new result records
  match through public APIs. This is dense-domain validation, not an exhaustive
  sweep of every FP32 encoding; RDNA3 additionally relies on borrowed provenance.
- Extended the existing parameterized exact tests to all five operations,
  including host rounding/FTZ variations, unchanged controls/sticky flags,
  architecture witnesses, DPP, sparse EXEC and aliases. The capture generator
  now accepts `exp` and `log`. Raw records and probes are under
  `../goc-build/coverage-work/{exp,log}-exact-captures/`.
- Validation: all 1,005 tests pass with Clang 21 and GCC 15 Release; all 45
  UnaryExact/Unary/DppUnary/ExecMask/InstructionFlags/ExceptionRegisters tests
  pass under ASan/UBSan. The final explicit exact-operation dispatch list also
  passes all 25 parameterized exact tests again in Clang.
- Eleven alternating CPU-2-pinned samples, minimum 50 ms thread CPU time,
  compare twenty loose cases against `ad30bb4`: EXP/LOG plus SQRT/RSQ/RCP
  controls, AVX2 CPU tier, default/combined modifiers and null/non-null reporting.
  SIMD controls change by less than 0.8%. Scalar EXP/LOG range from -1.11% to
  +4.95%; the largest change is EXP with modifiers, 107.202 -> 112.510 ns.
  Default EXP changes 90.610 -> 91.423 ns, and LOG 91.551 -> 91.900 ns.
  The loose arithmetic bodies are unchanged; exact evaluation stays out of line.
  Artifacts use `../goc-build/coverage-work/exp-log-exact-*`; these measurements
  do not characterize exact throughput or whole-library performance.

### Exact RDNA4 unary exception reporting

Historical entry: newly inferred exact reporting was removed in the scope correction
below; FRACT exact results were also removed. See current public API comments.

- Implemented optional `EXCP_FLAG_USER` accumulation for exact FP32 RCP,
  SQRT, RSQ, EXP and LOG, in both wave sizes. The classifier uses integer bits;
  loose calls still bypass it and leave optional state untouched. RDNA3 exact
  `TRAPSTS` reporting remains explicitly unsupported pending matching evidence.
- Fresh gfx1201 captures cover 10,485,760 `(result, flags)` pairs: five operations,
  two 8192-input corpora, 32 source/output modifier combinations, and four guest
  denormal policies. The ordinary corpus matches every previously captured
  numerical result. The second corpus concentrates on perfect squares and their
  successors, powers of two and adjacent inputs, and integer/fractional EXP
  boundaries. All four denormal policies produce identical pairs; fixture hashes
  share those identical expectations while tests exercise every policy.
- CLAMP suppresses new flags; OMOD suppresses underflow/inexact but retains
  applicable invalid/divide-by-zero/overflow flags. Approximation status cannot
  be inferred solely from mathematical exactness: hardware sets INEXACT for
  `sqrt(9) == 3`, whereas powers of four avoid it. Targeted captures exposed this
  distinction before the classifier was accepted.
- The shared exact execution loop accumulates only participating lanes, including
  DPP zero-fill/routing and Wave64 halves. Tests cover preserved preexisting bits,
  masks, aliases, DPP8/DPP16, host rounding/FTZ/DAZ and sticky-flag preservation,
  empty EXEC, loose opt-out and validation before writes. Existing unsupported
  reporting tests retain the RDNA3 contract.
- `tests/capture_unary_exact.py --exceptions` generates the ordinary status probe;
  adding `--exactness` selects the concentrated corpus. One identical-input Wave32
  executes per operand, with status cleared/read around the tested instruction.
  Fixed the new probe's initial textual operand renumbering so it cannot rewrite
  a C modulo expression such as `%254`. Discarded those erroneous captures and
  regenerated both corpora before forming any fixture.
- Clang 21 and GCC 15 Release each pass **1,015/1,015** tests. Targeted Clang
  ASan/UBSan passes **48/48** unary and DPP tests. Logs are
  `/tmp/goc-unary-report-{clang,gcc,san}-tests.log`.
- Eleven alternating before/after samples, pinned to CPU 2 with at least 50 ms
  thread CPU time per case, compare against `584382d`. Twelve loose AVX2
  SQRT/RSQ/RCP cases (default/combined modifiers, null/non-null state) range
  from -0.85% to +0.10%. Eight scalar EXP/LOG cases range from -4.62% to +5.13%;
  the largest regression is LOG with combined modifiers and non-null state,
  112.565 -> 118.335 ns. Reporting work remains exclusively in the exact helper;
  these measurements do not establish a whole-library or exact-path speedup.
  Raw samples and binaries use `../goc-build/coverage-work/unary-report-*`.

### FP32 integral-rounding architecture policies

- Split TRUNC/CEIL/RNDNE/FLOOR in both wave sizes into unsuffixed RDNA3 and
  `_rdna4` entry points. RDNA3 suppresses OMOD when IEEE mode is set or guest
  output denormals are preserved; RDNA4 applies scaling independently. This
  follows the ISA OMOD rule and rocjitsu's shared VOP3 rounding helpers, which
  pass the guest denormal mode to `effective_omod` (unlike forced-flush
  SQRT/RCP/RSQ/EXP/LOG). Both architectures share the numerical/SIMD kernels.
- Added explicit guest input flushing to scalar and AVX2 rounding. New gfx1201
  captures demonstrate that CEIL/FLOOR of subnormal inputs depends on this
  setting; the inspected rocjitsu rounding helpers do not apply it. The AVX2
  path flushes using vector bit operations and remains available with all source
  modifiers, OMOD, CLAMP and guest input flushing. No host FP controls change.
- Captured 4,194,304 native `(result, flags)` pairs across four instructions,
  four denormal policies, 32 modifiers and 8192 operands. Loose-result fixtures
  canonicalize only NaN payloads; status words remain in local capture files for
  subsequent exact/reporting implementation. Tests compare both widths and
  rotate all available CPU tiers. RDNA3 expectations use the documented OMOD
  suppression applied to the RDNA4 numerical corpus; these are not native RDNA3
  captures and do not establish bit-exact RDNA3 NaN or exception behavior.
- Added DPP8/DPP16 routing, masking, in-place aliases, invalid-modifier, empty-EXEC
  and optional-state tests across both architectures and all CPU tiers. Existing
  RDNA4 tests and benchmark entries now call `_rdna4` APIs. Exact numerical and
  requested exact state reporting for these four instructions remain unsupported.
- Shared the capture operand generator between the existing exact-unary tests
  and the new rounding tests in `tests/unary_capture_input.h`; arithmetic and
  status expectations still come from independent GPU fixture hashes.
- Regenerated the ISA/architecture/register-name audits: 666 exception-output
  signatures, 374 resolved names matching, zero mismatches, 292 pending review.
  The instruction-name inventory remains 806/809; this slice corrects semantics
  of existing APIs rather than filling one of the three missing HWREG names.
- Clang 21 and GCC 15 Release each pass **1,017/1,017** tests. Logs use
  `/tmp/goc-round-{clang,gcc}-tests.log`; native capture artifacts are
  `../goc-build/coverage-work/{trunc,ceil,rndne,floor}-report-captures/`.
- Targeted Clang ASan/UBSan passes **50/50** rounding/unary/DPP tests; log:
  `/tmp/goc-round-san-tests.log`.
- Eleven alternating CPU-2-pinned before/after samples, minimum 50 ms thread CPU
  time per case, compare against `9b23a0e`. Sixteen loose AVX2 rounding cases
  (default/combined modifiers and null/non-null state) range from -1.96% to
  +1.55%; the largest increase is 0.206 ns. Twelve SQRT/RSQ/RCP control cases
  range from -2.06% to +2.95%, at most 0.610 ns. Both the old dispatch source and
  old SIMD object were compiled for the baseline, so the comparison includes
  the SIMD parameter/branch change. Artifacts use
  `../goc-build/coverage-work/round-arch-*`; these are local per-call timings,
  not a whole-library performance claim.

### Exact FP32 integral rounding and RDNA4 reporting

Historical entry: newly inferred exact reporting was removed in the scope correction
below; FRACT exact results were also removed. See current public API comments.

- Enabled exact TRUNC/CEIL/RNDNE/FLOOR numerical results on both architectures
  and wave sizes. Source modifiers, guest input flushing, architecture-specific
  OMOD policy, CLAMP and DPP use the existing exact unary execution loop. All
  four quiet signaling NaNs, following rocjitsu's shared scalar/SIMD rounding
  helpers; there is no inferred RDNA3 IEEE-dependent NaN-preservation rule here.
  Native gfx1201 capture hashes validate raw payload/sign results and status.
  RDNA3 numerical policy follows the borrowed model and ISA; native RDNA3 status
  evidence is still absent and exact non-null `TRAPSTS` requests remain rejected.
- Extracted `round_integral_bits` into `src/round_integral.h`. It is the existing
  scalar-rounding extension of rocjitsu's integer nearest-even algorithm, now
  shared by FP16/FP32 scalar rounding, exact vector rounding, and the loose
  scalar RNDNE helper. Integer exponent/fraction and tie-handling rules now have
  one implementation. Scalar/vector state reporting and architecture policy
  remain visible at their call sites; no policy/adaptor framework was added.
- RDNA4 reporting accumulates INVALID for signaling NaNs and INPUT_DENORM for
  preserved subnormal inputs. Discarding fractions does not raise INEXACT.
  OMOD may overflow; CLAMP suppresses all new flags. The existing 4,194,304 native
  result/status captures supply raw pair hashes, deduplicated only after checking
  that changing the guest output-denormal policy leaves every pair identical.
- Added per-input raw result/status checks with alternating wave widths, masks,
  strict/non-strict exact semantics, IEEE and output-denormal settings. DPP8/DPP16
  tests cover in-place aliases, architecture OMOD policy, accumulation and host
  rounding/FTZ/DAZ changes. Validation preserves outputs; empty EXEC leaves state
  alone. Existing scalar-rounding tests cover the shared helper's FP16 corpus.
- Clang's Release vectorizer lowers some integer variable shifts through
  floating conversions (`cvttps2dq`), which may raise sticky host INVALID or
  INEXACT. Tests verify bit-exact results, unchanged host rounding/control bits,
  and preservation of existing sticky flags. This follows the standing rule
  allowing arithmetic status flags; no production code explicitly changes the
  host environment.
- Clang 21 and GCC 15 Release pass **1,020/1,020** tests. Targeted Clang
  ASan/UBSan passes **60/60** rounding/scalar-rounding/unary/DPP tests. Following
  the host-status assertion correction, GCC and ASan/UBSan each pass all five
  rounding-architecture tests again. Logs use `/tmp/goc-round-exact-*tests.log`.
- Eleven alternating CPU-2-pinned samples (50 ms minimum thread CPU time per
  case) compare loose AVX2 paths against `40a8d19`. Sixteen rounding cases
  range from -1.72% to +0.09%; twelve SQRT/RSQ/RCP controls range from -2.52%
  to +1.31%, at most 0.253 ns slower. Artifacts use
  `../goc-build/coverage-work/round-exact-*`. These timings cover default and
  combined modifiers with null/non-null state, not exact-path throughput.
- The loose scalar RNDNE caller also changed to use the shared helper. Eleven
  alternating samples for its four modifier/state combinations improve by
  2.50–15.43% in this local run (default null-state 72.648 -> 68.096 ns;
  combined modifiers 109.030 -> 92.367 ns). Scalar artifacts use
  `../goc-build/coverage-work/round-exact-scalar-*`.

### FP32 FREXP_MANT architecture, exactness and denormal handling

Historical entry: newly inferred exact reporting was removed in the scope correction
below; FRACT exact results were also removed. See current public API comments.

- Split `v_frexp_mant_f32` in both wave sizes: unsuffixed RDNA3 and `_rdna4`.
  RDNA3 suppresses OMOD when IEEE mode is enabled or output denormals are
  preserved; RDNA4 scales independently. Both now honor guest input flushing
  and support exact numerical results. Exact RDNA3 `TRAPSTS` reporting remains
  explicitly unsupported without matching status evidence.
- Borrowed rocjitsu `amdgpu::frexp_f32`'s raw-bit significand algorithm from
  `shared/fp_mode.h`, revision `ffc144c564c`, adapting C++20 leading-zero count
  to the existing C++17 helper. Exact evaluation and the loose scalar path share
  that small function. Zeros/infinities survive; signaling NaNs are quieted.
  This follows rocjitsu's common numerical model, not a new native RDNA3 capture.
- Fresh gfx1201 capture: 1,048,576 raw result/status pairs, four denormal policies,
  32 modifiers, 8192 operands. RDNA4 reports INVALID for **infinity as well as
  signaling NaNs**, and INPUT_DENORM for preserved subnormal inputs. CLAMP
  suppresses new status; no INEXACT is generated. Tests retain raw pair hashes
  for exact reporting and canonical-NaN result hashes for loose comparisons.
- Fixed AVX2 significand normalization: convert the 23-bit integer fraction
  exactly rather than multiplying a subnormal floating input by a power of two.
  The old multiplication lost inputs when host DAZ was set; the new path works
  under every tested host rounding/FTZ/DAZ setting. Sign bits remain explicit.
  Guest input flushing stays in the SIMD path; loose mode still skips reporting.
- Tests cover both architectures/widths/semantics, every available CPU tier,
  guest denormal/IEEE policy, DPP8/DPP16, sparse/full masks, in-place aliases,
  accumulated flags, validation and empty EXEC. An old loose test's strict
  signaling-NaN passthrough requirement was corrected to allow quieting, while
  preserving its checks on signs, payloads and non-NaN values.
- Updated the generated audits: 668 exception-output signatures, 378 resolved
  names matching, zero mismatches, 290 pending. Canonical-name inventory remains
  806/809; this is not a claim of completed cross-architecture review.
- Follow-up evidence: LLVM rejects the OMOD assembly spelling on
  `v_frexp_exp_i32_f32`. Its existing ignored-OMOD API contract needs a separate
  encoding/ISA review before borrowing FREXP_MANT's modifier corpus. The failed
  exponent probe produced no fixture; it did not affect the mantissa captures.
- Clang 21 and GCC 15 Release each pass **1,024/1,024** tests. Targeted Clang
  ASan/UBSan passes **60/60** FREXP_MANT, unary, DPP and FP16-unary tests (the
  AVX2 helper is shared with widened FP16 code). Logs use
  `/tmp/goc-frexp-mant-{clang,gcc,san}-tests.log`.
- Refreshed README entries for the ten exact FP32 unary operations and their
  architecture/state contracts. Unary benchmark labels now explicitly identify
  the RDNA4 APIs they call, matching the existing WMMA naming convention.
- Final AVX2 normalization uses the signed integer magnitude for subnormal lanes;
  converted normal encodings are discarded. This avoids redundant fraction/sign
  reconstruction without adding data-dependent fast-path branches. The existing
  boolean input-flush argument is retained; passing full flags did not establish
  a useful performance benefit.
- Eleven alternating CPU-2-pinned samples, at least 50 ms thread CPU time per
  case, compare against `0a77735`. Four normal-input FREXP_MANT cases cost
  4.95–7.41% more; across normal/subnormal cases the largest added time is
  1.251 ns (15.116 -> 16.367 ns). Subnormal inputs with combined modifiers improve
  15.69–27.76%. Twenty-four RNDNE/SQRT/RCP control cases range from -2.71% to
  +1.59%. This is a measured small correctness cost, not a claim of unchanged
  FREXP throughput. SIMD remains eight lanes, including guest input flushing;
  loose reporting remains skipped. No larger special-case dispatch was added.
  Final samples/binaries use `../goc-build/coverage-work/frexp-mant-*` and
  `/tmp/goc-frexp-mant-final-timing.log`.
- Revalidated the final normalization and benchmark labels: Clang/GCC again pass
  1,024 tests each; ASan/UBSan again passes 60 targeted tests. Final logs use
  `/tmp/goc-frexp-mant-final-{clang,gcc,san}-tests.log`.
- The corresponding eight loose scalar cases improve by 1.25–33.59% after
  replacing the library `frexp` call with the borrowed bit model. Default normal
  input/null-state changes 72.458 -> 55.284 ns; default subnormal input/null-state
  changes 86.164 -> 57.219 ns. Eleven alternating samples use the same timing
  protocol; scalar artifacts use `frexp-mant-scalar-*`.

### FREXP_EXP FP32 encoding and status evidence

- Extended `tests/capture_unary_exact.py` with `frexp_exp_i32`. Normal OMOD-zero
  cases use LLVM's `v_frexp_exp_i32_f32_e64` syntax. Nonzero OMOD cases use raw
  VOP3 fields because LLVM rejects those spellings for this integer-result
  instruction. Fixed vector registers are declared clobbered. This records
  empirical handling of those fields, not a claim that LLVM accepts them or
  that every encoding is architecturally specified across generations.
- Native gfx1201 captures cover 8192 inputs, all 32 NEG/ABS/OMOD/CLAMP
  combinations and four denormal policies: **1,048,576 result/status pairs**.
  A second capture using ordinary assembly for OMOD-zero cases is byte-identical
  to the all-raw probe. Capture SHA256:
  `ff03fa74576a6928eb67533d6b6ac49207bfb8b5e463b512090d8599ef2a25ee`.
- Every pair also matches an independent integer model: zero and NaN/infinity
  produce zero; a finite normal produces exponent-field minus 126; a preserved
  subnormal produces magnitude-bit-length minus 149. Guest input flushing makes
  subnormal results zero. NEG, ABS, OMOD and CLAMP do not change the integer
  result. Guest output-denormal policy has no effect on either result or status.
- Without CLAMP, infinity and signaling NaN raise INVALID (bit 0), and preserved
  subnormal input raises INPUT_DENORM (bit 1). Quiet NaNs produce no exception.
  CLAMP suppresses all generated flags, while OMOD does not suppress them.
  No other exception flags occurred in this corpus. This distinction from
  floating-result output modifiers must remain visible in the implementation.
- FNV word hashes of interleaved raw result/status pairs, using the existing
  capture-hash convention, are `0x26533113a2ffed63` (preserve input, no CLAMP),
  `0x92ccb1ca2a1f00a9` (preserve, CLAMP), `0xd89de8c37c933d32` (flush, no CLAMP),
  and `0xff2ec24730195ec6` (flush, CLAMP). Each is over 8192 pairs.
- Probe generation, HIP compilation and execution passed. Artifacts are under
  `../goc-build/coverage-work/frexp_exp_i32-{,checked-}report-captures/`;
  `/tmp/goc-check-frexp-exp.py` checks every pair and the encoding cross-check.
  This is evidence preparation: production FREXP_EXP still needs guest-input
  flushing, exact semantics, architecture/state API review, and regression tests.
  No RDNA3 hardware claim or FP64 conclusion follows from this capture.

### FP32 FREXP_EXP exact semantics and RDNA4 reporting

Historical entry: newly inferred exact reporting was removed in the scope correction
below; FRACT exact results were also removed. See current public API comments.

- Added exact FP32 exponent extraction using rocjitsu's integer `frexp_f32`
  model, including guest input-denormal flushing. The scalar subnormal path now
  uses leading-zero count instead of a per-bit loop. Both architectures share
  this calculation; guest IEEE/output-denormal policy does not alter the integer
  result. FP64 behavior remains unchanged and still needs a separate review.
- Split FP32 APIs into unsuffixed RDNA3 (`trapsts`) and `_rdna4`
  (`excp_flag_user`), before `_wave64`, because the native implicit registers
  differ. Exact RDNA4 reporting implements the verified INVALID/INPUT_DENORM
  rules and CLAMP suppression. RDNA3 exact non-null reporting remains explicitly
  unsupported; no RDNA3 hardware validation is claimed.
- The eight-lane AVX2 path supports guest input flushing and exact results when
  reporting is omitted. Loose mode always skips optional reporting, including
  with non-null state pointers. Exact reporting takes the scalar classifier.
  No new arithmetic policy framework or per-mask fast path was introduced.
- Regression tests preserve the raw hardware result/status hashes and cover all
  32 instruction modifiers, eight guest-policy combinations, both architectures
  and wave widths, all available CPU tiers, DPP routing, in-place aliases,
  inactive exceptional operands, sticky accumulation, empty EXEC and validation.
  Host rounding and all four DAZ/FTZ combinations are tested independently.
- Clang 21 and GCC 15 Release each pass **1,028/1,028** tests; targeted Clang
  ASan/UBSan passes **16/16** FREXP_EXP, instruction-flag and EXEC-mask tests.
  Final logs: `/tmp/goc-frexp-exp-final4-{clang,gcc,san}-tests.log`.
- Refreshed inventory, architecture/state audits and README. The canonical API
  inventory is still 806/809, not semantic completeness. Of 670 native-state
  signatures, 382 have resolved matching names, zero mismatch, and 288 still
  require review. The benchmark now identifies its FP32 RDNA4 API explicitly.
- Final loose AVX2 timing versus `333ddfc`: eight FREXP_EXP cases (normal or
  subnormal inputs, default or combined modifiers, null or non-null state)
  cost **1.90–3.47%**, at most **0.507 ns/call**. Default normal/null-state changes
  14.612 -> 14.955 ns. The 24 unchanged RNDNE/SQRT/RCP controls range from
  -1.16% to +1.78%. Eleven alternating samples are CPU-2-pinned, with at least
  50 ms thread CPU time per case. Data/binaries are under
  `../goc-build/coverage-work/frexp-exp-*`; final summary is
  `/tmp/goc-frexp-exp-final4-timing.log`.
- The final SIMD code uses one unsigned finite-input range check, with its lower
  bound selected by guest input flushing. The common path has the same vector
  operation count as before. One explicit out-of-line DPP helper keeps routing
  buffers off ordinary calls; the exponent calculation remains shared. Earlier
  mask formulations measured worse and were removed. This is a small measured
  cost for added guest-state correctness, not an assertion of identical speed.
- The same eight scalar cases range from -2.83% to +0.13%; default normal/null
  changes 32.764 -> 32.470 ns. The scalar model therefore adds no measured
  material cost in this sample. Samples use `frexp-exp-scalar-*`, with summary
  `/tmp/goc-frexp-exp-scalar-timing.log`.

### FRACT FP32 hardware/reference review

- Added `fract` to `tests/capture_unary_exact.py`. The ordinary and `--exactness`
  corpora each capture 8192 inputs across four denormal policies and all 32
  modifiers on gfx1201: **2,097,152 raw result/EXCP_FLAG_USER pairs** in total.
  HIP compilation/execution and per-pair reference comparison passed. Existing
  probe outputs are unchanged. No production FRACT behavior changes in this slice.
- Both ISA manuals describe `x - floor(x)`, guest rounding and saturation to
  `0x3f7fffff`. The inspected rocjitsu generated VOP1/VOP3 executors use host
  subtraction and floor, with architecture-specific OMOD handling in VOP3;
  they do not supply a host-independent exact model for this instruction.
  In particular, direct host subtraction can produce one for negative tiny
  inputs, so it cannot be imported unchanged as the exact instruction model.
- An independent integer reference decodes the significand and power of two,
  takes its exact fractional remainder (complementing it for negative inputs),
  rounds the resulting dyadic rational to FP32 nearest-even, then limits it to
  the largest FP32 value below one. It matches every captured result bit,
  including signed/payload-preserving NaN quieting and canonical `0xffc00000`
  for either infinity. Integer inputs and both signed zeros produce positive zero.
- Input flushing happens before fractional extraction. Otherwise subnormal
  inputs generate INPUT_DENORM. Output flushing zeroes positive subnormal
  results and generates UNDERFLOW|INEXACT. Negative tiny inputs produce
  `0x3f7fffff`, with INEXACT when the exact fractional remainder is not
  representable; this is independent of the explicit CLAMP modifier.
- Infinity and signaling NaN generate INVALID; quiet NaNs do not. Nonzero OMOD
  flushes subnormal intermediates/results and suppresses UNDERFLOW|INEXACT,
  retaining applicable INVALID/INPUT_DENORM. Explicit CLAMP suppresses all new
  flags and maps NaNs to zero. These rules match both complete captured corpora;
  they are not inferred solely from the mathematical fractional-part formula.
- Capture directories: `../goc-build/coverage-work/fract-report-captures/` and
  `../goc-build/coverage-work/fract-exactness-report-captures/`. SHA256 hashes:
  `cd7826945a8231b4aad48f8bccd1685bc36448d774b1242dca81e8646fb4868e` and
  `f59c4300e7a6eded7a0e4e7952a657b3887bca8cf372d42bb17e6191cee2e979`.
  `/tmp/goc-check-fract.py CAPTURE_DIRECTORY` verifies raw results/status against
  the dyadic reference; capture scripts are `/tmp/goc-fract{,-exactness}-capture.py`.
- Pending: integrate the integer model and regression fixtures, preserve loose
  SIMD throughput while adding guest denormal handling, split native-state APIs,
  and test both wave sizes/DPP/aliases. These probes use guest nearest-even only;
  they do not validate other guest rounding modes or native RDNA3 reporting.

### FP32 FRACT exact semantics and architecture/state APIs

Historical entry: newly inferred exact reporting was removed in the scope correction
below; FRACT exact results were also removed. See current public API comments.

- Implemented exact `x - floor(x)` by composing the existing integer FLOOR and
  FMA models, then applying FRACT's saturation below one. This reuses verified
  nearest-even rounding and exception classification without adding another
  floating-point arithmetic engine or modifying host FP controls.
- Unsuffixed FP32 FRACT APIs now use RDNA3's conditional OMOD policy and native
  `trapsts`; `_rdna4` comes before `_wave64` and uses `excp_flag_user`. Both have
  exact numerical results. Exact RDNA4 reporting implements the captured
  INVALID/INPUT_DENORM/UNDERFLOW/INEXACT rules, OMOD suppression and CLAMP
  suppression. Exact non-null RDNA3 reporting remains unsupported; hardware
  capture provenance is gfx1201, not native RDNA3 verification.
- Scalar and eight-lane loose AVX2 paths honor guest input/output denormal
  flushing while skipping optional reporting. The backend receives existing FP
  flags in its existing argument slot because FRACT needs both denormal bits;
  no additional argument or guest-state adapter was added. Other unary paths
  retain their prior flushing policies. All eleven FP32 unary operations now
  support exact results, simplifying the former exact-support dispatch list.
- New tests check both hardware corpora's result and raw result/status hashes
  (2,097,152 captured pairs), all modifiers, guest policies, both architectures
  and wave widths, CPU tiers, DPP routing, in-place aliases, inactive exceptional
  operands, sticky accumulation, empty EXEC and validation. Exact paths are
  tested under four host rounding modes and four DAZ/FTZ combinations; loose
  arithmetic retains its documented host nearest-even/denormals-enabled contract.
- Clang 21 and GCC 15 Release each pass **1,032/1,032** tests. Clang ASan/UBSan
  passes **64/64** targeted tests: 29 FRACT/unary/DPP/FP16-unary/FREXP_MANT tests
  and 35 existing exact unary tests. Logs use `/tmp/goc-fract-{clang,gcc}-tests.log`,
  `/tmp/goc-fract-san-tests.log` and `/tmp/goc-fract-san-unary-exact-tests.log`.
- Refreshed inventory and architecture/state audits: 672 native-state signatures,
  386 resolved names matching, zero mismatches, 286 pending. Canonical API
  inventory remains 806/809. README and public contracts describe FRACT's new
  support; common FP-flag comments refer to per-instruction contracts rather
  than stale incomplete lists of supported instructions.
- Final scalar/AVX2 loose paths avoid a redundant output-flush check when OMOD
  already mandates flushing. After that change, Clang/GCC again pass 1,032 tests
  each; 56 affected FP32 sanitizer tests pass. The earlier eight FP16-unary
  sanitizer tests also passed, and their implementation was not changed by this
  final adjustment. Final logs use `/tmp/goc-fract-final-{clang,gcc,san}-tests.log`.
- Final loose AVX2 timing versus `598530a`: eight FRACT cases (normal/subnormal,
  default/combined modifiers, null/non-null state) cost **2.48–4.53%**, at most
  **0.614 ns/call**. Default normal/null-state changes 13.653 -> 14.267 ns;
  combined modifiers change 16.048 -> 16.501 ns. The other eight AVX2 unary
  operations span -1.87% to +4.56% across 64 control cases. These are measured
  small costs/variation, not a claim of identical throughput. Eleven alternating
  CPU-2-pinned samples use at least 50 ms thread CPU time per case. Binaries and
  samples use `../goc-build/coverage-work/fract-*`; the final summary is
  `/tmp/goc-fract-final-timing.log`. No sparse-mask specialization was added.

### HWREG field-access hardware evidence

- Added `tests/capture_hwreg.py` and `tests/check_hwreg_capture.py`. The gfx1201
  capture verifies **20,480** `(initialized, updated, extracted)` register
  triples for user-mode `MODE` and `EXCP_FLAG_USER`, all encoded offsets and
  widths, zero/all-one seeds, four source patterns, and SGPR/literal SETREG.
- Confirmed writable masks `0x098000ff` and `0xc000007f`, clipping at bit 31,
  preservation of unselected bits, exception clearing rather than sticky OR,
  and literal/SGPR equivalence. `HWREG_AUDIT.md` records provenance, the raw
  capture hash, reproduction command, and the limits of this evidence.
- The checker passes the real hardware capture and rejects truncated,
  oversized, and deliberately corrupted triples. Regenerating the HIP probe
  produces the same source that was compiled and run for the capture.
- No production APIs changed in this evidence-only checkpoint. Inventory
  remains **806/809**; privileged accesses, other register IDs, snapshot read
  effects, RDNA3 masks, and the public HWREG APIs remain outstanding.

### RDNA4 SETREG native-register APIs

- Added `goc_s_setreg_b32_rdna4` and `goc_s_setreg_imm32_b32_rdna4`. The literal
  form delegates to the same concrete decoder. Explicit operands remain in
  assembly order, guest `STATUS` is passed by value, and six native register
  outputs are separately nullable. `SCRATCH_BASE` uses one `uint64_t *` rather
  than two half-register pointers. These scalar operations have no EXEC input.
- MODE and EXCP_FLAG_USER updates match the gfx1201 capture. Privileged
  STATE_PRIV, EXCP_FLAG_PRIV and TRAP_CTRL masks follow the manual's field
  definitions; they have reference-test coverage, not privileged GPU captures.
  Read-only and privilege-denied writes are no-ops. Unknown/reserved IDs return
  an API validation error. Exception writes perform replacement, allowing clears.
- Exact privileged scratch reporting remains explicitly unsupported when either
  `scratch_base` or `state_priv` is requested. Loose mode skips it and null opts
  out. The manual's alignment requirement does not prove a low-bit hardware
  write mask or rule out a SCRATCH_EN side effect, so no such rule is guessed.
- Added hardware hashes for all 20,480 captured write cases, a bit-by-bit reference
  covering all 64 IDs × 32 offsets × 32 widths and both privilege states for
  both spellings, and null/error/host-environment tests. Existing arithmetic and
  SIMD implementations are unchanged; no existing fast path is rerouted.
- Canonical API presence is now **808/809**, but the two new names have only
  RDNA4 variants. RDNA3 SETREG, both GETREG APIs, snapshot side effects and exact
  scratch writes remain outstanding. Refreshed audits show 674 exception-output
  signatures, 388 resolved names matching, zero mismatches and 286 still pending.
- Clang 21 and GCC 15 Release each pass **1,037/1,037** tests; targeted Clang
  ASan/UBSan passes **19/19** HWREG, guest-mode and instruction-flag tests.
  Final logs: `/tmp/goc-hwreg-final-{clang,gcc,san}-tests.log`. The raw capture
  checker also continues to pass. No performance claim is inferred from these
  correctness tests.

### RDNA3/RDNA4 GETREG read-value APIs

- Added `goc_s_getreg_b32` and `_rdna4`, with separate architecture-ID switches
  and shared integer field extraction. The required scalar destination precedes
  the encoded HWREG selector; implicit inputs follow by value, as in PLAN.md.
  There is no opaque state object or EXEC argument. Native register values come
  from the emulator, including guest clocks, PC, allocation and identity.
- The input boundary is explicit: these are **wave-visible read values**.
  The emulator supplies zero snapshot data for non-sampled waves and zero
  scratch for waves without allocation. The APIs do not synthesize a physical
  snapshot bank, ownership routing or guest clock. The RDNA3 clock selector
  reads its low half; RDNA4 has separate low/high selectors, allowing callers
  to supply the guest counter value observed at each instruction.
- RDNA4 snapshot-PC-high reads expose five separate optional native snapshot
  outputs. Requested non-loose updates remain unsupported because reset/unlock
  behavior is unresolved. Null opts out; loose mode skips updates. Required
  read values are independent of these pointers. No lock register or post-read
  values are invented. No RDNA4 snapshot read-update rule is assumed for RDNA3.
- Tests cover both maps, every ID/offset/width, clipped/width-32 fields, both
  semantics, aliases, unknown/write-only selectors, errors, counter-half
  rollover inputs, all snapshot-output pointer combinations and host FP state.
  RDNA4 read hashes match the 20,480 gfx1201 field values from the HWREG capture.
  Native RDNA3 reads and snapshot lifecycle behavior are not hardware-validated.
- Clang 21 and GCC 15 Release each pass **1,041/1,041** tests; targeted Clang
  ASan/UBSan passes **23/23** HWREG/guest-mode/instruction-flag tests. Logs are
  `/tmp/goc-getreg-{clang,gcc,san}-tests.log`. Existing arithmetic/SIMD paths
  were not modified.
- Canonical API presence is **809/809**, not completed ISA semantics. RDNA3
  SETREG variants, exact privileged scratch writes and snapshot state updates
  remain open, as do the previously recorded arithmetic/architecture gaps.
  The inventory and architecture queue are regenerated; native exception-output
  naming counts remain 674 signatures, 388 matching, zero mismatches, 286 pending.

### RDNA3 SETREG register fields and architecture split

- Added unsuffixed `goc_s_setreg_b32` and `goc_s_setreg_imm32_b32` for RDNA3.
  Each literal form delegates to its architecture's concrete write decoder;
  architecture-specific masks and permissions remain visible in explicit
  switches. The shared instruction-flag test accepts either native-output list.
- MODE uses mask `0x08afffff`: section 3.4.9 explicitly marks exception-enable
  index 8 (MODE bit 20) reserved, resolving the earlier ambiguity in the broader
  EXCP_EN field. STATUS uses Table 5's privileged-write mask `0x2a34211f`;
  PRIV itself is not writable. TRAPSTS uses named-field mask `0x0017fdff` and
  replacement semantics, including clearing selected exception flags. These
  rules have manual/reference-test evidence, not native RDNA3 captures.
- STATUS privilege is supplied by value even if its native output is null.
  Five optional outputs cover MODE, STATUS, TRAPSTS, FLUSH_IB and 64-bit
  FLAT_SCRATCH. FLUSH_IB reporting and privileged scratch/possible SCRATCH_EN
  updates remain unsupported for non-loose non-null requests. Loose mode skips
  these unavailable effects; null opts out. No flush command or persistent
  write-only register value is invented. TBA follows the manual's read-only
  rule rather than rocjitsu's conflicting privileged descriptor.
- Tests cover both spellings, every selector/offset/width, both semantics and
  privilege states, field masks reconstructed independently from manual ranges,
  exception clearing, reserved-bit and PRIV preservation, output opt-outs,
  unchanged outputs on errors, validation ordering and host FP controls.
- Clang 21 and GCC 15 Release each pass **1,044/1,044** tests; targeted Clang
  ASan/UBSan passes **26/26** tests. Logs use
  `/tmp/goc-setreg3-{clang,gcc,san}-tests.log`. Arithmetic and SIMD paths remain
  unchanged.
- All three HWREG mnemonics now have separate RDNA3/RDNA4 APIs, but unresolved
  native-state effects still prevent a completeness claim. Canonical API
  presence remains **809/809**. The refreshed naming audit has 676 signatures,
  390 matching resolved names, zero mismatches, and 286 pending reviews.

## Scope correction: remove newly reverse-engineered exact behavior

The earlier entries are a chronological work log, not current support promises.
The hardware-derived unary and FP32 FMA reporting claims above are superseded
by this cleanup. The mandate is loose implementation plus ports of existing
rocjitsu exact models, with no new hardware reverse-engineering.

- Deleted custom FP16/FP32 FRACT exact paths; loose scalar/SIMD paths remain.
- Deleted new FP16/FP32 unary exception classifiers, including FREXP reporting,
  and FP32 ordinary/literal FMA classifiers. Their VOPD wrappers inherit rejection
  of non-null exact reporting. Removed newly inferred MIX_F32 no-op reporting.
- Kept rocjitsu-derived numerical rounding, FREXP, FMA, and SFU models. The FP16
  RCP/RSQ/SQRT adapter follows existing `transcendental::map_f16`; no new fitted
  approximation or correction tables are retained.
- Deleted the abandoned FP16 capture generator, capture-analysis scripts and
  combined result/status fixture. Tests retain numerical, architecture, DPP,
  aliasing, host-control, and loose-state-opt-out coverage; generated API tests
  check unsupported exact reporting for every affected public signature.
- Earlier explicitly requested exception implementations (FP16 FMA, comparisons,
  scalar rounding, division helpers, trigonometry) remain unchanged.

Validation: Clang 21 and GCC 15 Release each pass **1,027/1,027** tests,
including benchmark smoke tests. Clang ASan/UBSan passes **50/50** targeted
unary/FRACT/FMA/FREXP/rounding/global-state tests. `clang-format` and
`git diff --check` pass. Logs are in `../goc-build/coverage-work/`
(`remove-models-final-*`), `../goc-build-gcc/` and
`../goc-build-sanitize/` (`remove-models-*`). The autonomous goal remains paused.

## FP32/integer conversion architecture review

- Reviewed all six `conversion32.cpp` operations against rocjitsu revision
  `ffc144c564c`, `generated/shared/execute_shared.h` (`execute_v_cvt_f32_i32`,
  `execute_v_cvt_f32_u32`, `execute_v_cvt_i32_f32`, `execute_v_cvt_u32_f32`,
  `execute_v_cvt_nearest_i32_f32`, `execute_v_cvt_floor_i32_f32`) and
  `shared/fp_mode.h::effective_omod`. No hardware characterization was performed.
- Split integer-to-FP32 APIs: unsuffixed RDNA3 disables OMOD when IEEE is set or
  output denormals are preserved; `_rdna4` retains unconditional OMOD. IEEE and
  input/output-denormal guest flags are now accepted. Host nearest-even arithmetic
  remains required; guest rounding modes are not newly implemented.
- Truncating float-to-integer results share the existing saturation implementation,
  with RDNA3 `trapsts` and RDNA4 `excp_flag_user` APIs at both wave widths. Exact
  reporting remains unsupported. NEAREST/FLOOR retain shared names and no optional
  state output. Integer results ignore the accepted guest denormal/IEEE flags;
  the upstream bodies do not flush inputs or apply OMOD to integer results.
- Existing RDNA4 scalar/SIMD kernels and captured regression tests are preserved.
  Rocjitsu's generated NEAREST/FLOOR NaN-to-zero behavior differs from the existing
  RDNA4 signed-limit fixture. This change does not port that discrepancy or claim
  exact support: all six instructions remain loose-only. No new exact model or
  exception classifier was added.
- New tests cover all low modifiers and eight guest policies, Wave32/Wave64,
  scalar/v3/v4/Zen4 selections, DPP8/DPP16, sparse masks, unaligned buffers,
  in-place aliases, destination guards, error atomicity, optional-state opt-out,
  and unchanged host FP controls. Literal `3` with `mul:2` distinguishes RDNA3's
  result `3` from RDNA4's `6` when output denormals are preserved.

Validation: all 1,030 CTests pass with Clang 21 Release and GCC 15 Release;
20 targeted ASan/UBSan tests pass. Logs are `conversion32-tests4.log` under
`../goc-build/coverage-work/`, and `conversion32-tests2.log` under the GCC and
sanitizer build directories.

Performance: seven alternating before/after samples per case, pinned to CPU 2,
using thread CPU time and at least 20 ms per sample. Across six operations and
plain/modified forms, median time changes are -0.08% scalar, +0.02% AVX2 and
+0.26% AVX-512. The worst case, modified unsigned-to-FP32 AVX-512, changes from
3.654 ns to 3.861 ns (+5.7%). Both SIMD object files are byte-identical before
and after; disassembly of the corresponding dispatcher differs only in the
accepted-flags immediate. The small absolute change may reflect binary layout,
but it is measured and not dismissed as noise. Raw samples and medians are in
`../goc-build/coverage-work/conversion32-performance.json`. No arithmetic kernel
or per-lane reporting work was added.

Next architecture review: FP16 conversions. Rocjitsu's `cvt_f32_f16` helper
applies guest input flushing and architecture-dependent signaling-NaN quieting;
its `effective_f16_omod` gates RDNA3 scaling. The existing GoC conversion family
still needs that review and implementation; this slice does not close it.

## FP16 conversion architecture policies

- Reviewed all six `conversion16.cpp` mnemonics against rocjitsu revision
  `ffc144c564c`: generated conversion execution helpers, `fp_mode::effective_omod`,
  `fp_mode::cvt_f32_f16`, and `simd_glue.h::cvt_f32_f16_mode_simd`. No hardware
  characterization or new bit-exact model was developed.
- Added unsuffixed RDNA3 APIs with native `trapsts` outputs; preserved established
  RDNA4 behavior and fixtures under `_rdna4`, before `_wave64`. All six still
  provide loose semantics; strict exact and non-null exact reporting remain
  explicitly unsupported. Loose state outputs remain untouched.
- Floating-result conversions gate RDNA3 OMOD on output flushing and IEEE=0.
  RDNA4 OMOD remains independent. Integer-result conversions share the same
  saturation/truncation kernels and ignore these guest controls numerically.
- Widening FP16 to FP32 honors guest input flushing and preserves signaling NaNs
  on RDNA3 IEEE=0 when no arithmetic modifier is applied. RDNA4 and RDNA3 IEEE=1
  quiet them. Scalar, AVX2 and AVX-512 paths implement this policy. Native x86
  half widening quiets NaNs, so SIMD restores raw NaN encodings for RDNA3 IEEE=0
  before applying output modifiers. Host controls are never changed.
- Existing FP16-result loose rounding and CLAMP behavior are retained. This does
  not add arbitrary guest rounding modes, DX10_CLAMP control, or exact exception
  reporting. These limits must not be confused with a hardware-exact conversion
  claim. In particular, the upstream output helpers are broader than this loose
  implementation and are not copied wholesale.
- New tests cover all six operations, both architectures and wave widths, all
  low modifiers, 16 guest flag combinations, DPP, unaligned arrays, in-place
  aliases, sparse EXEC masks, destination guards, unchanged optional state,
  error atomicity, and unchanged host FP controls. Widening additionally checks
  every half encoding against an integer-only reference at every CPU level,
  including signed zeros, subnormals and NaN payload/signaling bits.

Validation: all 1,034 CTests pass with Clang 21 Release and GCC 15 Release;
23 targeted ASan/UBSan tests pass. Final logs are `conversion16-tests4.log`
under `../goc-build/coverage-work/`, and `conversion16-tests3.log` under the
GCC and sanitizer build directories.

Performance: seven alternating before/after runs, pinned to CPU 2, using thread
CPU time and at least 20 ms per case. Median time changes across six operations
and plain/modified forms are +0.18% scalar, +1.08% AVX2, and +0.17% AVX-512.
The worst case is plain unsigned-half-to-half AVX-512: 5.529 to 5.827 ns (+5.4%).
Raw samples are in `../goc-build/coverage-work/conversion16-performance.json`.

The initial runtime-policy loop slowed default widening by about 20% scalar
and 6–8% SIMD. The final code uses a separate guest-widening specialization:
one shared source algorithm, with policy checks compiled out of the ordinary
RDNA4 path. Existing SIMD entry signatures are preserved. All six AVX2 default
kernels retain identical instruction bytes; AVX-512 kernels retain their sizes,
with some compiler scheduling/register-allocation differences. This adds one
internal specialization and one named guest-widening entry per SIMD level,
rather than duplicating source algorithms or burdening unrelated conversions.
The remaining small timing changes are recorded rather than hidden.

The broader goal is still active. FP64 conversions and the remaining unresolved
cross-architecture families still require review; this is not a whole-ISA
completion claim.

## FP64 conversion architecture review

- Reviewed the six FP64 conversion mnemonics against rocjitsu revision
  `ffc144c564c`, `generated/shared/execute_shared.h` and its conversion SIMD
  helpers, plus RDNA3 ISA section 7.2.3.1's output-modifier rule. That rule
  explicitly includes double-precision results. The upstream conversion bodies
  share host casts or saturating integer conversion; they do not supply a new
  empirical exact exception model to port.
- Floating-result conversions now use unsuffixed RDNA3 APIs with OMOD disabled
  when output denormals are preserved or IEEE is set. `_rdna4` APIs preserve
  unconditional scaling. Both use the existing shared numeric/SIMD kernels.
- The four reporting conversions expose native RDNA3 `trapsts` and RDNA4
  `excp_flag_user` outputs at both wave widths. Exact reporting is unsupported;
  loose calls leave optional state untouched. Integer-to-double conversions
  still have no optional state output.
- Guest IEEE/input/output-denormal flags are accepted. Beyond RDNA3 OMOD gating,
  the existing loose host-cast behavior is retained, following the upstream
  conversion bodies. Guest rounding control, exact payloads, and exact state
  reporting are not newly promised. Host nearest-even and enabled denormals
  remain required. No host FP-control writes or hardware characterization were
  introduced. DPP remains unsupported for this family.
- New tests cover all six operations, all modifiers and eight guest policies,
  Wave32/Wave64, all CPU selections, source/destination aliases, overlapping
  destination halves, unaligned buffers, sparse masks and destination guards.
  Literal finite witnesses distinguish RDNA3 and RDNA4 scaling. NaN tests
  separately verify loose quiet-NaN/integer-zero behavior across policies.
  Error tests check zero/nonzero EXEC, unsupported exact reporting, atomic
  outputs, invalid encodings, and preservation of host FP controls.

Validation: all 1,038 CTests pass with Clang 21 Release and GCC 15 Release;
20 targeted ASan/UBSan tests pass. Final logs are `conversion64-tests2.log`
under `../goc-build/coverage-work/`, `../goc-build-gcc/`, and
`../goc-build-sanitize/`.

Performance: seven alternating before/after samples, pinned to CPU 2, using
thread CPU time and at least 20 ms per case. Across all six conversions and
plain/modified forms, median time changes are +0.06% scalar, +0.16% AVX2, and
-0.03% AVX-512; the worst observed change is +1.01%. Both SIMD object files are
byte-identical to the baseline. Raw samples are in
`../goc-build/coverage-work/conversion64-performance.json`. No per-lane policy
checks or reporting work were added.

Next: byte/nibble conversion architecture policies, followed by the remaining
packed/normalized conversion review. The overall ISA completion audit remains
open.

## Byte/nibble conversion architecture policies

- Reviewed `v_cvt_f32_ubyte0..3` and `v_cvt_off_f32_i4` against rocjitsu
  revision `ffc144c564c`, `generated/shared/execute_shared.h` and
  `fp_mode::effective_omod`. All five generated VOP3 helpers explicitly apply
  that architecture policy after extracting the integer and converting it.
- Unsuffixed APIs now select RDNA3, where OMOD requires output flushing and
  IEEE=0. `_rdna4` variants retain unconditional OMOD, including Wave64 forms.
  Guest IEEE/input/output-denormal flags are accepted; there are no subnormal
  inputs or results to flush. No implicit global-state outputs are added.
- Both loose and strict exact semantics retain their existing scalar/AVX2/AVX-512
  implementations. Extraction, conversion and power-of-two scaling are exact
  for these bounded inputs under every host rounding mode. No new empirical
  model or hardware characterization was used.
- Tests cover every byte/nibble value, both semantics and wave sizes, eight guest
  policies, every OMOD/CLAMP combination, every CPU level, DPP routing, sparse
  masks, aliases, unaligned buffers and destination guards. Host tests exercise
  all four rounding modes and both FTZ/DAZ settings with pre-existing exception
  flags, and verify no changes to host rounding, controls or sticky flags.
- Existing RDNA4 fixture/benchmark callers now explicitly select `_rdna4`.
  Numeric kernels remain shared; the only new runtime policy is at the API
  boundary before the existing execution paths.

Validation: all 1,041 CTests pass with Clang 21 Release and GCC 15 Release;
28 targeted ASan/UBSan tests pass. Logs are `byte-conversion-tests.log` under
`../goc-build/coverage-work/`, `../goc-build-gcc/`, and
`../goc-build-sanitize/`.

Performance: seven alternating before/after samples, pinned to CPU 2, with
thread CPU time and at least 20 ms per case. Across the five conversions and
plain/modified forms, median time changes are +0.12% scalar, -0.01% AVX2, and
-0.69% AVX-512; all measured slowdowns are below 1%. Both SIMD object files
are byte-identical to the baseline. Raw samples are in
`../goc-build/coverage-work/byte-conversion-performance.json`.

Packed-conversion follow-up: rocjitsu's shared `v_cvt_pk_u8_f32` body truncates,
whereas GoC's pre-existing RDNA4 fixture expects nearest-even. This is a model
mismatch, not evidence that RDNA3 hardware differs from RDNA4. Do not invent an
architecture-specific rounding rule or reverse-engineer a new exact model from
that discrepancy. The three two-source packed conversions and normalized
conversions also still need their native-state/API review; they are not closed
by this byte/nibble slice.

## Packed and normalized conversion state/API review

Reviewed against rocjitsu revision `ffc144c564c`: shared packed-RTZ and normalized
execution helpers; architecture-specific `VCvtPkI16F32Vop3`, `VCvtPkU16F32Vop3`,
and `VCvtPkNorm{I,U}16F16Vop3` bodies; `util::f32_to_f16_rtz`; and RDNA3 ISA
packed-conversion descriptions. The architecture-specific bodies matter here:
the shared-helper scan does not find all of these mnemonics.

| Forms | Numeric implementation | API decision |
|---|---|---|
| `v_cvt_pk_rtz_f16_f32` | Existing shared round-toward-zero half conversion; packed OMOD ignored | RDNA3 `trapsts`, `_rdna4` `excp_flag_user` |
| `v_cvt_pk_i16_f32`, `v_cvt_pk_u16_f32` | Shared saturation and truncation, NaN-to-zero | RDNA3 `trapsts`, `_rdna4` `excp_flag_user` |
| `v_cvt_pk_norm_i16_f16`, `v_cvt_pk_norm_u16_f16` | Shared double-product normalized conversion, nearest-even | RDNA3 `trapsts`, `_rdna4` `excp_flag_user` |
| `v_cvt_pk_norm_i16_f32`, `v_cvt_pk_norm_u16_f32` | Same normalized algorithm | Retain shared unsuffixed API, no optional state output |
| `v_cvt_norm_i16_f16`, `v_cvt_norm_u16_f16` | Same normalized algorithm, selected destination half | Retain shared unsuffixed API, no optional state output |

All decisions also apply to Wave64. The numeric scalar and SIMD kernels remain
shared. Guest IEEE/input/output-denormal flags are now accepted without numeric
effect, following the reviewed conversion bodies. Unlike ordinary floating
results, these forms do not acquire architecture-dependent OMOD scaling.
The forms' existing supported/ignored modifier rules remain unchanged.

All nine remain loose-only; exact optional reporting remains explicitly
unsupported regardless of STRICT, and loose calls leave state untouched.
Rocjitsu's RTZ utility preserves raw NaN payload/signaling bits while GoC's
existing RDNA4 fixture and loose SIMD contract quiet them. No new hardware
rule is inferred from that discrepancy and no new exact claim is made.
The existing safe NaN-to-zero integer behavior is retained rather than copying
an upstream floating-to-integer cast that may be undefined for NaN. No hardware
characterization or host FP-control changes were introduced.

New tests compare both architectures and widths with independent integer
references for all modifiers, eight guest policies, all CPU selections, ordinary
and DPP8/DPP16 forms, sparse masks, source/destination aliases, unaligned storage,
and destination guards. Separate tests cover exact rejection, optional-state
contracts, zero/nonzero EXEC error atomicity, and unchanged host FP controls.
The original RDNA4 captured/reference suites now call the explicit `_rdna4`
APIs where a native-state split was needed.

Validation: all 1,043 CTests pass with Clang 21 Release and GCC 15 Release;
33 targeted ASan/UBSan tests pass. Logs are `packed-architecture-tests.log`
under `../goc-build/coverage-work/`, `../goc-build-gcc/`, and
`../goc-build-sanitize/`.

Performance: seven alternating before/after samples per case, pinned to CPU 2,
using thread CPU time and at least 20 ms. Across nine conversions and their
plain/modified forms, median time changes are -0.14% scalar, -0.08% AVX2,
and -0.03% AVX-512. The worst measured slowdown is +1.09%. All four SIMD object
files are byte-identical to the baseline. Raw samples are in
`../goc-build/coverage-work/packed-architecture-performance.json`. No per-lane
policy branches or reporting work were added.

The separate `v_cvt_pk_u8_f32` rounding discrepancy remains open; this slice does
not resolve it or claim completion of the wider architecture audit.

## Integer conversion architecture review

Reviewed `v_cvt_i32_i16`, `v_cvt_u32_u16`, `v_cvt_pk_i16_i32`, and
`v_cvt_pk_u16_u32` against rocjitsu revision `ffc144c564c`: both architectures'
`generated/rdna*/vop3_exec.cpp` extension handlers and delegation to the shared
packed-conversion helpers in `generated/shared/execute_shared.h`. Both targets
sign-extend or zero-extend the selected half for widening, and saturate each
32-bit input before packing source A into the low half and B into the high half.
There is no floating-point policy or implicit exception-register update here.
The shared unsuffixed APIs, including Wave64, remain appropriate. No new
numerical model or hardware characterization was needed.

The existing implementation supports loose and exact semantics with the same
integer algorithm. `integer_conversion_test.cpp` covers every 16-bit input and
saturation boundaries, both source-half selections, scalar/SIMD CPU selections,
Wave32/Wave64, DPP8/DPP16, aliases, sparse EXEC, unaligned buffers, guard words,
validation failures and preservation of host FP controls. This review covers the
GoC operand/modifier contract; it does not make GoC an instruction decoder or
claim support for every ISA encoding. Guest floating-point flags remain outside
this integer API's accepted flag set, except the existing ignored FP16-overflow
flag.

Validation: all nine integer-conversion CTests pass in the existing Clang 21
Release, GCC 15 Release and Clang ASan/UBSan builds. No production source or
binary changed in this review. The explicit review ledger now records these
four instructions and the 32 conversions reviewed in the preceding sections.
Its generator tests reject unknown or duplicate mnemonics and missing evidence
sections/test files. This ledger is a scoped contract checklist, not a replacement
for the documented exact-semantics and encoding limitations.

## LDEXP FP32/FP64 architecture review

Reviewed `v_ldexp_f32` and `v_ldexp_f64` against rocjitsu revision `ffc144c564c`:
`generated/shared/execute_shared.h::execute_v_ldexp_f{32,64}_vop3`,
`shared/fp_mode.h::ldexp` and `effective_omod`, and
`shared/division.h::div_round`. The input-denormal rule is applied before scaling;
output flushing applies to the rounded result. RDNA3 ISA section 7.2.3.1 and
rocjitsu agree that IEEE mode or output-denormal preservation disables RDNA3
OMOD. RDNA4 applies OMOD independently of those settings.

- Unsuffixed APIs now select RDNA3; `_rdna4` APIs preserve the established RDNA4
  behavior. This applies to Wave32 and Wave64. Optional state parameters are
  `trapsts` and `excp_flag_user`, respectively.
- Both architectures accept guest IEEE/input/output-denormal flags. Input and
  output flushing preserve the zero sign, independently of host FP controls.
  Existing OMOD tininess handling, ABS/NEG, CLAMP, source aliases and masked
  writes remain supported. FP32 retains DPP8/DPP16 on A; FP64 DPP remains
  explicitly rejected. The signed exponent in B is not permuted.
- Both forms remain loose-only. Strict exact results and non-null exact optional
  state reporting remain unsupported. Guest rounding modes and faithful NaN
  payload/signaling behavior are not newly claimed. Rocjitsu's integer rounding
  model has not yet been ported here; no new hardware characterization was done.
  Its `ScopedFenv` use is not copied: production never changes host FP controls.
- Scalar, AVX2 and AVX-512 paths share each architecture's effective OMOD policy.
  Guest flushing has SIMD implementations; it does not force a scalar fallback.
  A compile-time denormal-policy choice keeps the default loops free of new
  per-lane flag checks. The guest scalar entry is kept out of line because the
  compiler otherwise merges its loop into the default loop. This adds compiled
  variants while keeping one source algorithm per CPU path.

`ldexp_architecture_test.cpp` covers literal scaling/flushing witnesses, all
32 low modifier combinations, eight guest policies, both architectures and wave
widths, all available CPU selections, overflow/underflow boundaries, signed
zeros, NaN classes, DPP, aliases (including coincident FP64 destinations), sparse
EXEC, unaligned words, guard words, error atomicity and optional-state contracts.
It checks host rounding controls, FTZ/DAZ and preservation of existing sticky
flags. Established RDNA4 reference and captured tests now use `_rdna4` names.

Performance: seven alternating before/after samples, pinned to CPU 2, with thread
CPU time and a 20 ms minimum per sample. Across FP32/FP64 plain and modified
forms, median time changes are +0.29% scalar, -1.50% AVX2 and +1.06% AVX-512.
The largest measured slowdown is plain FP32 AVX-512, 5.279 ns to 5.451 ns
(+3.26%). Raw samples are in `../goc-build/coverage-work/ldexp-performance.json`.
These measurements cover the established default guest-denormal policy, not a
claim that newly supported flushing has no cost. Loose optional reporting adds
no per-lane work.

Validation: all 1,048 CTests pass with Clang 21 Release and GCC 15 Release;
27 targeted ASan/UBSan tests pass. Logs are `ldexp-tests2.log` under
`../goc-build/coverage-work/`, `../goc-build-gcc/`, and
`../goc-build-sanitize/`. All nine audit-tool unit tests pass. The native-state
naming audit now resolves 502 of 732 reporting signatures, with zero mismatches;
230 remain unresolved. This does not establish reporting accuracy or completeness.

## FP16 exponent architecture review

Reviewed `v_ldexp_f16` and `v_frexp_exp_i16_f16` against rocjitsu revision
`ffc144c564c`, including architecture-specific true16 VOP3 bodies in
`generated/rdna{3,4}/vop3_exec.cpp`, the shared execution helpers, and
`shared/fp_mode.h::{ldexp_f16,effective_f16_omod,finish_arithmetic_f16}`.
RDNA3 ISA's VOP1/VOP3 FREXP descriptions specify a signed integer destination.

- Unsuffixed APIs now select RDNA3 with `trapsts`; `_rdna4` APIs use
  `excp_flag_user`, at both wave widths. LDEXP disables RDNA3 OMOD when IEEE
  mode is set or output denormals are preserved. FREXP's integer algorithm
  remains shared; the API split names the native implicit register.
- LDEXP accepts guest IEEE/input/output-denormal flags. Explicit flushing is
  implemented with bit classification, preserving the zero sign. Scalar and
  eight-lane AVX2/F16C paths retain selected source/destination halves, ABS/NEG,
  OMOD, CLAMP, FP16 overflow saturation, DPP8/DPP16 and whole-VGPR aliases.
  Higher CPU selections use the same existing AVX2 kernel. No sparse-EXEC
  specialization or host FP-control writes were introduced.
- FREXP accepts those guest flags without changing its integer result, following
  the upstream exponent extraction's treatment of inputs. Sign modifiers,
  floating output modifiers and the FP16 overflow flag do not change GoC's
  integer result. Both forms remain loose-only; exact results and exact
  non-null optional-state reporting are explicitly unsupported.
- The upstream generated FREXP bodies convert the exponent back to FP16 and
  apply floating result modifiers, contradicting the ISA's integer destination
  contract. GoC retains its existing signed-integer result rather than copying
  that behavior or treating it as hardware evidence. Separately, upstream's
  LDEXP shared scalar body scales its wide intermediate before narrowing;
  GoC retains its existing RDNA4 hardware-tested rounding-before-OMOD behavior.
  This review does not resolve those upstream discrepancies or claim an exact
  model. No hardware characterization was performed.

Guest flushing uses a separate compiled LDEXP specialization so default loops
keep their existing arithmetic. The source algorithm remains shared. The default
FREXP path gains no per-lane policy checks. A shared test reference now supplies
both the established and architecture-policy suites; it is independent of the
production half-conversion implementation.

New tests cover all low modifiers and half selectors, sixteen combinations of
IEEE/denormal/FP16-overflow settings, both architectures and widths, every
available CPU selection, DPP routing, aliases, sparse EXEC, preserved destination
halves, guard words, validation and optional-state contracts. An additional
exhaustive sweep checks every FP16 encoding under all eight IEEE/denormal modes.
Host rounding controls, FTZ/DAZ and existing sticky flags are checked for
preservation. Existing RDNA4 captures and numerical regressions now call the
explicit `_rdna4` APIs.

Performance: seven alternating before/after samples per case, pinned to CPU 2,
using thread CPU time and at least 20 ms per sample. Across plain/modified LDEXP
and FREXP, median time changes are +0.47% scalar, -0.05% for the AVX2 CPU
selection, and -0.38% for the v4 CPU selection (which also dispatches AVX2 here).
The worst measured slowdown is scalar plain LDEXP, 204.812 ns to 214.055 ns
(+4.51%); modified AVX2 LDEXP changes from 24.388 ns to 25.402 ns (+4.16%).
Raw samples are in `../goc-build/coverage-work/half-exponent-performance.json`.
These compare the existing default guest-denormal policy, not the cost of newly
supported flushing.

Validation: all 1,052 CTests pass with Clang 21 Release and GCC 15 Release;
21 targeted ASan/UBSan tests pass. Logs are `half-exponent-tests.log` under
`../goc-build/coverage-work/`, `../goc-build-gcc/`, and
`../goc-build-sanitize/`. All nine audit-tool tests pass. The naming audit now
resolves 510 of 736 reporting signatures, with zero mismatches and 226 still
unresolved; these counts do not assert exception-reporting accuracy.

## FP32/FP64 FREXP exponent architecture review

Reviewed `v_frexp_exp_i32_f32` and `v_frexp_exp_i32_f64` against rocjitsu revision
`ffc144c564c`: `shared/fp_mode.h::frexp_f32`, shared VOP1/VOP3 execution helpers,
and architecture-specific delegation in `generated/rdna{3,4}/vop{1,3}_exec.cpp`.
RDNA3 ISA VOP1/VOP3 FP64 descriptions specify signed integer output.

- FP32 already uses the borrowed integer `frexp_f32` result model in both
  architectures, with guest input flushing, loose/exact results and native
  `trapsts`/`excp_flag_user` APIs. IEEE and output-denormal flags do not change
  the exponent. Both wave widths and FP32 DPP remain covered by the existing
  exact/numerical regression suites. No new numerical or reporting model was
  added during this review.
- FP64 now has unsuffixed RDNA3 and `_rdna4` APIs, including Wave64, with native
  `trapsts` and `excp_flag_user` parameters. Both share the existing exponent
  computation and accept guest IEEE/input/output-denormal flags without numeric
  effect, following the upstream FP64 VOP1 extraction. The scalar and SIMD
  algorithms do not change. FP64 DPP remains explicitly rejected.
- FP64 remains loose-only. Its VOP3 upstream helper converts the integer exponent
  to FP32 and applies floating result modifiers, whereas its VOP1 helper and
  the ISA specify integer output. GoC keeps its existing integer exponent and
  ignored floating modifiers instead of copying that discrepancy. No new
  hardware inference or exact claim follows from this review.
- Exact non-null optional state reporting remains unsupported on both targets
  and both element widths. Loose calls leave optional state untouched. The
  historical FP32 reporting model described earlier in this file was removed
  during the scope correction and is not reinstated.

Two new `frexp_exp_test.cpp` cases cover FP64 architecture APIs, both wave
widths, all eight guest policies, all low modifiers, all available CPU selections,
source/destination aliases, signed subnormals, normals, infinities, signaling/quiet
NaNs, sparse EXEC, unaligned buffers, guards and error atomicity. Tests verify
results and unchanged host rounding, FTZ/DAZ and sticky flags across all host
control combinations. Existing exponent-field/leading-bit sweeps and RDNA4
numerical tests now name the explicit FP64 `_rdna4` APIs.

Performance: seven alternating before/after samples, CPU 2 affinity, thread CPU
time, 20 ms minimum per case. For FP64 normal/subnormal inputs with plain/modified
forms, median time changes are -0.68% scalar, +0.41% for the AVX2 CPU selection,
and +0.36% for v4 (which also uses AVX2 here). The worst measured slowdown is
1.26%, from 20.720 ns to 20.981 ns. The SIMD object is byte-identical before and
after. Raw samples are in `../goc-build/coverage-work/frexp64-performance.json`.

Validation: all 1,054 CTests pass with Clang 21 Release and GCC 15 Release;
17 targeted ASan/UBSan tests pass. Logs are `frexp64-tests.log` under
`../goc-build/coverage-work/`, `../goc-build-gcc/`, and
`../goc-build-sanitize/`. All nine audit-tool tests pass. The native-register
naming audit resolves 514 of 738 reporting signatures, with zero mismatches and
224 still unresolved; numerical/reporting completeness remains a separate gate.

## FP64 unary architecture policies

Reviewed all nine unary operations in `fp64.cpp`: `v_trunc_f64`, `v_ceil_f64`,
`v_rndne_f64`, `v_floor_f64`, `v_fract_f64`, `v_sqrt_f64`, `v_rcp_f64`,
`v_rsq_f64`, and `v_frexp_mant_f64`. Rocjitsu revision `ffc144c564c` delegates
both architectures' VOP3 forms to the corresponding shared execution helpers.
Each uses `fp_mode::effective_omod` with `fp_denorm_mode_f16_f64()` and IEEE mode.
The SFU helpers in `shared/transcendental.h` use host reciprocal/square-root
arithmetic with special-value handling; this is not a newly borrowed empirical
bit-exact FP64 model.

- Unsuffixed APIs now select RDNA3, with `_rdna4` before `_wave64` where needed.
  The trailing optional state pointers name `trapsts` or `excp_flag_user`.
  Both wave sizes share the existing scalar and AVX2 numerical kernels.
- All nine accept guest IEEE/input/output-denormal flags. These flags gate RDNA3
  OMOD: IEEE mode or preserved output denormals disables scaling. RDNA4 scaling
  is independent. The reviewed upstream unary bodies do not otherwise flush
  inputs/results using these guest flags; GoC retains that behavior.
- The RDNA3 output-modifier rule is centralized locally and also used by the
  existing FP64 ADD/MUL/FMA and legacy selection dispatch. No arithmetic loop,
  per-lane reporting work, or sparse-mask optimization was added. FP64 unary
  DPP remains explicitly rejected.
- All nine remain loose-only, with exact non-null optional reporting unsupported.
  Guest rounding modes, NaN payload/signaling fidelity and exact SFU behavior
  are not newly claimed. Existing GoC FRACT capping below one is retained; the
  upstream shared helper merely subtracts FLOOR. This review does not infer
  a hardware difference from that discrepancy or introduce a new exact model.

Three new cases in `fp64_unary_test.cpp` test literal scaling witnesses for every
operation, both targets and wave sizes, all eight guest policies, all low
modifiers, all available CPU selections, normal/subnormal/special inputs,
sparse EXEC, whole-VGPR and cross-word aliases, coincident destinations, unaligned
storage and guards. Separate tests cover error atomicity, optional state and
host rounding/FTZ/DAZ/sticky-flag preservation. Loose SFU results use the existing
independent long-double reference with a floating comparison; coincident
high-word stores are checked against a disjoint invocation because an allowed
ULP difference can cross a word boundary. Existing RDNA4 numerical/output tests
now call the explicit `_rdna4` APIs.

Performance: seven alternating before/after samples per case, pinned to CPU 2,
using thread CPU time and at least 20 ms per sample. Across all nine operations
with plain/modified forms, median time changes are +0.14% scalar, +0.18% for
AVX2 and +0.08% for the v4 CPU selection (also AVX2 for these operations).
The largest measured slowdown is modified scalar RNDNE, 107.216 ns to
108.601 ns (+1.29%). The SIMD object is byte-identical before and after.
Raw samples are in `../goc-build/coverage-work/fp64-unary-performance.json`.

Validation: all 1,057 CTests pass with Clang 21 Release and GCC 15 Release;
25 targeted ASan/UBSan tests pass, including existing FP64 arithmetic/selection
regressions for the centralized modifier rule. Logs are `fp64-unary-tests.log`
under `../goc-build/coverage-work/`, `../goc-build-gcc/`, and
`../goc-build-sanitize/`. All nine audit-tool tests pass. The naming audit
resolves 550 of 756 reporting signatures with zero mismatches and 206 remaining
unresolved; this remains separate from numerical and reporting fidelity.

## Floating relational comparison architecture review

Reviewed the 84 nonconstant floating CMP/CMPX mnemonics (14 predicates for each
of FP16/FP32/FP64 and CMP/CMPX), in both wave sizes. At rocjitsu revision
`ffc144c564c`, all 336 corresponding RDNA3/RDNA4 VOPC/VOP3 execution bodies
use `shared/comparison.h`, directly or through generated shared helpers. The
format-specific input-denormal policy, raw integer ordering, signed-zero
normalization, predicate inversion, and ABS-before-NEG rules agree. CMP writes
a predicate mask and CMPX writes replacement EXEC; GoC returns either ordinary
result through the required `d` pointer, including zero for empty EXEC.

The architectures share numerical kernels but require distinct native exception
register contracts. Unsuffixed APIs now take RDNA3 `trapsts`; `_rdna4` APIs take
`excp_flag_user`, with `_wave64` last. Exact predicate results remain supported
for both architectures with null optional state. Existing RDNA4 exact reporting
and its previously captured fixtures are retained. No faithful RDNA3 comparison
exception model was found upstream: non-null `trapsts` with non-loose semantics
returns `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE` before touching any output, even
with empty EXEC or without strict semantics. No hardware reverse-engineering or
new exception model is part of this change.

Both architectures accept guest IEEE and input/output-denormal flags. Only input
flushing changes predicate results. Loose mode ignores either optional state
pointer and keeps the scalar/AVX2/AVX512 comparison dispatch. RDNA3 wrappers add
one instruction-level reporting-contract check and delegate to the same kernels;
there is no new per-lane branch, state work, or numerical abstraction.

`float_compare_architecture_test.cpp` checks all 84 mnemonics, both targets and
wave sizes, all eight guest policies, all source modifiers, all available CPU
selections, loose and strict exact results against the existing independent
reference, and non-null loose state preservation. It also covers empty EXEC,
unsupported reporting atomicity, DPP8/DPP16, aliased predicate destinations,
and host rounding/FTZ/DAZ/sticky-flag preservation. Existing RDNA4 hardware,
exhaustive FP16, Wave64, and exception tests now call explicit RDNA4 entry points.
Constant F/T and CLASS instructions are separate families, outside this review.

Performance: seven alternating before/after runs on CPU 2, thread CPU time,
minimum 20 ms per case, nine representative ordered/unordered/inverted
comparisons across all three formats, plain and modified/non-null loose-state
forms. Median changes are -0.01% scalar, +0.04% AVX2 and -0.12% AVX512;
the largest slowdown is scalar FP32 unordered, 23.156 ns to 23.924 ns (+3.31%).
Both SIMD object files are byte-identical. Raw results are in
`../goc-build/coverage-work/compare-architecture-performance.json`; the upstream
body/delegation review list is `compare-upstream-review.json` in that directory.

Validation: all 1,061 CTests pass with Clang 21 Release and GCC 15 Release;
26 distinct targeted ASan/UBSan tests pass, including comparison corpora,
optional-register rejection/loose contracts and instruction-flag validation.
Logs are `compare-tests.log` under `../goc-build/coverage-work/`,
`../goc-build-gcc/`, and `../goc-build-sanitize/`; the latter also has
`compare-contract-tests.log`. All nine audit-tool tests pass. The review ledger
now records 135 mnemonic contracts. The naming audit resolves 886 of 924
reporting signatures, with zero mismatches and 38 unresolved. Neither count is
a whole-project numerical-completeness claim.

## FP16 pack and FP32 byte-pack architecture contracts

Reviewed `v_pack_b32_f16` and `v_cvt_pk_u8_f32` on both architectures and wave
sizes. Rocjitsu revision `ffc144c564c` has the same generated VOP3 packing bodies
on both targets (apart from the spelling of the true16 selector field). The byte
conversion delegates to `execute_v_cvt_pk_u8_f32_vop3`. Neither reviewed body
uses IEEE or denormal control flags, and neither provides a faithful exception
classifier to borrow.

Unsuffixed entry points now take RDNA3 `trapsts`; `_rdna4` entry points take
`excp_flag_user`, before `_wave64` where applicable. Both accept guest IEEE and
denormal flags without numerical effect. Loose paths share the existing scalar,
AVX2 and AVX512 kernels, skip optional reporting, and preserve prior state.
Exact non-null reporting remains unsupported, including empty EXEC and non-strict
requests. No new per-lane reporting or host FP-control changes were introduced.

Two upstream-model limitations remain explicit:

- Upstream FP16 PACK combines raw source halves; GoC's pre-existing RDNA4 corpus
  establishes signaling-NaN quieting and source modifiers. The RDNA4 exact path
  and exhaustive fixture are retained. RDNA3 uses the existing kernel in loose
  mode and rejects strict exact semantics; the RDNA4 capture is not evidence
  for RDNA3 NaN behavior. Non-strict exact requests may fall back to loose.
- Upstream FP32 byte conversion truncates, whereas GoC's existing RDNA4 corpus
  expects nearest-even rounding. Both GoC architectures retain that existing
  loose policy, saturation and NaN-to-zero. This discrepancy does not establish
  a hardware architecture difference. Neither target gains strict exact support.

The RDNA3 manual describes packing FP16 sources and converting/packing an
unsigned byte, but does not resolve these upstream-model discrepancies. No new
hardware probing or bit-exact model was developed. This review records the loose
contracts and known limitations rather than claiming new empirical exactness.

Four new tests in `pack_architecture_test.cpp` cover both operations, targets,
wave sizes, all eight guest policies, all source modifiers, all available CPU
paths, zero/full/sparse EXEC, every whole-register output alias with guards,
DPP8/DPP16, unsupported exact/reporting atomicity and host rounding/FTZ/DAZ
preservation. Existing RDNA4 numerical, exhaustive pack, byte-rounding and
hardware-fixture tests now call explicit RDNA4 entry points.

Performance: seven alternating before/after runs, pinned to CPU 2, using thread
CPU time and at least 20 ms per case. Both operations were measured with plain
and modified/non-null loose-state calls. Median time changes are -1.64% scalar,
-0.21% AVX2 and -0.75% AVX512; the largest measured slowdown is +0.13%. All four
SIMD objects are byte-identical. Raw samples are in
`../goc-build/coverage-work/pack-architecture-performance.json`.

Validation: all 1,065 CTests pass with Clang 21 Release and GCC 15 Release;
38 targeted ASan/UBSan tests pass. Logs are `pack-architecture-tests.log` under
`../goc-build/coverage-work/`, `../goc-build-gcc/`, and
`../goc-build-sanitize/`. All nine audit-tool tests pass. The ledger records
137 reviewed mnemonic contracts; the native-output audit resolves 894 of 928
signatures with zero mismatches and 34 unresolved. Remaining families and
whole-project gates are still open.

## Packed FP16 ADD/MUL architecture policies

Reviewed `v_pk_add_f16` and `v_pk_mul_f16` on RDNA3/RDNA4. Both generated VOP3P
bodies at rocjitsu revision `ffc144c564c` use
`fp_mode::packed_binary_f16`: source selection and per-half negation precede
FP16 input flushing, arithmetic, destination rounding/clamp and output flushing.
Packed arithmetic has no OMOD gate. The helper also contains architecture/IEEE
NaN rules and guest rounding support; these are not claimed by GoC's loose path.

GoC now accepts IEEE and input/output-denormal flags for packed ADD/MUL and
extends the existing ordinary FP16 arithmetic flush rules to the packed scalar
and AVX2 loops. Both output halves use original inputs before masked stores.
No additional algorithm, sparse-mask optimization or per-lane state reporting
was introduced. Unsuffixed APIs take RDNA3 `trapsts`; `_rdna4` APIs take
`excp_flag_user`, with `_wave64` last. Both architectures share their numerical
kernels. Loose calls leave state unchanged; exact non-null reporting remains
unsupported before operand access, including empty EXEC and non-strict requests.

Strict exact results remain unsupported. In particular, the upstream packed
helper constructs `ScopedFenv`, which explicitly changes host FP controls. That
implementation cannot be copied as-is: the host-environment prohibition takes
precedence over porting upstream numerical code. No environment guard or new
hardware-derived exact model was added. A future exact port must adapt the
borrowed rounding/NaN logic without changing host controls; it is not established
by this loose-policy review. Guest directed rounding is also not newly exposed.

Three new cases in `packed_binary_test.cpp` cover both operations, targets and
wave sizes, all guest policies, all packed source selectors/negations/clamp,
all available CPU paths, overflow modes, sparse/full EXEC, destination aliases
and guards. Literal witnesses separate input flushing from output flushing and
check signed tiny multiplication results. Validation covers exact/reporting
errors with untouched outputs and host rounding/FTZ/DAZ preservation. Existing
packed min/max and ordinary FP16 arithmetic tests remain regression coverage for
the shared implementation; their contracts are unchanged by this slice.

The scalar lane loop is now named `binary_scalar` and remains shared with
ordinary FP16 arithmetic and selection. A compile-time guest-flush switch keeps
packed default-mode checks outside that loop. This adds one local parameter,
not a second numerical implementation; a maintainer still edits arithmetic,
modifiers, conversion and stores in one scalar body. The helper is not inlined
so the compiler does not merge the default and guest branches back into a
per-lane check. SIMD dispatch remains direct.

Performance: seven alternating before/after samples per case, pinned to CPU 2,
thread CPU time, at least 20 ms each, plain and modified/non-null loose-state
ADD/MUL. Packed medians are -0.31% scalar, +0.20% AVX2 and +0.72% for the v4
selection (which also uses AVX2); the largest slowdown is +0.95%. Ordinary FP16
ADD/MUL regression medians are -0.32%, -0.65% and +0.07%, respectively, with a
maximum +1.54%. Raw samples are `packed-binary-performance.json` and
`half-binary-regression-performance.json` in `../goc-build/coverage-work/`.

Validation: all 1,068 CTests pass with Clang 21 Release and GCC 15 Release;
42 targeted ASan/UBSan tests pass, including ordinary/packed arithmetic,
legacy/new min/max, DPP, instruction flags and optional-state contracts. The
exhaustive sanitizer DPP min/max case completes in about 209 seconds; it was
allowed to finish rather than restarted. Logs are `packed-binary-tests.log`
under `../goc-build/coverage-work/`, `../goc-build-gcc/`, and
`../goc-build-sanitize/`; the final test-bracing correction was rechecked with
all ten `PackedBinary` sanitizer cases in `packed-binary-final-tests.log`.
All nine audit-tool tests pass. The ledger records 139 contracts; the native
state audit resolves 902 of 932 signatures with zero mismatches and 30 unresolved.

## FP16 EXP/LOG architecture policies

Reviewed `v_exp_f16` and `v_log_f16` VOP1/VOP3 forms on RDNA3/RDNA4 at rocjitsu
revision `ffc144c564c`. Both use `transcendental::log_exp_f16` and its SIMD
counterpart. Input subnormals follow the half denormal policy; results round to
half and then undergo output flushing before OMOD. RDNA3 disables OMOD when
IEEE is set or output denormals are preserved; RDNA4 keeps scaling enabled.

Unsuffixed APIs now take RDNA3 `trapsts`; `_rdna4` APIs take `excp_flag_user`,
with `_wave64` last. The existing `run`/`unary_wave64` architecture policy handles
both names. Scalar and AVX2 loose paths now accept guest IEEE/denormal flags and
apply the input/output flush stages. Optional loose state remains unchanged;
strict exact results and exact non-null reporting remain unsupported. The
upstream helper explicitly constructs `ScopedEnvironment`; no such guard was
copied. Its double-precision LOG/EXP plus direct half rounding remains a future
exact-port candidate requiring a compliant adaptation and validation. No new
hardware-derived approximation or exception model was developed.

Three new tests in `half_log_exp_architecture_test.cpp` cover both operations,
architectures and wave sizes, all eight guest policies, all source/destination
half selectors, ABS/NEG/OMOD/CLAMP, all available CPU paths, overflow mode,
DPP8/DPP16, masks, aliases, unselected half preservation and guards. An independent
double-precision reference allows one half ULP for loose finite results. Literal
witnesses verify EXP scaling, LOG of a flushed tiny input and EXP output flushing.
Exact/reporting errors preserve outputs and state; tests check host rounding and
FTZ/DAZ controls without requiring ordinary arithmetic to preserve sticky flags.
Existing RDNA4 fixtures and numerical tests now call explicit RDNA4 APIs.

A single `GuestDenormals` template parameter in the existing scalar/SIMD bodies
keeps default EXP/LOG loops free of per-lane flush checks. Dispatch selects that
variant once per call. The scalar function is kept out of line to prevent loop
merging from reintroducing those checks. Other unary operations retain their
existing guest-aware instantiations. The arithmetic, modifier and store logic
remain shared rather than duplicated as separate algorithms.

Performance: seven alternating before/after samples, pinned to CPU 2, thread
CPU time, at least 20 ms per case, plain and modified/non-null loose-state calls.
EXP/LOG medians are -0.07% scalar, +0.43% AVX2 and -0.02% for the v4 selection
(also AVX2); the largest slowdown is +0.86%. SQRT/RCP regression medians are
-0.05%, -0.91% and -1.06%, with a maximum slowdown of +0.66%. Raw samples are
`half-log-exp-performance.json` and `half-unary-regression-performance.json`
in `../goc-build/coverage-work/`.

Validation: all 1,071 CTests pass with Clang 21 Release and GCC 15 Release;
26 targeted ASan/UBSan tests pass, including existing unary exact, DPP,
modifier/alias, instruction-flag and optional-state coverage. Logs are
`half-log-exp-tests.log` under `../goc-build/coverage-work/`,
`../goc-build-gcc/`, and `../goc-build-sanitize/`. All nine audit-tool tests pass.
A production-source scan finds no explicit host FP-environment mutations.
The ledger records 141 contracts; the native-state audit resolves 910 of 936
signatures, with zero mismatches and 26 unresolved.

## DIV_FMAS architecture policies

Reviewed `v_div_fmas_f32` and `v_div_fmas_f64` on both architectures against
rocjitsu `ffc144c564c`. Both generated VOP3 handlers delegate to the same
`execute_shared.h` helpers and integer `shared/division.h::div_fmas` model.
Inputs always preserve denormals; `div_round` applies guest output flushing
while preserving the sign of zero. No host FP-environment guard is needed or
copied. Nearest-even fused post-scaling remains the existing borrowed algorithm;
other guest rounding modes remain outside the currently exposed FP controls.

Unsuffixed APIs now use RDNA3 `trapsts`; explicit `_rdna4` APIs retain
`excp_flag_user`, before `_wave64`. RDNA3 ISA section 7.2.3.1 disables OMOD
when IEEE is set or output denormals are preserved; RDNA4 retains its established
independent OMOD behavior. All source ABS/NEG, CLAMP, execution masks, VCC input
bits and whole-register aliases remain supported in loose semantics. The shared
scalar, AVX2 and AVX512 integer kernels now support guest output flushing.
Input-flush and IEEE flags are accepted; input flushing is deliberately ignored.

Exact results without effective OMOD use the borrowed integer model on both
architectures. The existing RDNA4 exact OMOD implementation and fixtures remain
intact. Rocjitsu's shared FMAS handlers do not supply an RDNA3 exact OMOD model,
so strict exact with active RDNA3 OMOD is explicitly unsupported instead of
assuming that RDNA4's captured modifier behavior applies. Non-strict requests
may fall back to loose semantics. Exact non-null reporting remains unsupported
for RDNA3 and for the newly supported RDNA4 output-flush policy; the existing
RDNA4 reporting model is retained for preserved outputs. These errors occur
before operand reads, including zero EXEC. Loose calls skip reporting and leave
optional state unchanged. No new hardware-derived numerical/exception model
was developed.

Four architecture tests cover both widths and wave sizes, all eight guest
policies, all 512 modifier combinations, all available CPU paths, masked stores,
source/destination and D0/D1 aliases, literal signed-subnormal and scaling
witnesses, borrowed rocjitsu fused-rounding/overflow/cancellation cases, reporting
errors and unchanged host FP state. Existing RDNA4 numerical and exception
fixtures now call the explicit RDNA4 names. The shared scalar kernel is out of
line so architecture wrappers do not duplicate the arithmetic; output flushing
runs after that arithmetic loop rather than adding a branch to every scalar FMA.

Performance: seven alternating before/after runs, pinned to CPU 2, thread CPU
time, at least 20 ms per case, covering FP32/FP64 and plain/modified loose calls
with optional reporting enabled in the modified cases. Median time changes are
+1.25% scalar, +0.66% AVX2 and +0.05% AVX512; the largest slowdown is +3.15%.
Raw samples are `../goc-build/coverage-work/div-fmas-performance.json`.

Validation: all 1,075 CTests pass in Clang 21 Release and GCC 15 Release;
27 targeted ASan/UBSan tests pass. The four architecture tests were rechecked
with Clang and sanitizers after adding braces required by GCC's dangling-else
warning around a test assertion. Logs are `div-fmas-tests.log` and
`div-fmas-final-tests.log` under `../goc-build/coverage-work/`,
`../goc-build-gcc/`, and `../goc-build-sanitize/` as applicable. All nine audit-tool
tests pass. A fresh disassembly and undefined-symbol scan of all 192 library
objects in each compiler build finds no explicit host FP-environment writes or
mutator references (`div-fmas-fp-environment-audit.json` in the coverage work
folder). The ledger records 143 contracts; the native-state naming audit resolves
918 of 940 signatures, with zero mismatches and 22 unresolved. These counts are
review progress, not whole-project completion claims.

## FP32/FP64 DIV_FIXUP architecture policies

Reviewed both architectures' FP32/FP64 VOP3 handlers at rocjitsu `ffc144c564c`.
They delegate to the shared integer `division.h::div_fixup` and `div_apply_omod`
models. Guest input flushing affects the original denominator/numerator, never
the provisional quotient. Output-denormal mode does not generally flush that
quotient; nearest-even extreme exponent-gap recovery already produces signed
zero. `fp_mode::effective_omod` gates RDNA3 output scaling on preserved output
denormals and IEEE mode, while RDNA4 applies scaling independently.

Unsuffixed FP32/FP64 APIs now take native RDNA3 `trapsts`; `_rdna4` variants
retain `excp_flag_user`, with `_wave64` last. The shared numerical model supports
loose and exact results with all eight combinations of the exposed IEEE and
denormal flags. A separate integer preprocessing helper flushes B/C when asked,
then calls the established kernel with that input-flush bit consumed. This
preserves A and all source reads before stores, and lets both architectures reuse
the same scalar, AVX2 and AVX512 arithmetic without adding guest-flush branches
to each SIMD batch. No host FP-environment changes are performed.

Existing RDNA4 exact reporting remains supported when input flushing is off.
Exact non-null reporting with input flushing, and all RDNA3 exact non-null
reporting, return `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE` before source access,
including zero EXEC. Rocjitsu's classifier includes gfx1201-specific findings;
this slice does not assume those prove RDNA3 flags or invent a classifier for
the new input policy. Loose calls leave optional state unchanged. FP16's promoted
arithmetic, narrowing and saturation rules are covered by the subsequent
[FP16 review](#fp16-div_fixup-architecture-policies).

Three new architecture tests cover FP32/FP64, both wave sizes and architectures,
all eight guest policies and all 512 source/output modifier encodings, every
available CPU level, masks, source/destination and D0/D1 aliases. Expected values
use the existing independent long-double reference with only B/C preflushed;
literal cases borrowed from rocjitsu's division tests distinguish preserved A
from flushed B/C. Errors leave destination/state untouched, DPP remains rejected
for these two formats, and all host rounding/FTZ/DAZ controls remain unchanged.
Existing RDNA4 hardware and exception fixtures use explicit RDNA4 names.

Performance: seven alternating before/after runs, CPU 2 affinity, thread CPU
time, at least 20 ms per case, plain and source-modified FP32/FP64 calls with
non-null loose reporting in the modified cases. Median time changes are -2.47%
scalar, +0.70% AVX2 and -0.89% AVX512; the largest slowdown is +3.67%. Both SIMD
objects are byte-identical to the baseline (9,976 and 9,736 bytes respectively).
Raw samples are `../goc-build/coverage-work/div-fixup-performance.json`.

Validation: all 1,078 CTests pass in Clang 21 Release and GCC 15 Release;
27 targeted ASan/UBSan cases pass, including existing FP16 DPP/alias coverage,
FP32/FP64 architecture tests, modifiers and optional-state contracts. Logs are
`div-fixup-tests.log` under `../goc-build/coverage-work/`, `../goc-build-gcc/`
and `../goc-build-sanitize/`. All nine audit-tool tests pass. Disassembly and
undefined-symbol scans of all 192 library objects in both compiler builds find
no explicit host FP-environment mutation (`div-fixup-fp-environment-audit.json`
in the coverage work folder). The ledger records 145 contracts; the native-state
naming audit resolves 926 of 944 signatures, with zero mismatches and 18 unresolved.
These counts do not establish whole-project completion.

## FP16 DIV_FIXUP architecture policies

Ported the RDNA3 generated `VDivFixupF16Vop3` sequence from rocjitsu
`ffc144c564c`: integer exceptional-case selection via `div_fixup_f16`, exact
power-of-two scaling in the promoted domain, nearest-even half narrowing with
finite-overflow saturation, and `finalize_omod_f16`. GoC implements these stages
with integer bit operations, so no borrowed environment guard or host rounding
change is needed. This is a port of the upstream RDNA3 contract, not newly
reverse-engineered hardware behavior; the RDNA4 capture-backed numerical and
exception models are retained separately.

The RDNA3 provisional-overflow result remains infinity even with `FP16_OVFL`;
only finite scaling overflow saturates. Promoted scaling can turn an input half
subnormal into a normal output, and halving just below twice the minimum normal
can round up to a normal result. Active OMOD then flushes rounded subnormals and
signed zero to +0. These stages differ from the retained RDNA4 captured model's
pre-scaling flush and provisional-overflow saturation. RDNA3 gates OMOD on guest
IEEE/output-denormal policy as for other ordinary half results. DX10_CLAMP=1 is
the modeled RDNA3 NaN-clamp policy; arbitrary guest rounding/DX10_CLAMP controls
remain outside the current API.

Unsuffixed half APIs now use RDNA3 `trapsts`; `_rdna4` retains `excp_flag_user`,
before `_wave64`. All DIV_FIXUP formats accept the three exposed guest FP flags.
The existing input-flush preprocessing helper now supports selected half B/C
operands while preserving A, including DPP on A and all half selectors. Output
flushing does not independently discard the provisional quotient. Exact non-null
reporting remains unsupported for RDNA3 and for RDNA4 input flushing; loose
reporting remains skipped. Existing RDNA4 reporting without input flushing is
unchanged.

A single compile-time `Rdna3Half` choice shares operand classification, loads,
alias staging and masked stores. The RDNA3 half output-stage helper is shared by
the AVX2 and AVX512 backends through their existing operation wrappers; it handles
only the promoted-scale/narrow/flush distinction. Scalar code reuses the existing
integer power-of-two rounding helper. There is no separate duplicated FIXUP
algorithm and no sparse-mask optimization.

Four architecture tests cover every provisional half encoding at all four OMOD
settings with saturation on/off, both architectures and every available CPU path.
They also cover eight guest policies, both wave sizes, sampled source/output
modifiers and half selectors, DPP8/DPP16, masks, aliases with each source, retained
destination halves/guards, literal policy differences, reporting errors and
host-control preservation. An independent promoted-arithmetic reference supplies
expected RDNA3 outputs. Existing exhaustive RDNA4 DPP/alias and hardware/exception
fixtures continue to test the explicit RDNA4 names.

The new host-state test also exposed a pre-existing Clang baseline transformation:
auto-vectorized variable integer shifts use `cvttps2dq`, which can raise a sticky
exception flag. The before archive contains the same instruction. This does not
violate the standing prohibition on explicit FP-environment changes. The public
FIXUP comment and new tests now distinguish unchanged controls from permitted
arithmetic sticky flags instead of promising preservation of every status bit.

A standalone C++20 harness directly includes rocjitsu's `division.h` and
`util/data_types.h` and reproduces the generated RDNA3 output sequence with
DX10_CLAMP=1. All 4,194,304 lane comparisons match, across all four available CPU
selections, every provisional half encoding (with permutations for B/C), guest
policies, saturation, and source/output modifiers. Its source and log are
`half-div-fixup-upstream.cpp` / `.log` in `../goc-build/coverage-work/`; GoC's
production code and committed tests still build as C++17 without rocjitsu.

Performance: seven alternating before/after samples, CPU 2 affinity, thread CPU
time, at least 20 ms per case, plain and source-modified loose calls with non-null
optional state in the modified cases. RDNA4 FP16 median time changes are +0.07%, +0.06%, +1.13%
for scalar, AVX2 and AVX512, with a maximum +1.45%. FP32/FP64 regression medians
are +1.24%, +0.36%, +1.05% respectively, with a maximum +2.31%. Raw samples are
`half-div-fixup-performance.json` and `half-div-fixup-wide-performance.json`
in the coverage work folder.

Validation: all 1,082 CTests pass with Clang 21 Release and GCC 15 Release;
34 targeted ASan/UBSan tests pass, including both FIXUP architecture suites,
existing DPP/aliases, arithmetic exceptions, instruction flags and global-state
contracts. Logs are `half-div-fixup-tests.log` under `../goc-build/coverage-work/`,
`../goc-build-gcc/`, and `../goc-build-sanitize/`. All nine audit-tool tests pass.
A disassembly/undefined-symbol scan of all 192 library objects in each compiler
build finds no explicit FP-control writes or environment-mutator references
(`half-div-fixup-fp-environment-audit.json` in the coverage work folder).
The ledger records 146 contracts; native-state naming resolves 930 of 946
signatures, with zero mismatches and 16 unresolved. Whole-project completion
remains unproven.

## TRIG_PREOP architecture policies

`v_trig_preop_f64` now distinguishes rocjitsu's RDNA3 1201-bit 2/pi
lookup from the retained RDNA4 1184-bit capture-backed table. The unsuffixed
Wave32/Wave64 APIs use RDNA3 `trapsts`; `_rdna4` precedes `_wave64` and uses
`excp_flag_user`. Both support loose and empirical-exact numerical results;
non-null exact global-state reporting remains explicitly unsupported. Loose
calls skip reporting.

The borrowed model is `execute_v_trig_preop_f64_vop3` in rocjitsu revision
`ffc144c564c`, including `scale_u53_f64_rtz`, `effective_omod` and `finish_f64`.
The upstream implementation temporarily sets host rounding modes. GoC does not
copy those guards: constexpr integer table construction implements RTZ lookup,
and integer power-of-two scaling implements nearest-even output modification.
RDNA3 scales before flushing the rounded tiny result; the existing RDNA4 path
retains pre-scaling flushing as well. RDNA3 ignores OMOD when output denormals
are preserved or IEEE mode is enabled. Input/output-denormal and IEEE flags
are accepted, but otherwise do not alter the lookup. Arbitrary guest output
rounding modes remain outside the exposed API. A's fraction/sign do not affect
lookup, including special encodings, and A's unused low-word pointer may be null.

A compile-time architecture choice shares lookup indexing, validation, wave
splitting, alias staging and stores. There are two constexpr lookup arrays;
AVX2 still gathers four FP64 lanes and AVX512 gathers eight. The small output
scaling branches show the architecture distinction directly, without a new
policy framework or host-state helper. No sparse-mask specialization is added.

Four architecture tests cover every exponent/selector with all OMOD/CLAMP
settings, both wave sizes and all available CPU selections; all eight guest
policies; source and destination aliases, inactive lanes and storage guards;
literal table-tail and normal/subnormal witnesses; errors, nullable reporting,
and host rounding/FTZ/DAZ controls. The independent test reference uses wider
floating arithmetic and explicit RTZ correction instead of production integer
helpers. Existing RDNA4 exhaustive hardware fixtures retain their expectations
and now call the explicit RDNA4 names. No new hardware model was inferred.

Seven alternating before/after timing samples use CPU 2 affinity, thread CPU
time and at least 20 ms per case. Plain/OMOD+CLAMP RDNA4 loose calls change by
+1.17%/+1.67% for scalar, +0.36%/-1.53% for AVX2, and -2.09%/-0.90% for AVX512.
Modified cases pass a non-null optional state pointer. Raw samples and harness
are `trig-preop-performance.json` and `trig-preop-timing.cpp` under
`../goc-build/coverage-work/`.

Validation: all 1,086 CTests pass with Clang 21 Release and GCC 15 Release;
17 targeted ASan/UBSan tests pass, covering both TRIG_PREOP suites, instruction
flags and architectural-state contracts. All nine audit-tool tests pass.
Disassembly and undefined-symbol scans of all 192 library objects for each
compiler find no explicit FP-control writes or environment-setter references.
Logs and `trig-preop-fp-environment-audit.json` are under the coverage work folder;
GCC and sanitizer logs use the `trig-preop-` prefix in their build directories.
The review ledger now records 147 contracts; native-state naming resolves 934
of 948 signatures, with zero mismatches and 14 unresolved. These counts are
review progress, not proof of complete architecture coverage.

## MULLIT architecture policies

`v_mullit_f32` now has unsuffixed RDNA3 and explicit `_rdna4` variants in both
wave sizes, with native `trapsts`/`excp_flag_user` outputs. These are loose
numerical implementations. Strict exact requests and non-null exact reporting
remain explicitly unsupported; loose calls skip optional reporting.

The RDNA3 lighting/sentinel calculation, OMOD gating, and final +0 normalization
follow rocjitsu revision `ffc144c564c`, `execute_v_mullit_f32_vop3`,
`effective_omod` and `finalize_omod_f32`. RDNA3 disables OMOD when output denormals
are preserved or IEEE mode is set. Active OMOD scales before flushing tiny
results and signed zeros to +0. RDNA4 retains the existing captured pre-scaling
flush and signed-zero result when a normal product becomes tiny after scaling.
CLAMP models DX10_CLAMP=1; alternate DX10 behavior and arbitrary guest rounding
remain outside this contract. No new bit-exact or hardware-derived model is added.

Explicit guest input/output flushing is supported according to the ISA's ordinary
FP32 denormal controls: input flushing includes the lighting-condition operand C,
and output flushing retains the sign unless active RDNA3 OMOD already produced
+0. This extends the borrowed helper, which does not explicitly implement those
ordinary flush stages. All three exposed guest FP flags are accepted. The
implementation never changes the host FP environment; host nearest-even with
preserved denormals remains the numerical precondition for loose arithmetic.

The architecture choice is compile-time and only changes output scaling inside
the shared scalar/AVX2/AVX512 arithmetic. A local `compute_flushed` helper stages
explicit guest flushing around those same kernels. The default SIMD kernels
have no new guest-flag checks; AVX2 still handles eight FP32 lanes and AVX512
sixteen. The dispatch helper is shared by direct and flushed calls, and a single
out-of-line scalar loop per architecture avoids duplicating that arithmetic.
DPP, Wave64 row handling, alias rules and mask-independent work are retained.

Four architecture tests cover all 512 instruction-modifier combinations, eight
guest policies, random and special values, both wave sizes/architectures, DPP8
and DPP16, masks, shared/aliased sources and destinations, guards, reporting
errors and all host rounding/FTZ/DAZ settings. A further Cartesian traversal in
one of those tests checks all 4096 triples of the existing special-value corpus
under all 512 modifiers against the RDNA3 reference. Literal witnesses cover
OMOD gating, subnormal normalization, signed-zero differences and condition-input
flushing. Existing RDNA4 capture digests are unchanged and use the `_rdna4` APIs.

Seven alternating before/after timing samples, CPU 2 affinity, thread CPU time,
at least 20 ms per case: plain/OMOD+CLAMP RDNA4 loose calls change by
+0.79%/-0.43% for scalar, +1.80%/+1.67% for AVX2 and +1.91%/+2.14% for AVX512.
Modified calls pass a non-null optional state pointer. The harness and raw data
are `mullit-timing.cpp` and `mullit-performance.json` in
`../goc-build/coverage-work/`.

Validation: all 1,090 CTests pass with Clang 21 Release and GCC 15 Release;
20 targeted ASan/UBSan tests pass, including both MULLIT suites, instruction
flags and architectural-state contracts. All nine audit-tool tests pass.
Disassembly and undefined-symbol scans of all 192 objects for each compiler
find no explicit FP-control writes or environment-setter references. Clang
logs and `mullit-fp-environment-audit.json` are in the coverage work folder;
GCC/sanitizer logs use the `mullit-` prefix in their build directories. The ledger
records 148 architecture contracts, and native-state naming resolves 938 of 950
signatures with zero mismatches and 12 unresolved. Remaining naming reviews are
half-result DOT2 and SIN/COS; broader whole-project coverage remains unproven.

## Half-result DOT2 architecture contracts

`v_dot2_f16_f16` and `v_dot2_bf16_bf16` have identical RDNA3/RDNA4 numerical
handlers in rocjitsu revision `ffc144c564c`: both their ordinary and modifier
execution bodies compare byte-for-byte equal after extraction from the generated
per-architecture VOP3 files. Both call shared `fp_mode::dot2_f16`/`dot2_bf16`.
GoC retains one scalar implementation per format and the existing shared AVX2
kernel; no architecture-specialized arithmetic is introduced.

The architectural state output is distinct: unsuffixed APIs now name RDNA3
`trapsts`, and new `_rdna4` APIs name `excp_flag_user`, before `_wave64`.
These small wrappers are needed for the native-register contract, not because
DOT2's arithmetic differs. Loose calls continue to skip optional reporting;
non-null exact requests return `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE` before operand
access, including empty EXEC. Numerical strict exact semantics remain unsupported.

RDNA3 ISA section 7.2.4 and the shared rocjitsu helpers establish the fixed
half-result policy: F16 supports denormals; BF16 flushes its input/output
denormals; both narrow nearest-even. F16 alone honors FP16_OVFL. The exposed
input/output-denormal and IEEE flags are now accepted without changing this
policy. Six source ABS/NEG modifiers, C/D half selection, DPP8/DPP16, and
preservation of the other destination half are retained. OMOD, CLAMP and A/B
half selectors remain outside the API's supported modifier set. The operand
interface takes already-expanded VGPR words; instruction-decoder processing of
inline constants is not added here.

Rocjitsu's helpers change host rounding through `ScopedFenv`. GoC does not copy
or invoke those guards: these remain loose implementations with a host
nearest-even/preserve-denormals numerical precondition. Ordinary arithmetic may
raise sticky flags; no host FP controls are changed. No new empirical exact
model, numerical reverse-engineering or exception model is introduced.

Four new architecture tests cover all eight guest policies, both formats and
architectures, all supported modifier/half-selection combinations, DPP, wave
sizes, masks, shared sources, destination aliases, retained halves and guards.
They also check every accumulator encoding under every guest policy, literal
rounding/overflow/denormal behavior including BF16's ignored FP16_OVFL, optional
state/error contracts and unchanged host rounding/FTZ/DAZ controls. Small-integer
results provide an independent reference; existing RDNA4 literal, random and
hardware/DPP fixture tests now call the explicit RDNA4 APIs. The Clang AVX2 object
is byte-for-byte unchanged from the before archive.

Performance: seven alternating before/after samples with CPU 2 affinity, thread
CPU time and at least 20 ms per case, covering both formats with plain and
ABS/NEG/half-selection calls. Modified calls pass a non-null optional state.
All changes are between -2.03% and +0.96%; both higher CPU selectors use the
same unchanged AVX2 kernel. Raw samples and harness are
`half-dot-performance.json` / `half-dot-timing.cpp` in the coverage work folder.
The upstream handler hashes/locations are in `half-dot-upstream-review.json`.

Validation: all 1,094 CTests pass with Clang 21 Release and GCC 15 Release;
24 targeted ASan/UBSan tests pass for both half-result DOT2 suites, instruction
flags and architectural-state contracts. All nine audit-tool tests pass. Scans
of all 192 library objects for each compiler find no explicit FP-control writes
or environment-setter references (`half-dot-fp-environment-audit.json`). Clang
logs use the `half-dot-` prefix under `../goc-build/coverage-work/`; GCC and
sanitizer logs use that prefix in their build directories. The review ledger
records 150 contracts. Native-state naming resolves 946 of 954 signatures,
with zero mismatches and eight unresolved SIN/COS signatures. This does not
establish whole-project completion.

## FP32 SIN/COS architecture policies

`v_sin_f32` and `v_cos_f32` now expose unsuffixed RDNA3 `trapsts` and explicit
`_rdna4` `excp_flag_user` APIs in both wave sizes. Both share the existing integer
range reduction and captured approximation ported from rocjitsu
`util/amdgpu_trig.h`. Its already-supported denormal and NaN-policy parameters
are now wired to the exposed guest flags. No coefficients or numerical model
were inferred from new hardware work.

The RDNA3 policy follows rocjitsu revision `ffc144c564c`, generated shared SIN/COS
VOP3 handlers, `transcendental::{sin,cos}_f32`, `quiets_nan`, `effective_omod` and
`apply_omod_f32`. IEEE mode enables NaN quieting and disables OMOD. FP32 trig
uses the forced-flush OMOD policy independently of guest output-denormal mode:
tiny results become +0 before scaling, while halving a normal result into the
subnormal range produces signed zero. The existing RDNA4 result model retains
its validated RNE scaling followed by a tiny-result/+0 flush. Guest denormal
handling precedes either output stage. CLAMP models DX10_CLAMP=1; alternative
DX10 behavior is not added.

Numerical empirical-exact semantics work with all three guest FP flags. RDNA3
non-null exact reporting remains unsupported; RDNA4 exact reporting with guest
input/output flushing is also rejected before operand access. Existing RDNA4
reporting with preserved denormals remains unchanged. Loose calls skip reporting.
Upstream generated handlers use a host-environment guard, but the borrowed
integer evaluator needs none: GoC does not copy or call those guards.

The scalar/exact loop is shared through compile-time architecture and SIN/COS
choices. A small integer output stage expresses the RDNA3/RDNA4 distinction.
The existing eight-lane AVX2 approximation is unchanged byte-for-byte. RDNA3 and
explicit guest-flush loose calls stage input/output policies around that same
SIMD kernel; default RDNA4 calls retain direct dispatch. These calls do not fall
back to scalar transcendental evaluation. DPP, Wave64 row handling, masks and
alias staging remain shared. No sparse-mask optimization or host-state helper
is introduced.

Three architecture tests combine existing captured golden results with independent
output scaling, all guest policies and supported modifiers, both wave sizes,
DPP8/DPP16, masks, aliases, guard preservation, signaling-NaN/IEEE and denormal
witnesses, reporting errors and host rounding/FTZ/DAZ controls. Approximate
comparisons apply only to participating loose numerical results; guards,
inactive lanes, zero policies and NaNs are checked as bits. Existing RDNA4
numerical and exception hardware fixtures use the explicit RDNA4 APIs and retain
their original expectations.

A standalone C++20 harness directly includes rocjitsu's `util/amdgpu_trig.h` and
`fp_mode.h`, applying the actual integer evaluator and RDNA3 output helpers at
host nearest-even without using any environment guard. All 8,388,608 lane
comparisons match, across both operations, eight guest policies, all source/output
modifiers, random/denormal/NaN/normalization witnesses and all available CPU
selectors. Source and output are `trig32-upstream.cpp` and `.log` in the coverage
work folder. Production GoC and committed tests remain independent C++17 code.

Seven alternating before/after timing samples, CPU 2 affinity, thread CPU time,
at least 20 ms per case: RDNA4 SIN/COS plain and OMOD+CLAMP cases change between
-1.58% and +1.58% across scalar and both AVX2-dispatch CPU selectors. Modified
calls pass a non-null optional-state pointer. Raw timing data and harness are
`trig32-performance.json` / `trig32-timing.cpp` in `../goc-build/coverage-work/`.

Validation: all 1,097 CTests pass with Clang 21 Release and GCC 15 Release;
27 targeted ASan/UBSan tests pass, including both numerical/architecture suites,
the original exception hardware corpora, instruction flags and native-state
contracts. The generated reporting-test exclusion now names only the reporting
RDNA4 FP32 variants, so new RDNA3 variants enter the exhaustive rejection test.
All nine audit-tool tests pass. Scans of all 192 library objects per compiler
find no explicit FP-control writes or environment-setter references
(`trig32-fp-environment-audit.json`). Logs use the `trig32-` prefix in the Clang
coverage work folder and GCC/sanitizer build directories. README and PLAN now
state the architecture-specific reporting limits. The review ledger records
152 contracts; native-state naming resolves 954 of 958 signatures, with zero
mismatches and four unresolved FP16 SIN/COS signatures. Whole-project completion
remains unproven.

## FP16 SIN/COS architecture policies

`v_sin_f16` and `v_cos_f16` now distinguish unsuffixed RDNA3 `trapsts` from
`_rdna4` `excp_flag_user`, for both wave sizes. The integer numerical model and
half-result rounding/output stage remain shared. Policies are ported from
rocjitsu revision `ffc144c564c`, generated shared VOP3 handlers,
`transcendental::map_f16`, `effective_f16_omod`, `apply_omod_f16` and
`finalize_omod_f16`. RDNA3 quiets signaling NaNs only in IEEE mode and enables
OMOD only when output denormals are flushed and IEEE is clear. RDNA4 always
quiets signaling NaNs and honors OMOD. Guest input flushing precedes the model;
guest output flushing follows half rounding and precedes output scaling.
Active OMOD flushes tiny results and maps signed zero to +0. CLAMP retains
DX10_CLAMP=1 behavior. FP16_OVFL is accepted but cannot affect the bounded results.

Exact numerical semantics support all three guest FP flags. Non-null exact
reporting is unsupported on RDNA3 and with guest input/output flushing on RDNA4,
including zero EXEC and non-strict calls; existing preserved-denormal RDNA4
reporting is unchanged. Loose calls skip reporting. No host-environment guards
are copied or called. The existing eight-lane AVX2 kernel remains unchanged;
guest policies are staged around it when needed, and default RDNA4 calls retain
direct dispatch. Half selection, masks, DPP and alias staging remain shared.

Four architecture tests cover every half encoding with every guest policy and
OMOD, all 128 modifiers with DPP/half selectors/masks/aliases/guards in both wave
sizes, literal NaN/IEEE and denormal witnesses, unsupported reporting, and host
rounding/FTZ/DAZ controls. All CPU selectors are exercised. Loose tolerances
apply only to written finite numerical halves; the other half, inactive lanes,
guards and NaNs are compared bitwise. Existing RDNA4 fixtures retain their
expected results and use the explicit architecture names.

A separate C++20 harness invokes the actual upstream FP16 model and output
helpers, without environment guards, over both architectures, all half encodings,
eight guest policies, four OMOD settings, varied ABS/NEG/CLAMP/FP16_OVFL, both
operations and every available CPU selector. All 33,554,432 lane comparisons
match. The harness and log are `trig16-upstream.cpp` and `.log` in
`../goc-build/coverage-work/`. Production code and committed tests remain C++17.

Seven alternating before/after samples pinned to CPU 2, using thread CPU time
and at least 100 ms per case, put existing RDNA4 loose SIMD calls between
-0.51% and +1.04% of baseline. Scalar calls are 3.21–5.24% slower with guest-policy
handling. The scalar loop remains shared rather than duplicating it for default
flags. The AVX2 object is byte-identical. Raw data and harness are
`trig16-performance.json` and `trig16-timing.cpp` in the coverage work folder.
Modified benchmark cases pass a non-null state pointer, checking the loose opt-out.

Validation: all 1,101 CTests pass with Clang 21 Release and GCC 15 Release;
27 targeted ASan/UBSan tests pass, covering the new architecture suite, existing
half-trig numerical and exception fixtures, instruction flags and native-state
contracts. All nine audit-tool tests pass. Scanning all 192 objects in each
compiler's rebuilt library finds no explicit FP-control writes or environment
setter references (`trig16-fp-environment-audit.json`). The reporting-test
exclusion now names only explicit RDNA4 FP16/FP32 variants, leaving RDNA3 APIs
covered by the unsupported-reporting tests. The review ledger records 154
contracts; native-state naming resolves all 962 signatures with zero mismatches
or unresolved names. This closes that naming queue, not the broader semantic
review: 655 in-scope mnemonics still lack a recorded contract review.

## Scalar integer arithmetic architecture contracts

Reviewed all 29 canonical mnemonics implemented in
[`scalar_integer.cpp`](src/scalar_integer.cpp): 12 names common to both targets,
seven RDNA3 names renamed in RDNA4, and three RDNA4-only 64-bit operations.
The seven renamed pairs are ADD/SUB U32/I32, ADDC/SUBB U32, and ADDK I32 versus
RDNA4's explicit CO/CI spellings. RDNA4 XML lists the older spellings as aliases;
GoC's existing forwarding entry points are sufficient. Shared names need no
architecture suffix. This review does not cover scalar FP, comparisons, bitwise,
bitfield, pack or register-control families.

Evidence includes both XML instruction descriptions, operand widths and implicit
SCC reads/writes; RDNA3 ISA sections 16.1–16.3; rocjitsu revision `ffc144c564c`
per-architecture SOP1/SOP2/SOPK handlers and their shared implementations; and
`simd_glue.h` signed-overflow helpers. Both architecture handlers delegate the
12 common names to the same scalar helpers, with no SIMD macro or FP-policy
indirection. The seven renamed helper bodies are identical. The three 64-bit
NC/multiply forms exist only on RDNA4 and neither read nor write SCC.
Extracted upstream bodies, hashes and delegation locations are recorded in
`scalar-integer-upstream-review.json` in `../goc-build/coverage-work/`.

The semantic review confirms:

- Unsigned add/subtract report carry/borrow; signed forms report signed overflow.
  ADDC/SUBB read the old SCC bit and write the new carry/borrow. The subtrahend
  plus borrow is widened before comparison, including `0xffffffff + 1`.
- ABS and ABSDIFF report nonzero. The RDNA3 manual explicitly wraps subtraction
  before ABS, including `0x80000000 - 1 -> 0x7fffffff`. Rocjitsu's widened
  absolute-difference helper is incorrect at this boundary; GoC already follows
  the manual and its retained GFX1201 fixtures. No new hardware model is inferred.
- MIN/MAX SCC uses a strict comparison, so ties clear SCC. Low/high products and
  sign extension do not write SCC. Signed high multiplication uses signed
  32-bit inputs and extracts the high half of their 64-bit product.
- ADDK/MULK sign-extend all 16 immediate bits. ADDK reports signed overflow;
  MULK wraps its 32-bit product without changing SCC. SEXT ignores higher input
  bits. All result widths, including the three 64-bit forms, wrap as specified.
- These are scalar operations independent of EXEC and wave width. Source values
  are passed by value; read/modify/write immediate destinations are read before
  stores. The documented GoC overlap rule writes SCC last when D and SCC share
  storage. There are no instruction modifier bits or optional exception outputs.
  Existing flag validation rejects unsupported flags before operand access.
- Loose and empirical-exact results agree. The implementation is integer-only,
  has no host-FP environment access, and intentionally uses scalar code for the
  single result per call. There is no vector dispatch to preserve here.

Existing tests independently check 983,040 GFX1201 result/SCC triples, every
signed immediate, sign-extension captures, output aliases, errors and host FP
state. RDNA3 spelling tests compare arithmetic/overflow and all immediates with
wider arithmetic. Added tests compose low/high operations into 64-bit carry and
borrow chains over 32,768 boundary/random input pairs per architecture, CPU
selector and semantics, including input/output SCC reuse and ignored upper SCC
bits. Manual ABS/ABSDIFF witnesses and MIN/MAX tie SCC are now explicit tests.
No production arithmetic changes were needed; the scalar integer object remains
byte-identical with Clang Release. These tests supplement existing captures;
they do not claim new RDNA3 hardware measurements.

Validation: all 1,103 CTests pass with Clang 21 Release and GCC 15 Release;
all 14 scalar integer/spelling tests pass under ASan/UBSan. The scalar integer
production object is byte-identical before/after under both release compilers,
so this review adds no runtime or binary-code cost. All nine audit-tool tests
pass. Logs use `scalar-integer-review-` in the Clang coverage work folder and
GCC/sanitizer build folders. The ledger now records 183 contracts; 626 in-scope
mnemonics remain without recorded reviews. Native-state naming is unchanged.

## FP32 ternary minmax architecture contracts

Reviewed the fourteen canonical FP32 three-input selection mnemonics implemented
in [`minmax.cpp`](src/minmax.cpp): five RDNA3 MIN3/MAX3/MINMAX/MAXMIN/MED3
names and nine RDNA4 NUM/propagating names. RDNA4 lists the older five names as
assembly aliases for its NUM instructions, but callers must select the canonical
RDNA4 GoC APIs: RDNA3 IEEE/OMOD policies differ. These already-distinct canonical
names need no new `_rdna4` suffix. Binary, FP16/FP64 and packed forms are outside
this review increment.

The review checked both XML sets, per-architecture VOP3 ordinary and DPP
handlers, shared execution bodies, `try_execute_ternary_vop3_fp_simd`,
`apply_vop3_dst_mod_f32`, `fp_mode::effective_omod` / `apply_omod_f32`,
`util::ieee_minimum_simd` / `ieee_maximum_simd`, and RDNA3 manual sections
7.2.3.1, 16.7 and 16.12. Upstream revision is `ffc144c564c`; per-mnemonic
architecture/alias/helper hashes are retained in `minmax3-upstream-review.json`
in `../goc-build/coverage-work/`. Shared helper names alone were not treated as
proof of compatible behavior.

RDNA3 MIN/MAX selection follows the manual: IEEE mode gives signaling NaNs
priority and quiets them; otherwise a lone NaN loses to a number and two NaNs
select A unchanged. Each step of a ternary operation applies that rule.
MIN3/MAX3 repeat the selection; MINMAX/MAXMIN reverse it at the second step.
MED3 uses MIN3 when any input is NaN, otherwise removes the first input
numerically equal to the maximum and selects the maximum of the other two.
The latter distinction matters for signed-zero ties. RDNA4 NUM variants prefer
numbers; propagating variants retain GoC's existing signaling-NaN priority.
All selection comparisons order -0 below +0.

Input flushing precedes selection, and selected output denormals are retained
without OMOD: the output-denormal mode controls RDNA3 OMOD eligibility, not a
separate post-selection flush. This is the VALU rule in RDNA3 section 16.7;
the memory-atomic comparison-only flushing rule in chapter 13 does not apply.
RDNA3 enables OMOD only when output flushing is enabled and IEEE is clear.
RDNA4 enables it independently. OMOD and DX10-style CLAMP apply after the final
selection. DPP routes A before source modifiers, with four independent Wave64
rows. Full-register aliases and inactive lanes retain the existing contract.
The scalar and eight-lane AVX2 kernels implement all policies without changing
host FP controls or attempting sparse-mask optimization.

These APIs remain loose-only. Upstream ordinary MIN/MAX use host `fmin`/`fmax`,
its MED3 expression is not a captured NaN/zero-tie model, and its propagating
helpers return a canonical quiet NaN rather than GoC's retained payload policy.
This does not supply an empirical bit-exact port for the complete instructions.
No strict-exact support or new hardware-derived model was introduced. Strict
exact requests fail explicitly; non-null non-loose optional `trapsts` or
`excp_flag_user` reporting fails before operand access. Loose SIMD calls leave
optional state unchanged.

The existing tests cover all 512 modifiers, special/random values, literal
NaN/zero/order witnesses, masks, destination aliases and Wave64 DPP. The new
combined test crosses every operation, all eight guest policies, all modifiers,
both wave widths and every CPU selector, varying DPP8/DPP16 routes, row masks,
full/partial EXEC and whole-register destination aliases. It compares against
the independent ordered-integer selection/sorting reference and verifies all
source words, inactive lanes, guards and unchanged optional state. No production
code changes were necessary.

Validation: all 1,104 CTests pass with Clang 21 Release and GCC 15 Release;
all 19 targeted minmax/spelling/DPP arithmetic tests pass under ASan/UBSan.
All nine audit-tool tests pass. Scalar and AVX2 minmax production objects are
byte-identical to baseline under both release compilers, so no throughput or
binary-code regression is introduced (`minmax3-production-objects.json`).
Logs use `minmax3-review-` in the Clang coverage work folder and GCC/sanitizer
build folders. The ledger records 197 contracts; 612 in-scope mnemonics still
lack recorded reviews. Exact selection and reporting are documented limitations,
not counted as newly implemented behavior.


## FP16 ternary minmax architecture contracts

Reviewed the fourteen canonical ternary FP16 selection instructions in
[`half_minmax.cpp`](src/half_minmax.cpp), the FP16 counterparts of the preceding
FP32 review. Five RDNA3 canonical names and nine RDNA4 NUM/propagating names
already distinguish the architecture policies; RDNA4 assembly aliases must not
be mistaken for identical semantics of the RDNA3 APIs.

The review examined per-architecture ordinary/DPP bodies at rocjitsu revision
`ffc144c564c`, true16 operand selection, effective OMOD helpers, scalar and AVX2
GoC selection/conversion paths, and RDNA3 manual MIN/MAX and MED3 rules.
The upstream generated bodies use host min/max and a floating median network;
propagating forms canonicalize NaNs. They do not supply a complete empirical
bit-exact model to port. These APIs remain loose-only, with strict exact and
non-null exact global-state requests explicitly unsupported. Loose paths leave
optional state unchanged. No host-environment guard was borrowed.

RDNA3 IEEE mode prioritizes and quiets signaling NaNs. Otherwise its all-NaN
selection retains A. RDNA4 NUM selection prefers numbers; GoC's propagating
selection retains its existing signaling-NaN priority. Input flushing precedes
selection. Output-denormal mode only gates RDNA3 OMOD here, not a separate
post-selection flush. RDNA3 OMOD requires output flushing with IEEE clear;
RDNA4 OMOD is independent. RDNA3 median removes the first numerically maximal
operand, whereas the retained GFX1201 FP16 fixture uses ordered signed-zero
median selection. Half conversion preserves signaling bits where needed by
these policies, including in the F16C-based AVX2 path.

The reference now retains NaN payloads and models guest IEEE/input flushing/
OMOD eligibility. The new combined test crosses all fourteen operations, eight
guest policies, 512 low modifier combinations, both wave widths and available
CPU selectors. It varies half selectors, overflow saturation, DPP8/DPP16,
Wave64 row masks, EXEC and destination aliases, checking every source/destination
word and guard bitwise, including the untouched destination half. Existing
all-8,192-modifier and exhaustive-half tests remain. Payload comparisons test
consistency of the existing loose implementation; they do not establish new
hardware bit-exact support.

At this checkpoint binary and packed selection reviews remained open (resolved
in the following review). In particular, upstream
`fp_mode::packed_binary_f16` flushes selected output subnormals through its
generic finalization, whereas RDNA3 section 16.10 specifies per-half ordinary
MIN/MAX and section 16.7 makes their output-denormal mode irrelevant. GoC
retains the selected subnormal. This discrepancy needs separate review and is
not hidden by marking packed contracts reviewed here.

Validation: all 1,105 CTests pass under Clang 21 Release and GCC 15 Release;
all 18 targeted FP16 ternary/DPP tests pass under ASan/UBSan. All nine audit-tool
tests pass. Scalar and AVX2 production objects are byte-identical to the
pre-change audit builds under both compilers (`half-minmax-production-objects.json`
in `../goc-build/coverage-work/`). Logs are `half-minmax-review-{build,tests}.log`
in each build directory. The ledger now records 211 contracts, with 598 in-scope
mnemonics still requiring recorded reviews; these counts are not a completion
percentage or a claim of complete bit-exact support.


## FP16 binary and packed selection architecture contracts

Reviewed six ordinary and six packed FP16 MIN/MAX canonical names: the RDNA3
legacy pair and the RDNA4 NUM/propagating quartet in each form. Existing canonical
names already distinguish their policies, so no additional architecture suffixes
are needed. Review evidence includes both ISA XML sets, per-architecture
VOP2/VOP3/VOP3P bodies and DPP delegation, rocjitsu revision `ffc144c564c`'s
`packed_binary_f16`, and GoC's shared scalar/AVX2 half conversion and selection.
Per-handler locations and hashes are in `half-minmax-upstream-review.json` in
`../goc-build/coverage-work/`.

The preceding packed-output-denormal discrepancy is resolved for GoC's contract.
RDNA3 sections 16.7 and 16.10 define packed selection through ordinary MIN/MAX,
whose output-denormal control does not flush selected subnormals.
The [RDNA4 ISA reference](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content),
sections 16.7 (page 301), 16.10 (pages 395–396), and 16.12 (pages 516–517),
specifies the same composition for NUM and propagating forms. Thus GoC retains
its current input-flush-only selection behavior. The ordinary RDNA3 OMOD gate
still depends on output-flush mode and IEEE; packed forms have no OMOD.

Upstream's generic packed helper applies output flushing after selection, in
conflict with that documented composition, and installs a host FP-environment
guard. Neither behavior is copied. The upstream policy tests inspected for
packed output flushing cover ADD/MUL, not a selection output-flush witness;
three-input packed tests cover CDNA rather than RDNA. This review did not run
hardware or invent a new empirical model. The guard prohibition takes priority
over borrowing upstream code. The ordinary generated host min/max expressions
and propagating canonical-NaN helpers likewise do not constitute a full empirical
exact port. All twelve APIs remain loose-only; strict exact and non-null exact
optional reporting fail explicitly. Loose scalar/SIMD paths skip optional state.

Existing tests cover literal payload/signaling priority for every guest policy,
ordinary selectors/OMOD/DPP, every half encoding, all ordinary/packed modifiers,
EXEC and aliases in both wave widths. The new packed test crosses six operations,
eight guest modes, all 512 modifier combinations, both widths and every available
CPU selector. It varies full/partial EXEC and all three destination aliases,
checks both result halves bitwise against a raw-integer selection reference, and
checks untouched source words, guard words and unchanged optional state. The
older packed numeric reference now limits output flushing to ADD/MUL; its prior
unconditional flush had not been exercised with guest flags for selection.
No production code or SIMD fallback policy changes were needed.

Validation: all 1,106 CTests pass with Clang 21 Release and GCC 15 Release;
all 23 targeted ordinary/packed FP16 binary tests pass under ASan/UBSan.
All nine audit-tool tests pass. Scalar and AVX2 binary objects are byte-identical
to the pre-change audit builds under both compilers; hashes are in
`packed-selection-production-objects.json` in `../goc-build/coverage-work/`.
Build/test logs use `packed-selection-` in each build directory. The ledger
records 223 contracts, leaving 586 in-scope mnemonics without recorded reviews.


## Scalar EXEC Boolean architecture contracts

Reviewed all 24 B32/B64 SAVEEXEC/WREXEC instructions in
[`scalar_exec.cpp`](src/scalar_exec.cpp). Both RDNA3 and RDNA4 per-architecture
SOP1 handlers delegate to the same shared bodies at rocjitsu revision
`ffc144c564c`; both XML sets contain the canonical names. The review inspected
every Boolean expression, old/new destination selection, complement width, EXEC
replacement and SCC update, plus `Wavefront::exec`, `read_exec`, `set_exec`,
`write_exec` and `write_scc`. RDNA3 manual section 16.3 agrees with the Boolean
rules. Per-instruction helper hashes, handler lines, XML descriptions and
encodings are recorded in `scalar-exec-upstream-review.json` in
`../goc-build/coverage-work/`.

SAVEEXEC returns the old EXEC value in D; WREXEC returns the computed replacement.
NOT0 complements the explicit source, NOT1 the old EXEC value. SCC reports
whether the computed replacement is nonzero even when the optional EXEC output
is omitted. Calls execute with incoming EXEC zero. B32 APIs operate on supplied
32-bit words; B64 APIs operate on raw 64-bit values, including high bits that may
serve as scalar storage in Wave32. Integration with a wave's architectural
register pair remains the caller's job: these value APIs do not own a wavefront
or apply the emulator's register-write decoding rules.

The public output order is D, optional EXEC, then required SCC. Existing truth-table
tests check all twelve operations in each width, both semantics, output aliasing,
null optional EXEC and rejection of every instruction flag bit before pointer
access. The new two tests compose nested branch masks through AND_SAVEEXEC,
AND_NOT1_WREXEC, OR_SAVEEXEC and XOR_SAVEEXEC. They check each lane's membership
in inner-then, inner-else and outer-else paths, each saved destination, restoration
of original EXEC, and SCC, for deterministic random masks and zero/full masks.
They run both semantics with STRICT at every available CPU level in both widths.

These are exact integer operations shared across architectures; no FP model,
host FP mutation, SIMD requirement or new architecture suffix is involved.
No production changes were necessary. EXEC reporting is implemented, so an exact
non-null request does not need an unsupported-state fallback. The inexpensive
loose-mode update remains permitted by the optional-state contract.

Validation: all 1,108 CTests pass under Clang 21 Release and GCC 15 Release;
all four scalar EXEC tests pass under ASan/UBSan. All nine audit-tool tests pass.
The scalar EXEC production object is byte-identical to the pre-change audit
build under both release compilers (`scalar-exec-production-objects.json`).
Logs use `scalar-exec-review-` in each build directory. The ledger records 247
contracts; 562 in-scope mnemonics still need recorded reviews.


## Scalar integer comparison architecture contracts

Reviewed 30 integer/bit comparison mnemonics in
[`scalar_compare.cpp`](src/scalar_compare.cpp): eighteen SOPC register forms
shared by RDNA3/RDNA4 and twelve RDNA3-only SOPK immediate forms. Both XML
inventories and every applicable per-architecture handler were checked against
rocjitsu revision `ffc144c564c`'s shared bodies. The review followed SCC storage
and GoC's integer predicate helper as well as the entry-point wrappers.
Helper hashes, handler locations and XML descriptions are recorded in
`scalar-integer-compare-upstream-review.json` in `../goc-build/coverage-work/`.
Floating comparison predicates and their exception reporting are outside this
review increment, even though they share the translation unit and some tests.

Register comparisons provide six signed and six unsigned 32-bit predicates,
unsigned 64-bit equality/inequality, and zero/one bit tests at both widths.
Signed ordering in GoC flips the sign bit before unsigned comparison, avoiding
implementation-defined out-of-range unsigned-to-signed conversion. Bit indices
use the low five or six bits. Integer LG is equivalent to inequality; the float
ordered/unordered distinction does not apply. SOPK signed immediates are
sign-extended from sixteen bits; unsigned immediates are zero-extended. RDNA4
has no canonical SOPK counterparts, so no new architecture suffix is needed.

SCC is the required ordinary result, always zero or one. Inputs are passed by
value, allowing SCC storage to overlap source storage; byte stores also support
a word of uint64_t storage. There is no EXEC predicate and no additional implicit
state output. Instruction modifiers are invalid in all 64 bits. The common
validation accepts guest input/output denormal flags, which have no effect on
these integer operations. Both semantics give exact integer results. No host
FP environment change or SIMD path is involved.

Existing tests retain hardware corpus hashes for all register predicates,
signed/unsigned boundaries, all bit positions with modulo counts, 64-bit high
words, source/SCC aliases and host-environment preservation. The exhaustive
65,536-immediate test now includes immediate-minus-one and immediate-plus-one
on both sides of signed extension, in addition to zero, integer limits and
equality witnesses. Its oracle widens to int64_t instead of using GoC's sign-bit
ordering. Invalid instruction-flag tests now exercise all 64 bits rather than
only the low word. No production changes were necessary.

Validation: all 1,108 CTests pass with Clang 21 Release and GCC 15 Release;
all ten scalar comparison/spelling tests pass under ASan/UBSan. All nine audit-tool
tests pass. The scalar comparison object is byte-identical to the pre-change
audit builds under both release compilers (`scalar-compare-production-objects.json`).
Logs use `scalar-compare-review-` in each build directory. The ledger records
277 contracts, with 532 in-scope mnemonics still awaiting recorded review.


## Scalar field and count architecture contracts

Reviewed all twenty instructions in [`scalar_field.cpp`](src/scalar_field.cpp):
four BFE, two BFM, four population counts, six leading/trailing/sign counts, and
four in-place BITSET forms. Both architecture XML sets and all per-architecture
SOP1/SOP2 handlers map to the same rocjitsu shared helpers at revision
`ffc144c564c`. Review followed the BFM helper in `shared/simd_glue.h`, the count
and mask expressions, and GoC's `bits.h` primitives. Canonical CTZ/CLZ/CLS names
already cover both architectures; FF1/FLBIT are assembly aliases, not distinct
semantics. Hashes, handler lines, descriptions and aliases are retained in
`scalar-field-upstream-review.json` in `../goc-build/coverage-work/`.

BFE uses a five/six-bit offset and seven-bit width, clips to the source word,
returns zero for zero width, and sign-extends the clipped field for signed forms.
Unused descriptor bits are ignored. BFM masks both width and offset to five/six
bits, so a width equal to the word size produces zero, unlike BFE's full-width
extraction. Overflowing mask bits are discarded. BFE and population counts
write SCC from nonzero results; BFM, scans and BITSET do not write SCC. BITSET
reads its ordinary destination before modifying the selected bit. Count scans
return UINT32_MAX on zero; sign counts complement negative inputs first and
also return that sentinel for all ones. These are the upstream integer models,
not newly inferred numerical or exception models.

Both semantics are exact, with scalar implementations at every CPU selector.
There is no EXEC predicate or optional implicit output. Required D/SCC output
aliasing follows the documented last-write order, and invalid flags fail before
output access. The count helpers guard zero before compiler CLZ/CTZ builtins;
full-width extraction avoids a shift by the word size. No host FP control is
read or written by these implementations.

Existing tests retain captured result/SCC hashes, all field widths/offsets,
bit positions and sentinel cases, aliases and host-state preservation. The BFE
test now also sets every ignored descriptor bit, checking the same result and
SCC in exact semantics. A new test composes BFM, unsigned BFE and population
count across all low-seven-bit width/offset encodings in both word sizes and
both semantics, with high garbage bits on BFM operands. It checks clipping,
zero-width behavior, expected population and SCC preservation/update. No
production changes were required.

Validation: all 1,109 CTests pass with Clang 21 Release and GCC 15 Release;
all seven scalar field tests pass under ASan/UBSan. All nine audit-tool tests
pass. The production scalar field object is byte-identical to the pre-change
audit builds under both compilers (`scalar-field-production-objects.json`).
Logs use `scalar-field-review-` in each build directory. The ledger records
297 contracts; 512 in-scope mnemonics still need recorded reviews.


## Scalar Boolean and shift architecture contracts

Reviewed all thirty instructions in [`scalar_bits.cpp`](src/scalar_bits.cpp):
eighteen binary/unary Boolean operations, two bit reversals, six ordinary shifts,
and four fixed-shift-add forms. Both ISA XML sets and every applicable SOP1/SOP2
handler were checked against rocjitsu revision `ffc144c564c`; all delegate to the
same shared bodies. Review followed GoC's bit-reversal helper as well as result
and SCC writes. Helper bodies/hashes, handler locations, descriptions and aliases
are retained in `scalar-bits-upstream-review.json` in
`../goc-build/coverage-work/`. No architecture split is necessary.

Boolean operations and ordinary shifts write SCC from nonzero results; reversal
does not write SCC. NOT1 complements the second source. Shift counts are modulo
32/64, and arithmetic right shifts propagate the original sign bit. GoC expresses
sign propagation with unsigned shifts and explicitly handles count zero, avoiding
implementation-defined signed shifts and undefined full-word shifts. Bit reversal
uses the same full-word permutation as upstream's bit-at-a-time model.

The LSHL1/2/3/4_ADD forms compute the shifted sum at wider precision, truncate D,
and report whether any high bits remain. This agrees with RDNA3 manual section
16.1 and upstream, rather than treating SCC as nonzero D or carry from addition
after a truncated shift. These are exact integer semantics in both modes, with
required SCC where specified, no EXEC predicate and no additional global output.
Flags are validated before stores. D is stored before SCC; caller-side output
aliases follow that public order. No host FP environment changes occur.

Existing tests cover captured result/SCC hashes, scalar output aliases, all low
count encodings, sign propagation, flag errors and FP-state preservation. The
ordinary shift test now sets all high count bits too, comparing against the
bit-by-bit reference. New literal shift-add cases include nonzero/no-overflow,
zero/overflow, shift-only overflow above bit 32, UINT32_MAX, and the boundary
between the largest nonoverflowing and first overflowing sums. They run at all
CPU selectors in both semantics with STRICT. No production change was needed.

Validation: all 1,110 CTests pass with Clang 21 Release and GCC 15 Release;
all five scalar Boolean/shift tests pass under ASan/UBSan. All nine audit-tool
tests pass. Production scalar Boolean/shift objects are byte-identical to the
pre-change audit builds under both compilers (`scalar-bits-production-objects.json`).
Logs use `scalar-bits-review-` in each build directory. The ledger records 327
contracts; 482 in-scope mnemonics still need recorded reviews.


## Scalar move and packing architecture contracts

Reviewed eighteen instructions in [`scalar_move.cpp`](src/scalar_move.cpp) and
[`scalar_pack.cpp`](src/scalar_pack.cpp): MOV/CMOV at both widths, MOVK/CMOVK,
GETPC, four halfword packs, BITREPLICATE, CSELECT at both widths, and
QUADMASK/WQM at both widths. Both ISA XML sets and per-architecture handlers
were checked at rocjitsu revision `ffc144c564c`. All seventeen non-GETPC handlers
delegate to the corresponding shared bodies. GETPC's architecture-specific
handlers both add instruction size to PC; the ordinary four-byte form has no
literal operand, consistent with GoC's PC+4 result. The review also inspected
SCC reads and quad/bit spreading. Bodies, hashes, handler lines and XML
descriptions are in `scalar-transfer-upstream-review.json` in
`../goc-build/coverage-work/`.

Conditional moves leave D untouched when SCC bit zero is clear; CSELECT chooses
A when set, B otherwise. Higher caller-supplied SCC bits are ignored. MOVK and
CMOVK sign-extend a sixteen-bit immediate. GETPC reads the supplied address by
value and returns the next address with unsigned wrap, without writing PC.
Packs place the selected A half low and B half high. BITREPLICATE duplicates
each input bit into adjacent pairs. QUADMASK compresses each nonempty four-bit
group into one low result bit; WQM fills that group. Only QUADMASK/WQM update
SCC, based on nonzero results. Other instructions have no SCC output.

These are exact integer/register operations in both semantics, with no EXEC
predicate or host FP changes. Shared canonical names suffice. Inputs passed by
value permit destination/source overlap; required D/SCC aliasing uses the public
last-write order. All instruction flags are invalid, including when a conditional
move would not write. The packing-family common validator permits guest denormal
flags, which have no effect on these bit operations. These value APIs do not
implement emulator register decoding or control flow.

Existing tests cover raw basis bits, all immediate encodings, every quad-presence
pattern, captured packed results/SCC, aliases, invalid flags, PC carry/wrap and
host-state preservation. Immediate conditional moves now test high SCC bits
set with both values of bit zero for every immediate. A new CSELECT test checks
each SCC bit with both low-bit states, raw signaling-NaN-shaped words, and
32-/64-bit destination/source aliases in both strict semantics. No production
changes were needed.

Validation: all 1,111 CTests pass in Clang and GCC Release builds, and all twelve
scalar move, packing and guest-mode tests pass under ASan/UBSan. All nine
audit-tool tests pass. Production move/packing objects are byte-identical to the
pre-change builds under both compilers (`scalar-transfer-production-objects.json`).
Logs use `scalar-transfer-review-` in each build directory. The ledger records
345 contracts; 464 in-scope mnemonics still need recorded reviews.

## DS permutation architecture contract review

Rechecked the three canonical register-only permutation instructions against
rocjitsu `ffc144c564c`, including both architectures' execution delegates and
all five DS/VDS shared helpers in `generated/shared/execute_shared.h`.
RDNA3 Wave64 explicitly selects 32-lane groups; RDNA4 uses the wave width.
This agrees with RDNA3 manual section 12.5.2 and the earlier RDNA4 review above.
The ordinary Wave32 APIs remain shared; only the changed Wave64 variants need
`_rdna4_wave64`. FI exists only on RDNA4 and does not need an architecture suffix.

The sixteen-bit offset is added with unsigned 32-bit wrap before byte-address
bits select a lane. Ordinary gather reads zero from inactive sources; FI reads
stored source values regardless of EXEC. Both preserve inactive destinations.
Scatter ignores inactive sources, zero-fills unwritten active destinations, and
selects the highest active source for collisions, as in rocjitsu. This remains
an explicit deterministic emulator choice: the hardware prose does not promise
a collision winner. No LDS memory or implicit global register is written.
Both semantics support these raw-bit operations. No instruction flags are accepted.

Existing tests cover random routing, sparse/full/empty EXEC, destination aliases,
address overflow, collisions and the half-wave boundary. New tests check every
gather source/destination pair with a single active destination, in both strict
semantics, including FI, offset wrap and destination/address aliasing. They also
check all 64 invalid instruction-flag bits for FI with zero/full EXEC, null input
pointers and preserved destinations. No implementation changes were required.

Validation: all 1,113 CTests pass under Clang and GCC Release; all seven DS
permutation tests pass under ASan/UBSan. All nine audit-tool tests pass. Logs
use `ds-permute-review-` in each build directory. XML membership and upstream
file hashes are saved in `../goc-build/coverage-work/ds-permute-upstream-review.json`.
The ledger now records 348 contracts, with 461 in-scope mnemonics still needing
recorded reviews. These counts do not imply complete encoding or numerical coverage.

## DS swizzle architecture contract review

Rechecked both rocjitsu architecture delegates and the DS/VDS shared swizzle
helpers at `ffc144c564c`, alongside RDNA3 manual section 16.15, pages 513–515,
and the earlier RDNA4 review and retained fixtures above. The upstream helpers
implement only bit-mask/quad selection and read inactive sources unconditionally.
They are not a complete model for this instruction: the manual explicitly
specifies invalid-source zeroing (`thread_valid` in every pseudocode branch),
rotate and FFT. GoC already implements those rules and retains the four encoding
classes, thirty-two-lane row boundaries in Wave64, alias-safe source reads and
masked destination stores. There is no implicit global-state output or LDS access.

RDNA3's conflicting masked-rotate examples remain a documented limitation:
GoC follows the common pseudocode with row-local wrap, not those examples.
Existing RDNA4 captures support that choice only for RDNA4. This review does not
establish an RDNA3 hardware difference or remove that uncertainty, so the APIs
remain shared. No new numerical model or hardware probe was introduced.

Existing tests cover all 65,536 offset encodings, both wave widths, aliases,
retained RDNA4 corpora and invalid flags. Added raw-bit tests for every basic-mode
broadcast source and single active destination in both strict semantics, with
in-place operands and guard words. These directly distinguish the required
inactive-source zero from rocjitsu's unconditional source reads. Production
code is unchanged.

Validation: all 1,114 CTests pass in Clang and GCC Release builds; all four
swizzle tests pass under ASan/UBSan, including the retained hardware corpus.
All nine audit-tool tests pass. Logs use `ds-swizzle-review-` in each build
directory. The ledger records 349 contracts; 460 remain unrecorded. The
RDNA3 rotate-example ambiguity above remains explicit in that reviewed contract.

## Scalar lane-transfer architecture contracts

Reviewed READFIRSTLANE, READLANE and WRITELANE against rocjitsu `ffc144c564c`:
both architectures' VOP1/VOP3 READFIRSTLANE bodies, VOP3 indexed transfers, and
`RegisterAccess::read_scalar_selected_lane` / `write_scalar_selected_lane`.
The RDNA3 manual sections 16.8 and 16.12 explicitly confirm the zero-EXEC
fallback, ignored selector high bits, untyped data and EXEC-independent indexed
transfers. The RDNA4 handlers use the same rules. These APIs remain shared.

READFIRSTLANE selects the lowest active lane, falling back to lane zero when
EXEC is empty; zero EXEC therefore does not allow null operands. READLANE and
WRITELANE use the low five or six selector bits for the dispatched wave width,
without EXEC inputs. WRITELANE preserves every other lane. Scalar outputs can
alias input storage: the selected raw value is consumed before the store.
All instruction modifiers are rejected before operand access. These are exact
raw-bit transfers in both semantics with no implicit global-state outputs;
SGPR/VGPR register decoding and allocation bounds remain the emulator's job.

Existing tests cover selector wrapping, both semantics and wave widths, raw NaN
words, guards, aliasing, empty EXEC and all invalid flag bits. Added a test for
every pair of active lanes, including pairs across the Wave64 half boundary,
with the scalar result aliased to the later lane. It verifies selection priority
and preservation of all other words under both strict semantics. No production
change was needed.

Validation: all 1,115 CTests pass under Clang and GCC Release; all four
lane-transfer tests pass under ASan/UBSan. All nine audit-tool tests pass.
Logs use `lane-transfer-review-` in each build directory. The ledger records
352 contracts; 457 in-scope mnemonics remain without a recorded review.

## Scalar relative-move architecture contract review

Rechecked all five scalar relative moves against both architectures' SOP1
handlers at rocjitsu `ffc144c564c`, RDNA3 manual sections 3.3.1 and 16.3,
and the earlier two-manual review and retained RDNA4 captures above.
The upstream implementations agree with each other but conflict with the
manuals: ordinary offsets are truncated to eight bits, B64 offsets are scaled
by two, and RELSD_2 extracts adjacent eight-bit fields. GoC deliberately keeps
full-width ordinary DWORD offsets and the specified ten-bit fields at bits
0 and 16. Shared API names remain appropriate; upstream agreement alone is
not the evidence for these index semantics.

SGPR/VCC and TTMP regions cannot be crossed by indexing. Invalid sources
redirect to SGPR0 (SGPR0/1 for B64); invalid destinations discard writes.
Explicit array counts bound host storage, including empty and partial fallback
storage. Wide index arithmetic prevents overflow from producing a valid index.
B64 retains the documented even-base/even-M0 precondition; this API does not
model invalid misaligned encodings. Caller-provided storage/counts must also
reflect TTMP access restrictions. These raw-bit operations do not depend on
EXEC or update implicit global state; M0 is an input value, not a mutated register.

Existing tests retain the RDNA4 capture, full-M0 and region/count boundaries,
B64 word order, aliases and validation before operand access. New tests sweep
all 1,024 values of each split field at VCC_HI and TTMP15 boundaries, with
reserved bits clear/set, a shared source/destination array, guards and both
strict semantics. No new capture or production change was needed.

Validation: all 1,116 CTests pass in Clang and GCC Release builds; all five
scalar-relative tests pass under ASan/UBSan. All nine audit-tool tests pass.
Logs use `scalar-relative-review-` in each build directory. The ledger records
357 contracts; 452 in-scope mnemonics still need recorded reviews.

## Vector move and relative-move architecture contracts

Reviewed MOV_B32 and all four vector relative moves against rocjitsu
`ffc144c564c`, both ISA manuals' sections 3.3.2.2 and 7.2.2.1, and the
instruction descriptions. VOP1/VOP3 ordinary and modifier execution bodies
for the four relative moves are identical across RDNA3/RDNA4 (hashes recorded
in `../goc-build/coverage-work/vector-relative-upstream-review.json`).
Both select registers before DPP. The ISA's offset limit remains 255, whereas
rocjitsu permits 1023; GoC retains the documented 255 bound. Invalid sources
redirect to VGPR0; invalid destinations suppress writes. Split M0 fields occupy
bits [9:0] and [25:16], but extracting ten bits does not waive the offset limit.
Widened addition prevents wrap, and caller counts additionally bound storage.

This review found and fixed missing ABS_A/NEG_A support in MOV_B32, inherited
by all relative forms. Both manuals explicitly permit these input modifiers:
see RDNA3 section 7.2.2.1 and [RDNA4 section 7.2.2.1, page 70](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).
ABS clears bit 31, then NEG toggles it, after DPP routing. No FP conversion,
NaN quieting or exception calculation is needed. The unmodified copy loop is
retained separately, selected by one uniform modifier check. Wave64 and relative
moves reuse this implementation; no new abstraction or architecture suffix is
needed. OMOD/CLAMP remain explicitly unsupported for these B32 APIs; this review
does not establish their behavior or claim complete encoding support.

Existing tests cover both wave widths, register bounds, aliases, masks and DPP.
New coverage combines all five operations, both semantics and widths, all four
sign-modifier combinations, seven DPP descriptors plus direct execution,
sparse/full/empty EXEC, in-place routing and guard words. Inputs include signed
zeros, subnormals, infinities and signaling/quiet NaNs with preserved payloads.
Error tests now distinguish the newly supported sign bits from invalid modifiers.

Validation: all 1,117 CTests pass under Clang and GCC Release; all thirteen
move/relative-move/instruction-flag tests pass under ASan/UBSan. All nine
audit-tool tests pass. Logs use `vector-relative-review-` in each build directory.
Clang disassembly confirms that the unmodified case branches around sign-bit
processing; no timing-based performance claim is made. Existing SIMD kernels
are unchanged. The ledger records 362 contracts; 447 remain unrecorded.

## Half moves and swap architecture contracts

Reviewed MOV_B16, SWAP_B16/B32, SWAPREL_B32 and PERMLANE64_B32 against
rocjitsu `ffc144c564c` (both VOP1 architecture handlers, VOP3 half-move paths,
shared PERMLANE64 and true16 source/destination helpers), RDNA3 manual
sections 3.3.2.2 and 16.8, and the prior two-architecture review/fixtures above.
The RDNA4 half-move handler adds a SIMD shortcut but retains the same scalar
half-selection and preserved-half semantics. No architecture split is needed.

SWAP uses both original operands and preserves inactive lanes. Half swaps
preserve unselected halves, including when the two selected halves share one
VGPR. These operations reject arithmetic modifiers. SWAPREL extracts ten-bit
source/destination fields from M0 bits 0 and 16, but offsets above 255 remain
invalid even with larger caller-provided arrays. Either invalid operand cancels
both writes; wide index sums prevent wrapping. Rocjitsu checks allocation bounds
but does not enforce that smaller offset limit; GoC follows the documented rule.

PERMLANE64 is a Wave32 no-op with no operand access. Wave64 selects lane XOR 32
from the original source, including inactive source lanes, and masks destination
writes. MOV_B16 selects a half, applies ABS then NEG to its sign bit, and preserves
the other destination half. Its ignored OMOD/CLAMP behavior retains the existing
RDNA4 fixture evidence; no new hardware evidence for RDNA3 is claimed. Upstream
half-move handlers omit these sign modifiers, so they are not used as an oracle
for that behavior. All five are raw-bit contracts with no implicit global-state
outputs and are supported in both semantics.

Existing tests cover masks, aliases, half choices, DPP for MOV, index overflow,
invalid flags and retained RDNA4 captures. Exhaustive half-encoding/sign tests
now run in both strict semantics. A new SWAPREL test supplies 300 registers and
independently puts each offset at 255, 256 and 299, distinguishing the ISA limit
from the allocation bound while checking both writes and all untouched lanes.
No production change was required.

Validation: all 1,118 CTests pass with Clang and GCC Release; all sixteen
move/swap tests pass under ASan/UBSan. All nine audit-tool tests pass. Logs use
`swap-review-` in each build directory. The ledger records 367 contracts;
442 in-scope mnemonics remain without recorded reviews.

## PERMLANE16 architecture and SIMD contracts

Reviewed the four PERMLANE16/PERMLANEX16 fixed and variable selector contracts
against rocjitsu `ffc144c564c` VOP3 execution bodies, RDNA3 section 16.12 and
both XML inventories. Fixed-selector handlers agree between RDNA3/RDNA4 apart
from the spelling of the OPSEL field; variable-selector instructions exist only
on RDNA4. None requires an architecture suffix. Selectors choose within a
sixteen-lane row; X switches to the other row of the same thirty-two-lane group,
including in Wave64. Fixed selectors use sixteen packed nibbles from two scalar
words; VAR uses the low four bits of the destination lane's index word.

FI permits reading inactive sources. Without FI, an inactive source either
preserves D or supplies zero according to BOUND_CTRL. Inactive destination lanes
are always preserved. There are no arithmetic modifiers or implicit global-state
outputs. All source/index reads precede writes, including when D aliases A or
the VAR index register. Wave64 halves are independent, so the existing split
implementation is safe for whole-register aliases.

Reviewed the scalar loop and both SIMD implementations. AVX2 processes eight
lanes using two source-half permutations and selection; AVX-512 processes one
sixteen-lane row with a variable permutation. Both handle all FI/BOUND_CTRL
combinations without falling back to scalar, stage results before stores, and
use masked stores. No sparse-mask specialization or performance change was added.

Existing tests retain the RDNA4 raw corpus, random selectors, all alias patterns,
both semantics, upper Wave64 masks and validation. A new test checks every
selector at every destination in both widths, all four FI/BOUND_CTRL settings,
both semantics and every available CPU level. Only the destination is active;
VAR indices have all high bits set and alias the output. It independently checks
inactive-source fallback, ignored index bits, row pairing and guard preservation.
No production change was needed.

Validation: all 1,119 CTests pass under Clang and GCC Release; all six PERMLANE
tests pass under ASan/UBSan. All nine audit-tool tests pass. Logs use
`permlane-review-` in each build directory. The ledger records 371 contracts;
438 in-scope mnemonics still need recorded reviews.

## Vector integer comparison architecture contracts

Reviewed all 72 relational integer CMP/CMPX mnemonics: six predicates, signed
and unsigned 16/32/64-bit inputs, and both output forms. Checked 288 ordinary
VOPC/VOP3 architecture handlers at rocjitsu `ffc144c564c`, resolving delegated
shared helpers and verifying each predicate's operand type, comparison operator
and result destination. Handler hashes and checks are saved in
`../goc-build/coverage-work/integer-compare-upstream-review.json`. Representative
true16, CMPX and DPP paths were inspected directly, together with RDNA3 section
16.9. No differing RDNA3/RDNA4 value semantics were found; the APIs stay shared.

CMP's required output is the scalar condition mask; CMPX's is replacement EXEC,
not an additional VCC write. Both clear inactive result bits and return zero
for empty EXEC. The mask output remains required in both semantics; it is not
an optional global-state report. Integer bit patterns do not generate floating
exception outputs. Sixteen-bit forms select each source half independently;
64-bit inputs combine low/high VGPR words. Signed ordering is two's-complement.
DPP8/DPP16 routes A before half selection for 16/32-bit forms, leaving B in its
lane; filtered output bits are zero. DPP on 64-bit forms remains unsupported.

Reviewed the scalar predicate construction and SIMD implementations: AVX2 uses
eight-lane comparisons and sign-bit biasing, AVX-512 sixteen-lane unsigned
comparisons with the appropriate bias. The 64-bit paths compare high words
first and use unsigned low words to break ties. Both compute full masks before
applying EXEC. Wave64 computes both halves before publishing an aliasable scalar
output. Existing SIMD dispatch and performance are unchanged.

Existing tests cover independent boundary/random values, retained RDNA4 captures,
all predicates and widths, half selectors, DPP, masks, output/source aliases and
validation. The added chained CMP/CMPX test updates the caller's EXEC variable
in place, then compares reversed operands with reversed half selectors. It checks
that the second result cannot reactivate filtered lanes, across all types and
predicates, both wave sizes and semantics, and all available CPU levels.
No production change was required. Constant F/T comparisons are a separate
family and are not included in these 72 review records.

Validation: all 1,120 CTests pass under Clang and GCC Release; all eleven integer
comparison tests pass under ASan/UBSan. All nine audit-tool tests pass. Logs use
`vector-integer-compare-review-` in each build directory. The ledger records
443 contracts; 366 in-scope mnemonics still need recorded reviews.

## Constant comparison architecture contracts

Reviewed all 28 F/T CMP/CMPX forms: floating 16/32/64-bit inputs and signed/
unsigned 32/64-bit inputs. Both XML inventories confirm these canonical forms
are RDNA3-only. Checked 56 RDNA3 VOPC/VOP3 bodies at rocjitsu `ffc144c564c`,
resolving shared helpers and confirming the constant predicate; hashes/checks
are in `../goc-build/coverage-work/constant-compare-upstream-review.json`.
No RDNA4 suffix is needed because no same-mnemonic conflict exists.

F produces zero; T produces participating EXEC after DPP16 row/bank/boundary
filtering. CMP's required destination is a condition mask and CMPX's is the
replacement EXEC. Operand values are irrelevant, so null operand pointers are
allowed even for nonempty EXEC; the mask destination is always required.
DPP is supported for 16/32-bit forms and rejected for 64-bit forms. Constant
floating results support sign/half/CLAMP input flags without depending on them.

Guest IEEE mode was unnecessarily rejected for the floating forms. It is now
accepted alongside the guest denormal flags, without changing the constant
result. The optional RDNA3 register remains TRAPSTS: loose semantics leave it
unchanged; exact semantics with a non-null pointer still return
`GOC_ERROR_UNSUPPORTED_GLOBAL_STATE` before flag checks or operand/output access,
even for empty EXEC and non-strict requests. This review does not supply a new
exception model. Integer forms have no floating-state output. The private helper
now also names that pointer `trapsts`, matching its public callers.

Existing tests cover all predicates, both wave sizes/semantics, DPP filtering,
modifiers, null inputs and output preservation. Floating result/DPP and loose
state tests now include guest IEEE mode. Added rejection-order tests for every
floating form with empty EXEC, invalid flags, and both strict settings, verifying
that result and state outputs remain unchanged.

Validation: all 1,121 CTests pass under Clang and GCC Release; all seven constant
comparison tests pass under ASan/UBSan. All nine audit-tool tests pass. Logs use
`constant-compare-review-` in each build directory. The public contract now also
names TRAPSTS correctly and documents IEEE-mode acceptance. The ledger records
471 contracts; 338 in-scope mnemonics remain without recorded reviews.

## Classification architecture contracts

Reviewed the six FP16/FP32/FP64 CLASS CMP/CMPX mnemonics against both architectures
at rocjitsu `ffc144c564c`. The 24 VOPC/VOP3 handler locations, hashes and delegates
are recorded in `../goc-build/coverage-work/class-upstream-review.json`. CMP
handlers are identical between architectures; all six CMPX scalar bodies are
identical after the RDNA4 SIMD shortcut. Shared helpers classify raw fields,
including separate signaling/quiet NaNs and signed subnormals. No architecture
suffix is needed. CMP writes its required condition mask; CMPX writes replacement
EXEC without an additional VCC output.

Fixed rejection of guest IEEE mode and the instruction CLAMP flag. Both are
accepted without changing classification. The RDNA3 manual section 7.2.3.1 and
[RDNA4 section 7.2.3.1](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)
explicitly list CLASS among instructions that ignore CLAMP. Upstream classification
also ignores IEEE and denormal controls. Sign modifiers operate on raw bits;
FP16 half selection applies to both A and B. DPP routes A before classification,
with B local to the destination lane. FP64 DPP remains rejected. Scalar, v3
(eight lanes) and v4 (sixteen lanes) kernels are unchanged; accepted flags do not
add arithmetic or displace SIMD paths.

The existing no-exception-output API remains unchanged, matching rocjitsu's CLASS
handlers. The RDNA3 per-instruction pages contain a generic sNaN/qNaN invalid-note,
including a CLAMP condition inconsistent with the explicit ignored-CLAMP table.
This review does not resolve that documentation discrepancy or claim a newly
validated hardware exception model. Existing hardware fixtures cover numerical
classification, not a new RDNA3 exception experiment.

Added literal representatives of all ten classes against every selector bit,
both wave sizes, both semantics, all available CPU levels, all eight exposed
IEEE/denormal policy combinations and CLAMP on/off. Existing DPP/modifier/alias
tests now also exercise IEEE mode and ignored CLAMP. Existing exhaustive FP16,
class-mask, hardware-corpus and validation tests remain in place.

Validation: all 1,122 CTests pass under Clang and GCC Release; all ten CLASS
tests pass under ASan/UBSan. All nine audit-tool tests pass. Logs use
`class-review-` in each build directory. The ledger now records 477 contracts;
332 in-scope mnemonics still need recorded reviews. These counts do not resolve
the documented semantic limitations.

## Non-carry integer addition architecture contracts

Reviewed ADD3 and signed/unsigned 32-bit non-carry ADD/SUB plus unsigned SUBREV
at rocjitsu `ffc144c564c`. All 18 pairs of architecture-specific execution and
modifier bodies are identical: three VOP2 and six VOP3 instructions, each with
a modifier body. Locations, hashes and shared-helper names are recorded in
`../goc-build/coverage-work/integer-add-upstream-review.json`. Shared helpers
and `vop3_integer_add`/`vop3_integer_sub` implement wrapping or widened signed/
unsigned saturation without architecture queries. The canonical ADD3 name
also covers the RDNA4 `v_add3_nc_u32` spelling. No suffix is needed.

Two-input CLAMP saturates after addition/subtraction, including reversed
operand order for SUBREV. ADD3 always wraps modulo 2^32. Fixed GoC's rejection
of ADD3 CLAMP: the RDNA3/RDNA4 ignored-CLAMP tables explicitly include ADD3,
and rocjitsu's helper ignores it. The scalar path now explicitly excludes ADD3
from saturation, matching its existing v4 kernel. No carry, SCC or floating
exception output is produced. Other low instruction flags remain rejected.

Existing DPP routing is preserved, including GoC's gfx1201 fixture showing
SUBREV routes B before subtracting lane-local A. Upstream modifier staging
routes encoded SRC0; this review does not supply a new RDNA3 capture to resolve
the distinction between encoded-source staging and GoC's existing operand
contract. DPP8/DPP16 remain supported, with Wave64 row masks mapped to each half.
Whole-VGPR aliases and inactive destination lanes are preserved.

Expanded existing Cartesian-boundary, random-input, alias and literal-overflow
tests to cover ADD3 CLAMP. Expanded Wave64 tests also exercise it with both
semantics, all CPU levels, all DPP descriptors and row masks. Literal cases
include three UINT32_MAX operands and cancellation across bit 31, so accidental
saturation is observable. Existing hardware hashes were retained unchanged;
new ignored-CLAMP combinations are specification/upstream tests, not captures.

Dispatch is unchanged: v3 uses eight-lane explicit SIMD for two-input saturation;
wrapping operations retain the previously selected baseline path. v4 uses
sixteen-lane SIMD for all six operations. The compile-time ADD3 exclusion adds
no per-lane dynamic branch or mask-density logic.

Validation: all 1,122 CTests pass under Clang and GCC Release; all twelve targeted
integer-add/DPP/Wave64 tests pass under ASan/UBSan. All nine audit-tool tests pass.
Logs use `integer-add-review-` in each build directory. The ledger records
483 contracts; 326 in-scope mnemonics remain without recorded reviews.

## Integer multiplication architecture contracts

Reviewed the seven 32-bit-result integer multiply forms against rocjitsu
`ffc144c564c`: full-width MUL_LO_U32/MUL_HI_U32/MUL_HI_I32, plus low/high
signed/unsigned 24-bit products. All 19 architecture-specific execution/modifier
body pairs are identical (38 bodies), including VOP2/VOP3 encodings where
available. Locations, hashes and delegates are recorded in
`../goc-build/coverage-work/integer-mul-upstream-review.json`. Shared execution
helpers and `vop3_integer_mul`/`vop3_integer_mad` were checked for widening,
24-bit truncation/sign extension, high-word selection and saturation. These
contracts require no architecture suffix and no optional floating-state output.

GoC's existing arithmetic matches those models: discard each 24-bit operand's
upper byte before interpreting bit 23 as its sign; multiply in a wide domain;
select bits 63:32 for high results. Signed high results retain sign extension
above the 48-bit product. Only low 24-bit products accept CLAMP, saturating to
the signed/unsigned 32-bit range. All four 24-bit forms accept DPP on A; the
full-width forms reject DPP. Unsupported low flags still reject before operand
access, including empty EXEC. Both wave sizes preserve whole-VGPR aliases and
inactive destinations.

The v3/v4 kernels compute eight/sixteen lanes respectively. High products use
separate widening multiplies for even and odd lanes, then combine high words.
Signed saturation checks whether the high word equals the low word's sign
extension; unsigned 24-bit saturation checks a nonzero high word. No kernel,
dispatch or production source change was needed.

Added tests sweeping all 256 upper-byte values in both operands across 64
pairs of low encodings, with different upper-byte sequences for A and B,
including bit-23 sign boundaries and saturation thresholds. These run both
wave sizes and semantics on all available CPU levels with masked outputs.
A second test reconstructs full 64-bit signed/unsigned products from the
full-width low/high APIs using single-bit factors and their neighbors, covering
all bit positions and both sides of SIMD lane-packing boundaries. Existing
boundary/random, alias, DPP hardware-corpus and Wave64 routing tests remain.

Validation: all 1,124 CTests pass under Clang and GCC Release; all thirteen
integer-multiply/DPP/Wave64 tests pass under ASan/UBSan. All nine audit-tool tests
pass. Logs use `integer-mul-review-` in each build directory. The ledger records
490 contracts; 319 in-scope mnemonics remain without recorded reviews.

## Integer selection architecture contracts

Reviewed all fourteen signed/unsigned 32-bit MIN/MAX, MIN3/MAX3, MINMAX/MAXMIN
and MED3 instructions at rocjitsu `ffc144c564c`. All 36 architecture-specific
execution/modifier pairs are identical (72 bodies), spanning four VOP2 and
fourteen VOP3 instructions. Locations, hashes and delegates are recorded in
`../goc-build/coverage-work/integer-minmax-upstream-review.json`. Checked all
18 shared execution helpers for operand order, signed interpretation and result
selection. No architecture suffix or optional exception-register output is needed.

MINMAX combines A/B with MIN then combines C with MAX; MAXMIN does the reverse.
MED3 selects the middle integer, including duplicate values. GoC uses biased
unsigned ordering in the scalar implementation and native signed/unsigned
min/max in eight-lane v3 and sixteen-lane v4 kernels. Unlike the upstream mixed
helpers' `fmin`/`fmax` spelling, GoC needs no floating conversion for selection.
All operations retain exact integer input bits.

Fixed unnecessary rejection of CLAMP. Rocjitsu's helpers do not use it; integer
selection already yields a value in the result type's range, so integer saturation
cannot change it. GoC now accepts CLAMP without adding work to scalar or SIMD
selection. Other low flags remain unsupported. DPP8/DPP16 route A before
selection, with lane-local B/C and per-half Wave64 row-mask handling. Empty EXEC
permits null operands after validation. Inactive destination lanes and whole-VGPR
aliases remain preserved.

Added CLAMP-on/off tests over all triples from six signed/unsigned boundary
encodings, all destination-source aliases, both semantics, both wave sizes and
all available CPU levels. Existing DPP alias/random tests now include CLAMP;
existing hardware-corpus hashes remain unchanged. Invalid-flag tests still
exercise every other low bit. This is a port of the upstream modifier contract,
not a new empirical hardware model. SIMD kernels and dispatch are unchanged.

Validation: all 1,125 CTests pass under Clang and GCC Release; all thirteen
integer-selection/DPP/Wave64 tests pass under ASan/UBSan. All nine audit-tool
tests pass. Logs use `integer-minmax-review-` in each build directory. The ledger
records 504 contracts; 305 in-scope mnemonics remain without recorded reviews.

## Narrow-factor integer MAD architecture contracts

Reviewed signed/unsigned MAD with 16/24-bit factors and a 32-bit addend/result
at rocjitsu `ffc144c564c`. All eight RDNA3/RDNA4 execution/modifier body pairs
are identical (16 bodies). Locations, hashes and delegates are recorded in
`../goc-build/coverage-work/integer-mad-upstream-review.json`. Shared execution
helpers, true16 selection and `vop3_integer_mad` agree with GoC: select/truncate
factors before signed interpretation, retain the full-width addend, and saturate
only after the wide product-plus-addend. No architecture suffix or floating
exception output is needed.

Both semantics produce the same exact integer result. CLAMP selects signed or
unsigned 32-bit saturation; otherwise the sum wraps. The 16-bit forms allow
HIGH_A/HIGH_B; the 24-bit forms discard the upper byte and reject those selectors.
C has no half selector. DPP8/DPP16 route A before half selection, and Wave64
remaps row masks for each half. Validation precedes operand access, including
empty EXEC; whole-VGPR aliases and inactive destination lanes are preserved.

The existing eight-lane v3 kernel remains selected on v3/v4 hosts. Wrapping
uses low-word multiply/add. Saturating execution widens even and odd products,
adds a sign- or zero-extended C, clamps the full sum, then interleaves low words.
No production changes were required. This review does not claim a separate
sixteen-lane v4 kernel.

Added literal saturation-boundary tests for all four instructions, with addends
within four units on each side of the limit. Signed 24-bit witnesses require
cancellation of a product already outside the destination range, detecting
premature product saturation. Signed 16-bit cases exercise both limits;
unsigned cases cross UINT32_MAX. Tests use mixed lanes, all half selectors,
CLAMP on/off, both semantics/wave sizes, all available CPU levels, and D aliasing
C. Existing exhaustive 16-bit-factor, discarded-byte, DPP hardware-corpus,
random, alias and Wave64-routing tests remain in place.

Validation: all 1,126 CTests pass under Clang and GCC Release; all thirteen
integer-MAD/DPP/Wave64 tests pass under ASan/UBSan. All nine audit-tool tests pass.
Logs use `integer-mad-review-` in each build directory. The ledger records
508 contracts; 301 in-scope mnemonics remain without recorded reviews.

## Wide integer MAD architecture contracts

Reviewed the RDNA3 MAD_I64_I32/MAD_U64_U32 and RDNA4 MAD_CO_I64_I32/
MAD_CO_U64_U32 spellings at rocjitsu `ffc144c564c`. Four architecture handlers
and four shared helpers are recorded in
`../goc-build/coverage-work/mad64-upstream-review.json`. The inventory treats
the renamed forms as architecture-exclusive canonical mnemonics; GoC shares
the arithmetic through its existing spelling wrappers, without an `_rdna4`
suffix because these are different mnemonic names.

This review preserves the previously documented signed-mask correction:
rocjitsu's signed helpers report overflow, whereas RDNA3 section 16.12 explicitly
concatenates the full 65-bit sum into scalar bit 64 and vector bits 63:0.
GoC's existing RDNA4 fixtures likewise distinguish the extended sign from an
overflow flag. The unsigned scalar bit is ordinary carry. CLAMP changes the
vector result on overflow but leaves that pre-clamp scalar bit intact. The
required scalar output is not optional exception reporting. No new numerical
model or hardware experiment was introduced by this review.

The existing scalar/v3/v4 paths all preserve that contract. SIMD widths are
four/eight 64-bit lanes, respectively; both vector words are staged before
stores, with D1 winning coincident D0/D1 destinations. Wave64 defers its scalar
store until both halves finish, permitting the scalar output to overlap source
or destination words. It writes zero for empty EXEC, while vector destinations
remain untouched. DPP and low flags other than CLAMP are rejected before access.

Added a Wave64 test combining coincident D0/D1, D aliasing C0, and an unaligned
scalar output straddling lanes 31/32 of that same register. Every single active
lane and its complement are covered across all four spellings, both semantics,
CLAMP settings and available CPU levels. It checks the complete aliased register,
including inactive lanes and final scalar-write precedence. Existing wide-reference,
RDNA4 hardware-corpus, saturation, cross-register alias and validation tests remain.
No production change was required.

Validation: all 1,127 CTests pass under Clang and GCC Release; all ten wide-MAD/
Wave64 tests pass under ASan/UBSan. All nine audit-tool tests pass. Logs use
`mad64-review-` in each build directory. The ledger records 512 contracts;
297 in-scope mnemonics remain without recorded reviews.

## Vector carry and borrow architecture contracts

Reviewed all six 32-bit ADD/SUB/SUBREV carry-output forms, with and without
carry/borrow input, at rocjitsu `ffc144c564c`. All 18 architecture-specific
execution/modifier body pairs are identical (36 bodies): three VOP2 carry-input
forms and six VOP3 forms. Locations, hashes and delegates are recorded in
`../goc-build/coverage-work/carry-upstream-review.json`. Shared helpers agree on
unsigned carry/borrow, subtracting the incoming borrow, reversed operand order,
and CLAMP after the full operation. No architecture suffix or optional
floating-state output is needed.

The required scalar result reports carry for addition or borrow for subtraction,
independently of CLAMP. Saturated ADD yields UINT32_MAX and saturated subtraction
yields zero. Input mask bits belong to their destination lanes; DPP does not
route them. GoC retains its fixture-backed SUBREV routing of B, with other forms
routing A, as already documented for non-carry arithmetic. This review adds no
RDNA3 hardware capture. Masked-off scalar result bits are zero; masked-off vector
lanes are preserved. Empty EXEC writes scalar zero without requiring VGPRs.
Wave64 defers the scalar write until both halves have read their inputs.

The eight-lane v3 and sixteen-lane v4 paths compute overflow/borrow separately
for the A/B operation and incoming bit, OR those conditions, and then apply
optional saturation. Scalar and SIMD paths already agree; no production change
was required and SIMD dispatch is unchanged.

Added an integration test chaining a no-input-carry low-word instruction into
its carry-input high-word counterpart. It reconstructs full unsigned 64-bit
ADD/SUB/SUBREV results and compares the final carry/borrow against independent
64-bit arithmetic. The test covers the Cartesian boundary values, both wave
sizes and semantics, all available CPU levels, full/mixed EXEC, an in-place
vector destination and reuse of the scalar carry variable. Existing tests cover
CLAMP, arbitrary input masks, DPP, scalar-output aliases, validation and retained
RDNA4 hardware fixtures.

Validation: all 1,128 CTests pass under Clang and GCC Release; all twelve carry
tests pass under ASan/UBSan. All nine audit-tool tests pass. Logs use
`carry-review-` in each build directory. The ledger records 518 contracts;
291 in-scope mnemonics remain without recorded reviews.

## Compound integer architecture contracts

Reviewed eight three-input integer operations at rocjitsu `ffc144c564c`:
LSHL_ADD/ADD_LSHL, LSHL_OR/AND_OR/OR3/XOR3, XAD and LERP. All sixteen
architecture-specific execution/modifier pairs are identical (32 bodies),
and all eight shared helpers were checked. Locations, hashes and delegates
are recorded in `../goc-build/coverage-work/integer-ternary-upstream-review.json`.
XAD also covers the RDNA4 XOR_ADD spelling. No architecture suffix or optional
exception-register output is needed.

Shift counts use the low five bits; arithmetic stages wrap to 32 bits. LSHL_ADD
shifts A before adding C; ADD_LSHL adds A/B before shifting by C. AND_OR and XAD
likewise combine A/B first. LERP averages each unsigned byte independently,
using only that byte's low C bit for rounding; no carry leaks between bytes.
GoC's implementations match those existing upstream models.

Fixed rejection of CLAMP, which all eight upstream helpers ignore. The RDNA3
XAD instruction note explicitly specifies no saturation, and the ignored-CLAMP
table includes bitwise operations and ADD_LSHL. This change ports upstream's
existing behavior; it is not based on new hardware experimentation. All other
low flags remain unsupported. DPP8/DPP16 route A before the operation, preserving
lane-local B/C; Wave64 remaps row-enable bits per half. Empty EXEC and validation
precedence remain unchanged.

The existing v3 kernel handles shifts and byte LERP in eight lanes; simple
Boolean/XAD operations retain the baseline path selected by prior benchmarks.
The v4 kernel handles all eight operations in sixteen lanes, using ternary-logic
instructions for Boolean results and native byte average plus correction for
LERP. Accepting CLAMP does not introduce any per-lane work or alter dispatch.

Expanded the existing Wave32 reference checks to run both CLAMP settings,
including every shift count/bit position, Boolean truth tables, every byte pair
and rounding control, and boundary/random values. Literal stage-order witnesses
remain in place.
Expanded Wave64 selection/ternary routing tests to include CLAMP for every DPP
route, alias and semantics combination. Existing hardware hashes remain unchanged;
the added modifier cases are upstream/specification checks.
The Wave64 test oracle now checks DPP flag bits explicitly instead of treating
any nonzero instruction flag as routing; CLAMP-only calls exposed that assumption.

Validation: all 1,128 CTests pass under Clang and GCC Release; all fourteen
compound-integer/DPP/Wave64 tests pass under ASan/UBSan. All nine audit-tool tests
pass. Logs use `integer-ternary-review-` in each build directory. The ledger records
526 contracts; 283 in-scope mnemonics remain without recorded reviews.

## Bitfield and alignment architecture contracts

Reviewed BFE_I32/U32, BFI, BFM, BFREV, ALIGNBIT/ALIGNBYTE and PERM at rocjitsu
`ffc144c564c`. All eighteen architecture-specific execution/modifier pairs are
identical (36 bodies), covering eight VOP3 forms and BFREV's VOP1 form. Nine
shared helpers were checked. Locations, hashes and delegates are recorded in
`../goc-build/coverage-work/bitfield-upstream-review.json`. No architecture suffix
or optional exception-register output is needed.

BFE/BFM mask offset/width to five bits; width zero yields zero. Signed BFE
sign-extends the source before extraction, so fields crossing bit 31 include
copies of the source sign, then sign-extends the field itself. ALIGN uses A as
the high word and B as the low word, with five-bit bit counts or two-bit byte
counts. PERM selectors 0..7 copy bytes, 8..11 replicate selected halfword sign
bits, 12 yields zero, and 13..255 yield all ones. BFI is bitwise selection and
BFREV reverses all 32 bits. Existing GoC scalar/SIMD code agrees with upstream.

Fixed unnecessary rejection of CLAMP, which upstream ignores. The ISA's
ignored-CLAMP table includes bitwise operations, ALIGN and PERM. Other low flags
remain invalid. All eight operations retain DPP8/DPP16 on A, with B/C local;
Wave64 maps row-enable bits to each half. Validation precedes null-operand-safe
empty EXEC, and aliases/inactive destinations retain their existing behavior.

The v3 paths use eight lanes except BFI, whose previously benchmark-selected
baseline path is preserved. v4 uses sixteen lanes for every operation. Signed
extraction uses arithmetic shifts; ALIGN combines opposite-direction shifts;
PERM and BFREV use byte lookup/shuffle operations. CLAMP acceptance changes only
validation, not any arithmetic kernel or SIMD dispatch.

Expanded existing per-bit reference checks to both CLAMP settings, including
all offset/width/source-bit combinations, concatenated-bit/shift combinations,
PERM selectors and boundary/random words. Expanded Wave64 DPP/alias tests to
both settings, checking DPP bits explicitly for CLAMP-only calls. Existing
hardware-corpus hashes remain unchanged. These additions validate the existing
upstream modifier contract without new hardware experiments.

Validation: all 1,128 CTests pass under Clang and GCC Release; all seventeen
bitfield/DPP/Wave64 tests pass under ASan/UBSan. All nine audit-tool tests pass.
Logs use `bitfield-review-` in each build directory. The ledger records
534 contracts; 275 in-scope mnemonics remain without recorded reviews.

## Boolean architecture contracts

Reviewed AND/OR/XOR/NOT B16/B32 and XNOR B32 at rocjitsu `ffc144c564c`.
The review covers sixty architecture-specific execution/modifier bodies and ten
shared B32 helpers. Twenty handler pairs are identical; the ten remaining pairs
have identical scalar behavior and differ by RDNA4 true16 SIMD shortcuts. Their
selector loads and partial-half stores agree with the scalar helpers. Locations,
hashes and shared bodies are recorded in
`../goc-build/coverage-work/boolean-upstream-review.json`.

These operations are bitwise, with no floating-point policy or exception-state
output. B16 selects the input and destination halves and preserves the other
destination half. NOT has no B input; XNOR complements XOR. The existing shared
GoC implementations match both architectures, so no architecture suffix is
needed. DPP routes A before half selection, retaining local B and old D;
Wave64 maps row enables to its two halves.

Fixed rejection of CLAMP, which the upstream scalar implementations ignore and
the RDNA3 ignored-CLAMP table explicitly lists for bitwise operations. Accepting
it does not change the scalar algorithm or sixteen-lane v4 kernel. The previously
benchmark-selected baseline path remains in use below v4. No new abstraction or
SIMD fallback was introduced. Other arithmetic modifiers remain invalid. Also
corrected the stale public XNOR comment: both semantics already support exact
results.

Expanded the existing per-bit truth tables and all 65,536 half encodings to
both CLAMP settings. Wave64 tests now combine both settings with half selectors,
DPP routes/row masks, full and mixed EXEC, unaligned destinations, source aliases,
and strict exact/loose semantics on every available CPU level. Validation tests
still reject unrelated modifiers and preserve outputs on errors. Existing DPP
hardware fixtures are retained; this is a source-contract review, not a new
hardware characterization.

Validation: all 1,128 CTests pass under Clang and GCC Release; all 253 tests in
the Boolean/DPP selection pass under ASan/UBSan. All nine audit-tool tests pass.
Logs use `boolean-review-` in each build directory. The ledger records
543 contracts; 266 in-scope mnemonics remain without recorded reviews.

## Vector shift architecture contracts

Reviewed the 32/64-bit LSHLREV, LSHRREV and ASHRREV operations at rocjitsu
`ffc144c564c`. All fifteen common architecture-specific execution/modifier pairs
are identical. RDNA4 additionally has a VOP2 encoding of LSHLREV_B64, with the
same value/count behavior as VOP3; its decoder explicitly rejects DPP. The
review covers 31 handler bodies, ten shared execution helpers and their SIMD
shift helpers. Locations, hashes and shared bodies are recorded in
`../goc-build/coverage-work/shift-upstream-review.json`. Shared APIs remain
appropriate; no architecture suffix or optional exception-register output is
needed.

A is the 32-bit shift count, masked to five or six bits. B is the shifted value,
with the low VGPR first for 64-bit forms. Left shifts discard high bits, logical
right shifts insert zeros, and arithmetic right shifts replicate the sign bit.
GoC's scalar arithmetic-right path complements negative values around a logical
shift to avoid relying on implementation-defined signed shifts. This agrees
with the upstream signed-shift model. The 32-bit forms route A through DPP8/16;
64-bit forms reject DPP. Wave64 adjusts row masks between its two halves.

Fixed rejection of the ignored CLAMP modifier. All reviewed execution helpers
leave the shifted bit pattern unchanged by CLAMP. Other low modifiers remain
invalid, and validation precedes empty EXEC. The SIMD path still processes eight
lanes at a time on v3 and newer CPUs. It combines low/high 32-bit word shifts for
64-bit values, exploiting AVX2 logical out-of-range shift counts to insert zero
and using arithmetic shifts for sign fill. Sources are loaded before either
half is stored, supporting reversed and coincident destination aliases. No
arithmetic kernel, dispatch choice or abstraction changed.

Expanded every-count/per-bit tests to both CLAMP settings, including counts
0, 31, 32 and 63 and repeated modulo periods. Expanded Wave64 tests to combine
CLAMP with strict exact/loose semantics, routing, sign-bit and single-bit inputs,
full/mixed EXEC and source aliases on each available CPU level. Existing random
upper-count-bit tests, all destination-alias combinations, and the gfx1201 DPP
shift corpus remain unchanged. This validates the borrowed operation contract;
it introduces no hardware-derived numerical model.

Validation: all 1,128 CTests pass under Clang and GCC Release; all nineteen
shift/Wave64/DPP regression tests pass under ASan/UBSan. All nine audit-tool
tests pass. Logs use `shift-review-` in each build directory. The ledger records
549 contracts; 260 in-scope mnemonics remain without recorded reviews.

## Vector bit-count architecture contracts

Reviewed CLZ, CTZ, CLS, BCNT and MBCNT_LO/HI at rocjitsu `ffc144c564c`.
All eighteen architecture-specific execution/modifier pairs are identical
(36 bodies). Nine shared execution helpers were checked against GoC's scalar,
v3 and v4 implementations. Locations, hashes and shared bodies are recorded in
`../goc-build/coverage-work/bit-count-upstream-review.json`. No architecture
suffix or optional exception-register output is needed.

CLZ/CTZ return UINT32_MAX for zero, rather than a count of 32. CLS complements
negative inputs before counting leading zeros, including the sign bit, and
returns UINT32_MAX for both zero and all ones. BCNT adds the population count
to B modulo 2^32. MBCNT first intersects A with the low/high half of the physical
lane's exclusive prefix mask, then adds B with the same wrapping. The prefix
uses the destination lane number, independently of EXEC and DPP's source lane.
In particular, Wave32 MBCNT_HI copies B and Wave64 MBCNT_LO counts all A bits in
the upper half of the wave. Existing GoC arithmetic agrees with these helpers.

Fixed rejection of CLAMP, which all reviewed helpers ignore. Other low flags
remain invalid. Wave64 DPP stages both input halves before storing results and
retains physical lane numbering in the counting kernel. Both wave sizes retain
masked stores and source/destination aliases. The SIMD kernels are unchanged:
v3 processes eight lanes using nibble-table population counts and bit
propagation for scans; v4 processes sixteen lanes, with LZCNT for scans and the
same nibble-table population strategy. The existing baseline selection for
Wave32 MBCNT_HI below v4 is retained.

Expanded per-bit-pair/complement and half-encoding/random reference checks to
both CLAMP settings, including accumulator wrapping. Wave64 DPP prefix tests
now combine both settings with every row mask, empty/single/mixed/full EXEC,
unaligned storage and each source alias. Wave64 scan/population DPP tests also
exercise both settings in strict exact and loose semantics. Existing literal
sentinels, count composition, and gfx1201 Wave32/Wave64 DPP fixtures remain
unchanged. This review ports the existing operation/modifier contract without
new hardware characterization.

Validation: all 1,128 CTests pass under Clang and GCC Release; all sixteen
bit-count/DPP regression tests pass under ASan/UBSan. All nine audit-tool tests
pass. Logs use `bit-count-review-` in each build directory. The ledger records
555 contracts; 254 in-scope mnemonics remain without recorded reviews.

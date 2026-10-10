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
both targets. Exact support follows empirical evidence; unsupported exact/global
state requests must fail honestly under the documented contracts. Loose SIMD
paths must retain their existing performance and may skip optional state outputs.

## Evidence and starting point

- Start: `80470bd`, 682 instruction APIs, 674 tests passing with Clang and GCC.
- ISA and rocjitsu source: local TheRock `rocm-systems` revision `ffc144c564c`.
  XML content fingerprints are recorded in the inventory.
- Borrowed hardware captures establish only the targets and cases they capture.
  No new GPU captures have been made for this goal.
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
| DOT2 and DOT2ACC | FP32 DOT2 split into unsuffixed RDNA3 / `_rdna4`; shared modifiers/DPP/loose SIMD; GFX1100 captures, aliases and host-state tests | DOT2ACC added with the same exact model and loose SIMD; dual forms, RDNA3 reporting, and Wave64 remain |
| Floating min/max/median | Existing RDNA4 NUM forms | RDNA3 names and NaN/zero policies, signaling NaNs, all widths and packed forms |
| Scalar integer arithmetic | RDNA3 add/sub/addc/subb/addk spellings reuse RDNA4 implementations, as confirmed by XML aliases; all twelve CMPK forms added with exhaustive immediate tests | Continue full family audit; register/control operations tracked separately |
| Scalar/register moves | MOV/CMOV B32/B64 and MOVK/CMOVK implemented for both architectures; exhaustive immediate/condition tests | Relative addressing and hardware-register access; SAVEEXEC/WREXEC implemented at both register widths |
| Vector/register routing | READLANE/WRITELANE/READFIRSTLANE now shared for Wave32/64, including lane-index wrapping and zero EXEC; existing DPP/permlane/move coverage | Wave32 vector-relative moves now implemented with bounded views; swaps, permlane64 and broader cross-width behavior remain |
| Comparisons | Existing RDNA4 results/SCC/EXEC and exception tests | RDNA3 integer true/false forms now implemented; floating constant forms, Wave64, and architecture-specific floating policies remain |
| Basic floating arithmetic, FMA, conversions | Existing RDNA4 implementations and captures | Audit shared numerical behavior and latest rocjitsu exact models for both targets |
| Transcendentals, reciprocal, division | Existing partial exact coverage | Review newer rocjitsu models/captures, differing semantics and state outputs |
| Integer/vector arithmetic and bit operations | Existing broad RDNA4 coverage | Check RDNA3 spellings, modifier encodings, widths, carry outputs and all aliases |
| Interpolation | Existing RDNA4 paths | Compare RDNA3 contracts and empirical behavior |
| Dual-issue VOPD | No separately named APIs | Inventory independent operations and cross-operation source/output dependencies |

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

The signed RDNA3 spelling remains open. rocjitsu's RDNA3 and RDNA4 handlers
both report signed overflow, whereas GoC's RDNA4 hardware captures establish
that the scalar output is bit 64 of the full mathematical sum (its extended
sign). These differ even without overflow: `0 * 0 + (-1)` has scalar bit 1
in the RDNA4 capture model but bit 0 in rocjitsu. XML's prose says
"overflow/carryout" and does not resolve the discrepancy. Do not add a signed
alias or copy rocjitsu's overflow result without RDNA3-specific evidence.

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
SWAP_B16 and SWAPREL remain open because their half-register and relative-range
rules need separate auditing. Missing canonical API names: 67.

Validation for these additions: all 700 tests pass with Clang and GCC; the four
swap/routing tests also pass under ASan/UBSan.

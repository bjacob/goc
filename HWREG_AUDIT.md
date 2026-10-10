# Hardware-register implementation audit

This is an implementation checklist, not a coverage-completion claim. None of
`S_GETREG_B32`, `S_SETREG_B32`, or `S_SETREG_IMM32_B32` has a public GoC API yet.

Evidence inspected: AMD RDNA3 and RDNA4 ISA manuals, sections 3.4 and 16.2;
`amdgpu_isa_rdna3.xml` and `amdgpu_isa_rdna4.xml`, `OPR_HWREG`; and rocjitsu
revision `ffc144c564c`, `lib/rocjitsu/src/rocjitsu/vm/amdgpu/hwreg.cpp`.
The source copies are in the sibling TheRock checkout. Local manual text copies
are `../goc-build/coverage-work/rdna3.txt` and `/tmp/goc-rdna4-mirror.txt`.

## Register-ID coverage

The table includes every predefined ID in both XML operand maps: 24 RDNA3
entries and 19 RDNA4 entries. RDNA4 manual section 3.4 additionally identifies
ID 28 as `IB_STS2`; that prose-only entry is marked explicitly. A dash means
absent from that architecture's XML map, not permission to return zero.
Names retain XML spelling, including `WAVE_` and `SHADER_` prefixes. Manual
register names sometimes omit these prefixes.

| ID | RDNA3 XML | RDNA4 XML / prose |
|---|---|---|
| 1 | `MODE` | `WAVE_MODE` |
| 2 | `STATUS` | `WAVE_STATUS` |
| 3 | `TRAPSTS` | — |
| 4 | — | `WAVE_STATE_PRIV` |
| 5 | `GPR_ALLOC` | `WAVE_GPR_ALLOC` |
| 6 | `LDS_ALLOC` | `WAVE_LDS_ALLOC` |
| 7 | `IB_STS` | — |
| 8 | `PC_LO` | — |
| 9 | `PC_HI` | — |
| 10 | — | `PERF_SNAPSHOT_DATA` |
| 11 | — | `PERF_SNAPSHOT_PC_LO` |
| 12 | — | `PERF_SNAPSHOT_PC_HI` |
| 13 | `IB_DBG1` | — |
| 14 | `FLUSH_IB` | — |
| 15 | `SH_MEM_BASES` | `PERF_SNAPSHOT_DATA1` |
| 16 | `SHADER_TBA_LO` | `PERF_SNAPSHOT_DATA2` |
| 17 | `SHADER_TBA_HI` | `WAVE_EXCP_FLAG_PRIV` |
| 18 | `PERF_SNAPSHOT_PC_LO` | `WAVE_EXCP_FLAG_USER` |
| 19 | `PERF_SNAPSHOT_PC_HI` | `WAVE_TRAP_CTRL` |
| 20 | `SHADER_FLAT_SCRATCH_LO` | `WAVE_SCRATCH_BASE_LO` |
| 21 | `SHADER_FLAT_SCRATCH_HI` | `WAVE_SCRATCH_BASE_HI` |
| 23 | `HW_ID1` | `WAVE_HW_ID1` |
| 24 | `HW_ID2` | `WAVE_HW_ID2` |
| 25 | `POPS_PACKER` | — |
| 26 | `SCHED_MODE` | — |
| 27 | `PERF_SNAPSHOT_DATA` | — |
| 28 | `IB_STS2` | `IB_STS2` (manual only) |
| 29 | `SHADER_CYCLES` | `SHADER_CYCLES_LO` |
| 30 | — | `SHADER_CYCLES_HI` |

## Confirmed architecture differences

- The same ID can name unrelated registers: ID 18 selects
  `PERF_SNAPSHOT_PC_LO` on RDNA3 and `EXCP_FLAG_USER` on RDNA4. ID 17 selects
  `SHADER_TBA_HI` versus `EXCP_FLAG_PRIV`. All three instruction entry points
  therefore require an RDNA4 variant, with the unsuffixed name denoting RDNA3.
- RDNA3 `STATUS` permits selected privileged writes and contains `SCC` at bit
  0. RDNA4 `STATUS` is read-only; `SCC` moves to `STATE_PRIV` bit 9, which is
  writable only with `STATUS.PRIV` set. Returning raw RDNA3 status bits through
  an RDNA4 API would be incorrect.
- RDNA3 manual Table 5 marks these `STATUS` bits privileged-writable:
  0–4, 8, 13, 18, 20–21, 25, 27, 29. Other bits must be preserved. This is a
  field-specific rule, not permission to overwrite the entire register.
- RDNA3 `MODE` includes `DX10_CLAMP`, `IEEE`, trap-after-instruction and
  exception-enable fields. RDNA4 distributes exception controls into
  `TRAP_CTRL` and exception state into `EXCP_FLAG_USER`/`EXCP_FLAG_PRIV`.
  RDNA4 `MODE` instead documents FP round/denormal controls, `FP16_OVFL`,
  scalar-prefetch enable and performance disable. Do not reuse one whole-word
  writable mask across architectures.
- RDNA3 `FLAT_SCRATCH` is a 64-bit, 256-byte-aligned address with privileged
  writes. RDNA4 calls it `SCRATCH_BASE`. These are raw guest registers; updating
  them must not allocate or access host scratch memory.

## Read side effects and external state

- **RDNA4 GETREG is not universally read-only.** The performance-snapshot
  description in section 3.4.9 says to read `PERF_SNAPSHOT_PC_HI` last because
  that read resets/unlocks the snapshot registers for another capture. It also
  says only the sampled wave reads snapshot data; other waves read zero. A
  plain register snapshot without sampled-wave identity and lifecycle state
  cannot model this complete instruction behavior.
- The prose does not identify a separate architectural lock register, nor does
  it specify all post-read register values. Do not invent a named output
  register or assume that clearing `PERF_SNAPSHOT_DATA.VALID` fully models the
  unlock. Resolve this with an additional authoritative model or capture before
  claiming exact support for these reads.
- RDNA3 XML lists snapshot registers, but the inspected RDNA3 manual text does
  not describe the RDNA4 read-unlock protocol. Do not silently apply it to
  RDNA3 as though verified there.
- `SHADER_CYCLES` needs a caller-supplied guest clock value. Do not substitute
  the host wall clock or CPU cycle counter. RDNA4 shader reads of HI/LO are
  separate and may observe rollover; the documented freeze/unfreeze protocol
  in XML is for **host** access, not shader `S_GETREG_B32`.
- RDNA4 XML describes a 60-bit shader-cycle counter while the manual calls it
  64-bit. Raw caller-provided halves can preserve values without inventing
  the counter's progression or overflow model; the width discrepancy remains
  relevant to a full guest-clock implementation.
- Dynamic allocation, hardware identity, PC, status and scratch values must
  come from the emulator. GoC cannot infer them from arithmetic flags.

## Write semantics and non-register effects

- Decode `ID = imm16[5:0]`, `offset = imm16[10:6]`, and
  `size = imm16[15:11] + 1`. Sizes extend up to 32. Clip the accessed field at
  the end of the 32-bit register; avoid C/C++ shifts by 32.
- SETREG uses the shifted field mask intersected with the register's writable
  mask and privilege permissions, then merges into the old value. The RDNA3
  printed pseudocode omits the mask shift even though the operation's prose
  requires it. RDNA4 pseudocode and rocjitsu's field insertion include it.
  Treat this as a documentation discrepancy, not an architecture difference.
- A read-only field or a field unavailable at the current privilege is masked
  out by the manual's SETREG pseudocode. Rocjitsu's API-level `ReadOnly` and
  `Privileged` errors are simulator diagnostics, not proof that the hardware
  traps or that GoC should return the same error for a permitted no-op write.
- SETREG to an exception register performs masked replacement, allowing flags
  to be cleared. It must not reuse arithmetic's sticky-OR update operation.
- RDNA3 `FLUSH_IB` is write-only and has an instruction-buffer flush effect.
  Section 3.4's overview says “all waves”; its internal-register table describes
  the targeted wave and a bit-0 trigger. Its scope needs reconciliation, and
  merely storing a new integer value does not implement the instruction.
- Changing guest `MODE` must only update guest state. It must never set host
  rounding, denormal controls, exception masks or exception flags. The emulator
  remains responsible for deriving subsequent GoC call flags from guest mode.
- HALT, trap-control and scratch changes have emulator consequences beyond
  their raw bits. The library should report architectural register updates;
  the emulator applies scheduling, trap delivery and memory-model effects.

## What can and cannot be borrowed from rocjitsu

- Reuse the established descriptor decoding, clipped field extraction/insertion
  and register-ID maps. Translate its legacy internal status/exception storage
  only if the GoC API actually uses that representation; raw architectural
  register inputs need no such translation.
- Its RDNA3 `STATUS` descriptor is read-only despite the manual's privileged
  writable bits. Its RDNA3 scratch descriptors are likewise read-only despite
  privileged writes in the manual. These policies cannot be copied unchanged.
- Many register descriptors have `HwregState::Unsupported`, including snapshot
  and allocation state. Their presence in a table is not an implementation.
- Its supported write path is largely whole-value merge plus state dispatch;
  that does not establish reserved-bit write masks for every register.

## API and validation gates

- Preserve one trailing nullable output pointer per architectural register that
  the instruction may write. Do not put writable state in an opaque context
  object. Required instruction results remain required; optional side-output
  fidelity follows the existing loose/exact contract.
- Input packaging is the outstanding API choice: individual by-value implicit
  inputs match the current PLAN, while a named read-only HWREG state struct
  would avoid roughly twenty input parameters. Either must use explicit guest
  values and preserve the architecture-specific layouts above.
- Before declaring coverage, test every recognized ID for each architecture,
  all 32 offsets and 32 encoded widths, size-32 and clipped fields, SGPR and
  literal SETREG equivalence, privilege filtering, unaffected-bit preservation,
  exception clearing, aliases, and unchanged outputs on errors.
- Also test MODE writes under nondefault host FP settings without host control
  changes, snapshot ownership/unlock, and externally supplied clock rollover.
  Unsupported side effects must remain explicit; do not label the three
  inventory rows complete merely because symbols or extraction helpers exist.

// SPDX-License-Identifier: MIT

#ifndef GOC_INSTRUCTIONS_H_
#define GOC_INSTRUCTIONS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// EXEC-dependent Wave32 entry points use a 32-bit exec_mask (Wave64 uses 64 bits).
// Scalar, pseudo-scalar, WMMA and SWMMAC instructions ignore EXEC and omit exec_mask.
// Matrix instructions read and write all lanes. Each Wave32 VGPR pointer names 32
// contiguous uint32_t lane words. Pointer arrays and backing storage must be valid.
// Whole VGPRs may alias; distinct VGPR addresses must not overlap. Sources are
// conceptually read before writes. Inactive destination lanes and all destinations
// on error are unchanged. Loose FP32 paths require host nearest-even rounding
// with denormals enabled. Integer arithmetic paths preserve all host FP state.

// Optional architectural register outputs are appended to the operand list.
// excp_flag_user accumulates this invocation's participating-lane exceptions
// with bitwise OR, preserving all existing bits. NULL opts out. It must not
// overlap operand storage unless the instruction explicitly permits it.
// Errors leave it and all other outputs unchanged.
// Faithful optional global-state output is required only for bit-exact semantics
// with a non-NULL pointer, regardless of GOC_SEMANTICS_STRICT. Ordinary instruction
// results (including comparison masks and SCC) remain required in every mode.
// Loose semantics require no optional global-state output, even with a non-NULL
// pointer: SIMD paths are expected to skip it and leave the register unchanged.
// Any loose reporting is not guaranteed complete or accurate; loose arithmetic
// need not match hardware, so its exception flags need not match hardware either.
// Loose mode never returns GOC_ERROR_UNSUPPORTED_GLOBAL_STATE.
// Reporting is implemented for floating comparisons, scalar rounding,
// FP16 FMA/FMAC/literal/packed forms, V_DIV_FIXUP_F16/F32/F64, and
// V_DIV_FMAS_F32/F64, FP16/32 SIN/COS, and V_RCP_IFLAG_F32. MIXLO/MIXHI_F16
// and DOT2_F32_F16/BF16 generate no flags and leave the register unchanged. Other
// non-loose requests with a non-NULL pointer return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE
// before operand access, even for empty EXEC, before other validation. NULL opts
// out of reporting in every mode and preserves numerical paths and validation.

// Bit-preserving Wave32 move, shared by RDNA3 and RDNA4. One VGPR per operand.
// Supports loose and exact semantics, EXEC masking, and DPP8/DPP16 source routing.
// Low instruction-flag bits must be zero (arithmetic modifiers are not implemented).
// No FP interpretation or exceptions; inactive destinations remain unchanged.
int goc_v_mov_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a);

// Wave32 raw half move. HIGH_A/D select source/destination halves; the other
// destination half is preserved. ABS_A clears and NEG_A toggles the selected
// sign bit (in that order), with no FP conversion or NaN canonicalization.
// OMOD and CLAMP are accepted but ignored, matching hardware. DPP8/DPP16 route
// A before half/sign selection. Both semantics are exact; whole VGPRs may alias.
// Inactive destinations are preserved; zero EXEC permits null pointers.
int goc_v_mov_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a);

// Exchange selected halves in active lanes. HIGH_A/D are the only valid flags.
// Both operands are read/write; unselected halves and inactive lanes survive.
// Supports exchanging two halves of the same VGPR. Both semantics are exact.
// Zero EXEC permits null pointers. Shared RDNA3/RDNA4 Wave32/Wave64 operations.
int goc_v_swap_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, uint32_t *const *a);

int goc_v_swap_b16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, uint32_t *const *a);

// Exchange two read/write VGPRs in active lanes. Both operands use their
// original values; whole-register aliasing is allowed. Inactive lanes are
// unchanged. No instruction flags are supported. Both semantics are exact.
// Zero EXEC permits null pointers. Shared by RDNA3 and RDNA4.
int goc_v_swap_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, uint32_t *const *a);

int goc_v_swap_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, uint32_t *const *a);

// Wave32 PERMLANE64 is a no-op: no EXEC input and no operand access; null
// pointers are allowed. Wave64 copies A[lane ^ 32] into active D lanes, reading
// inactive source lanes too. D may alias A; inactive destinations are unchanged.
// No instruction flags are supported. Both semantics are exact. Shared by
// RDNA3 and RDNA4; neither operation interprets bits as floating-point values.
int goc_v_permlane64_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                         const uint32_t *const *a);

int goc_v_permlane64_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

// Register-only DS permutations; no LDS storage or global-state updates.
// ADDR and DATA each use one VGPR. Add the 16-bit byte offset to ADDR, then
// select address bits [6:2] within each 32-lane group. Wave32 is shared;
// RDNA3 Wave64 operates independently in each half. RDNA4 Wave64 instead uses
// bits [7:2] to route across all 64 lanes. BPERMUTE gathers DATA from the selected
// source; PERMUTE scatters DATA to the selected destination. EXEC applies to
// both reads and writes: inactive sources contribute zero, unfilled active
// destinations receive zero, and inactive destinations are preserved.
// For PERMUTE collisions GoC chooses the highest active source lane, following
// the ISA pseudocode; hardware prose does not guarantee a particular winner.
// All inputs are read before output stores; whole-register aliases are allowed.
// Both semantics are exact within this collision rule. No instruction flags
// are supported; errors preserve D. Zero EXEC permits null VGPR pointers.
int goc_ds_permute_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *addr, const uint32_t *const *data,
                       uint16_t offset);

int goc_ds_permute_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *addr,
                              const uint32_t *const *data, uint16_t offset);

int goc_ds_permute_b32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *addr,
                                    const uint32_t *const *data, uint16_t offset);

int goc_ds_bpermute_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *addr,
                        const uint32_t *const *data, uint16_t offset);

int goc_ds_bpermute_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *addr,
                               const uint32_t *const *data, uint16_t offset);

int goc_ds_bpermute_b32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *addr,
                                     const uint32_t *const *data, uint16_t offset);

// RDNA4 FI backward permutation: same byte addressing as BPERMUTE, but reads
// inactive source lanes too. EXEC only controls destination writes. Wave64
// routes across all 64 lanes. The other DS permutation contracts apply.
int goc_ds_bpermute_fi_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *addr,
                           const uint32_t *const *data, uint16_t offset);

int goc_ds_bpermute_fi_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *addr,
                                  const uint32_t *const *data, uint16_t offset);

// DS_SWIZZLE routes raw bits within each 32-lane row without LDS storage.
// offset selects bit-mask (<0x8000), quad (<0xc000), masked rotate (<0xe000),
// or FFT routing. Rotations wrap within the row. Shared RDNA3/RDNA4 ISA
// pseudocode semantics; rotate/FFT empirical validation is RDNA4-only.
// Inactive sources supply zero; inactive destinations are preserved. All source
// values precede output writes, permitting aliases. Both semantics are exact;
// no instruction flags are supported. Zero EXEC permits null VGPR pointers.
int goc_ds_swizzle_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, uint16_t offset);

int goc_ds_swizzle_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a, uint16_t offset);

// DPP8 permutes source A within each group of eight lanes before arithmetic
// modifiers. Pack eight 3-bit lane indices into bits 40..63, index 0 first, and
// set GOC_DPP8. Without FI, an inactive source supplies positive zero; FI reads
// its stored value. Inactive destinations remain unchanged. Supported by FP32
// FMA/FMAC, ADD/SUB/SUBREV/MUL/MUL_DX9_ZERO, binary min/max, and three-input
// min/max/median, and FP32 unary math (rounding, SQRT/RCP/RSQ/EXP/LOG, FRACT,
// FREXP mantissa), and wave32 CLZ/CTZ/CLS/BCNT/MBCNT, 32-bit shifts, and 32-bit Boolean
// AND/OR/XOR/XNOR/NOT, and 32-bit integer min/max/median (two or three inputs),
// plus LSHL_ADD/ADD_LSHL/LSHL_OR/AND_OR/OR3/XOR3/XAD/LERP. Each supports its
// applicable modifiers and whole-VGPR aliases. BFE/BFI/BFM/BFREV, ALIGNBIT,
// ALIGNBYTE, and PERM also support DPP, as do 32-bit ADD_NC/SUB_NC/SUBREV_NC
// (including CLAMP) and ADD3. SUBREV_NC_U32 permutes B instead of A.
// Signed/unsigned 24-bit MUL and MUL_HI support DPP, with CLAMP for low results.
// The four integer MAD forms support DPP with CLAMP and their 16-bit selectors.
// The 16-bit AND/OR/XOR/NOT and non-packed integer arithmetic forms support
// DPP with their source/destination selectors and applicable CLAMP modifiers.
// This includes non-packed 16-bit MAD, MIN3, MAX3, and MED3.
// FP16 ADD/SUB/SUBREV/MUL and binary/ternary min/max/median support DPP
// with all modifiers. FP16 FMA/FMAC support DPP in loose and exact semantics.
// FP16 unary math also supports DPP with all applicable modifiers, as do FP16
// and FP32 LDEXP/FREXP exponent, all six FP32/integer conversions, and all six
// FP16 conversions, byte/nibble-to-FP32 conversions, and FP32-to-byte packing.
static const uint64_t GOC_DPP8 = 1ULL << 32;
static const uint64_t GOC_DPP_FI = 1ULL << 33;
static const uint32_t GOC_DPP8_SELECT_SHIFT = 40;
static const uint64_t GOC_DPP8_SELECT_MASK = 0xffffffULL << 40;

// DPP16 permutes the same source as DPP8 within 16-lane rows before arithmetic
// modifiers (B for SUBREV_NC_U32, A otherwise). Select
// exactly one of GOC_DPP8/GOC_DPP16. DPP16 control is the architectural 9-bit
// encoding: quad_perm 0x000..0x0ff; row_shl 0x101..0x10f; row_shr 0x111..0x11f;
// row_ror 0x121..0x12f; row_mirror 0x140; row_half_mirror 0x141;
// row_share 0x150..0x15f; row_xmask 0x160..0x16f. Other controls are invalid.
// Row/bank fields enable destination rows of 16 lanes and banks of four lanes.
// Use both full MASK constants to enable all destinations; zero fields disable
// writes. FI permits inactive in-range sources. Without a readable source,
// BOUND_CTRL supplies positive zero; otherwise the destination is preserved.
// FI does not permit out-of-range sources. Supports the same instructions as DPP8.
static const uint64_t GOC_DPP16 = 1ULL << 34;
static const uint64_t GOC_DPP_BOUND_CTRL = 1ULL << 35;
static const uint32_t GOC_DPP_CTRL_SHIFT = 40;
static const uint64_t GOC_DPP_CTRL_MASK = 0x1ffULL << 40;
static const uint32_t GOC_DPP_ROW_SHIFT = 49;
static const uint64_t GOC_DPP_ROW_MASK = 0xfULL << 49;
static const uint32_t GOC_DPP_BANK_SHIFT = 53;
static const uint64_t GOC_DPP_BANK_MASK = 0xfULL << 53;

// PERMLANE flags are a bit field. FI permits reading inactive source lanes.
// Otherwise BOUND_CTRL selects zero for an inactive source; without it the
// destination is preserved. Neither flag enables inactive destination lanes.
static const uint32_t GOC_PERMLANE_FI = 1U << 0;
static const uint32_t GOC_PERMLANE_BOUND_CTRL = 1U << 1;

// Wave32 lane permutations support both semantics and preserve all host FP state.
// Guest FP flags have no effect. Each source index is its low four bits; the
// ordinary forms select within each group of 16 lanes, and X forms select from
// the other group. Whole-register aliasing, including d == a or b, is supported.

// Indices are successive nibbles of lo (lanes 0..7) and hi (8..15), repeated.
int goc_v_permlane16_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t lo, uint32_t hi);

// Indices are successive nibbles of lo (lanes 0..7) and hi (8..15), repeated.
int goc_v_permlanex16_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t lo, uint32_t hi);

// Indices come from each destination lane of b.
int goc_v_permlane16_var_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b);

// Indices come from each destination lane of b.
int goc_v_permlanex16_var_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

// Scalar packing/selection ignores EXEC, including zero EXEC; output pointers
// are required. Inputs are passed by value and may come from destination storage.
// Both loose and empirical exact semantics are supported; instruction_flags must
// be zero. Guest FP flags have no effect and all host FP state is preserved.
// Errors leave outputs unchanged. SCC outputs, where present, are written after
// d and win on overlap, including within a 64-bit destination. Other operations
// preserve SCC. Conditional selection reads only input_scc bit 0.

// Pack low A and low B into low/high destination halves.
int goc_s_pack_ll_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b);

// Pack low A and high B into low/high destination halves.
int goc_s_pack_lh_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b);

// Pack high A and low B into low/high destination halves.
int goc_s_pack_hl_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b);

// Pack high A and high B into low/high destination halves.
int goc_s_pack_hh_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b);

// Replicate each input bit i into output bits 2*i and 2*i+1.
int goc_s_bitreplicate_b64_b32(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint32_t a);

// Select a when input_scc bit 0 is 1, otherwise b.
int goc_s_cselect_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t input_scc);

// Select a when input_scc bit 0 is 1, otherwise b.
int goc_s_cselect_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                      uint64_t b, uint32_t input_scc);

// Set output bit i when input nibble i is nonzero; upper 24 bits are zero. SCC is result != 0.
int goc_s_quadmask_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t *scc);

// Set output bit i when input nibble i is nonzero; upper 48 bits are zero. SCC is result != 0.
int goc_s_quadmask_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint32_t *scc);

// Replace each nonzero input nibble with 0xf. SCC is result != 0.
int goc_s_wqm_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                  uint32_t *scc);

// Replace each nonzero input nibble with 0xf. SCC is result != 0.
int goc_s_wqm_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                  uint32_t *scc);

// Set guest MODE rounding bits [3:0] or denormal bits [7:4] to immediate[3:0].
// Higher immediate bits are ignored and every other MODE bit is preserved.
// mode is an optional read/write architectural register pointer; NULL opts out.
// Both loose and exact semantics perform this inexpensive update when non-NULL.
// No EXEC input, instruction flags, or host FP-environment changes. Errors leave
// MODE unchanged. Shared by RDNA3 and RDNA4.
int goc_s_round_mode(uint64_t flags, uint64_t instruction_flags, uint16_t immediate,
                     uint32_t *mode);

int goc_s_denorm_mode(uint64_t flags, uint64_t instruction_flags, uint16_t immediate,
                      uint32_t *mode);

// Store the next instruction address, pc + 4, in the required scalar pair D.
// pc is the byte address of this instruction; addition wraps modulo 2^64.
// No EXEC input or instruction flags. Both semantics are exact. Does not modify
// the guest PC or host FP state; errors preserve D. Shared RDNA3/RDNA4 operation.
int goc_s_getpc_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t pc);

// Scalar comparisons execute once per wave and ignore EXEC, including zero.
// scc is required and receives 0 or 1. Inputs are passed by value and may come
// from output storage, including a word within uint64_t storage. Both loose and
// empirical exact semantics are supported. instruction_flags must be zero.
// Preserves all host FP state independently of rounding; errors leave scc unchanged.
// FP16 reads low halves. Floating comparisons equate signed zeros and apply guest
// input flushing. Output flushing and FP16_OVFL have no effect; integer/bit tests
// also ignore input flushing. Ordered predicates are false for either NaN;
// negated predicates are their logical complements, including for NaNs.
// In exact mode with non-NULL excp_flag_user, signaling NaNs accumulate INVALID.
// Finite operand pairs accumulate INPUT_DENORM if either input is subnormal and
// input flushing is disabled. Loose mode leaves excp_flag_user unchanged.

// Wave32 relative VGPR moves shared by RDNA3/RDNA4. d/a are full register-file
// views starting at VGPR0, bounded by d_count/a_count. d_base/a_base are encoded
// register indices. RELS adds M0 to source; RELD to destination; RELSD to both.
// RELSD_2 uses M0[9:0] for source and M0[25:16] for destination. Per ISA section
// 3.3.2.2, offsets above 255 or resolved indices outside the allocation are out
// of range: sources redirect to VGPR0, and destination writes are discarded.
// A source allocation of zero returns zero. Zero counts permit null arrays.
// No indexing wraps; addition uses widened arithmetic. Whole VGPRs may alias.
// Both semantics are exact; DPP routes the selected source with MOV_B32 flags.
// Invalid flags leave all outputs unchanged.
int goc_v_movrels_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                      uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int goc_v_movreld_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                      uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int goc_v_movrelsd_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                       uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int goc_v_movrelsd_2_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                         uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

// Scalar lane transfers shared by RDNA3/RDNA4; one VGPR input/output.
// READFIRSTLANE reads the first active lane, or lane 0 for zero EXEC.
// READLANE/WRITELANE ignore EXEC; lane indices wrap modulo 32 or 64.
// WRITELANE preserves every other lane. All copy raw bits, support both
// semantics, require zero instruction_flags, and preserve host FP state.
// Scalar destinations may overlap source lane storage; values are read first.
int goc_v_readfirstlane_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a);

int goc_v_readlane_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, uint32_t lane);

int goc_v_writelane_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d, uint32_t a,
                        uint32_t lane);

int goc_v_readfirstlane_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a);

int goc_v_readlane_b32_wave64(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                              const uint32_t *const *a, uint32_t lane);

int goc_v_writelane_b32_wave64(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                               uint32_t a, uint32_t lane);

// EXEC Boolean updates shared by RDNA3/RDNA4. SAVEEXEC writes old exec_mask
// to d; WREXEC writes the new EXEC value to d. EXEC is replaced, not accumulated.
// NOT0 complements a, NOT1 complements exec_mask. SCC is new EXEC != 0.
// These execute even when exec_mask is zero. Both semantics are exact;
// instruction_flags must be zero. d and scc are required ordinary results;
// exec may be null to omit the implicit register update. Writes occur in order
// d, exec, scc; outputs may alias and later writes win. Errors preserve outputs.
int goc_s_and_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_or_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_xor_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_nand_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                            uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_nor_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_xnor_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                            uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_and_not0_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_or_not0_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_and_not1_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_or_not1_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_and_not0_wrexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                              uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_and_not1_wrexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                              uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int goc_s_and_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                           uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_or_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                          uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_xor_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                           uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_nand_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                            uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_nor_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                           uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_xnor_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                            uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_and_not0_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                                uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_or_not0_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                               uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_and_not1_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                                uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_or_not1_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                               uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_and_not0_wrexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                              uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int goc_s_and_not1_wrexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                              uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

// Scalar moves shared by RDNA3/RDNA4. Copy raw bits; MOVK/CMOVK sign-extend
// their 16-bit immediate. Conditional forms write only when input_scc bit 0
// is set, otherwise d is unchanged. SCC and EXEC are not modified; EXEC is
// ignored. Both semantics are exact and instruction_flags must be zero.
// Inputs passed by value may come from d. Errors preserve d and host FP state.
int goc_s_mov_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);
int goc_s_mov_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a);
int goc_s_cmov_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                   uint32_t input_scc);
int goc_s_cmov_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                   uint32_t input_scc);
int goc_s_movk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate);
int goc_s_cmovk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate,
                    uint32_t input_scc);

// RDNA3 immediate comparisons set SCC to 0 or 1. I32 forms sign-extend the
// 16-bit immediate; U32 forms zero-extend it. Both semantics are exact, EXEC
// is ignored, instruction_flags must be zero, and errors preserve SCC.
int goc_s_cmpk_eq_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_lg_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_gt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_ge_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_lt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_le_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_eq_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_lg_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_gt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_ge_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_lt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

int goc_s_cmpk_le_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint16_t immediate);

// SCC is a == b (signed).
int goc_s_cmp_eq_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a != b (signed).
int goc_s_cmp_lg_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a > b (signed).
int goc_s_cmp_gt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a >= b (signed).
int goc_s_cmp_ge_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a < b (signed).
int goc_s_cmp_lt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a <= b (signed).
int goc_s_cmp_le_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a == b (unsigned).
int goc_s_cmp_eq_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a != b (unsigned).
int goc_s_cmp_lg_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a > b (unsigned).
int goc_s_cmp_gt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a >= b (unsigned).
int goc_s_cmp_ge_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a < b (unsigned).
int goc_s_cmp_lt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is a <= b (unsigned).
int goc_s_cmp_le_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b);

// SCC is 1 when bit (b modulo 32) of a equals 0.
int goc_s_bitcmp0_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b);

// SCC is 1 when bit (b modulo 32) of a equals 1.
int goc_s_bitcmp1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b);

// SCC is 1 when bit (b modulo 64) of a equals 0.
int goc_s_bitcmp0_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                      uint32_t b);

// SCC is 1 when bit (b modulo 64) of a equals 1.
int goc_s_bitcmp1_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                      uint32_t b);

// SCC is a == b (unsigned).
int goc_s_cmp_eq_u64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                     uint64_t b);

// SCC is a != b (unsigned).
int goc_s_cmp_lg_u64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                     uint64_t b);

// SCC is a < b.
int goc_s_cmp_lt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a < b.
int goc_s_cmp_lt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a == b.
int goc_s_cmp_eq_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a == b.
int goc_s_cmp_eq_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a <= b.
int goc_s_cmp_le_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a <= b.
int goc_s_cmp_le_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a > b.
int goc_s_cmp_gt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a > b.
int goc_s_cmp_gt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a != b.
int goc_s_cmp_lg_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a != b.
int goc_s_cmp_lg_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a >= b.
int goc_s_cmp_ge_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is a >= b.
int goc_s_cmp_ge_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                     uint32_t b, uint32_t *excp_flag_user);

// SCC is both operands are numbers.
int goc_s_cmp_o_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                    uint32_t b, uint32_t *excp_flag_user);

// SCC is both operands are numbers.
int goc_s_cmp_o_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                    uint32_t b, uint32_t *excp_flag_user);

// SCC is either operand is NaN.
int goc_s_cmp_u_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                    uint32_t b, uint32_t *excp_flag_user);

// SCC is either operand is NaN.
int goc_s_cmp_u_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                    uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a >= b).
int goc_s_cmp_nge_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a >= b).
int goc_s_cmp_nge_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is 1 if either operand is NaN or a == b.
int goc_s_cmp_nlg_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is 1 if either operand is NaN or a == b.
int goc_s_cmp_nlg_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a > b).
int goc_s_cmp_ngt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a > b).
int goc_s_cmp_ngt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a <= b).
int goc_s_cmp_nle_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a <= b).
int goc_s_cmp_nle_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a == b).
int goc_s_cmp_neq_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a == b).
int goc_s_cmp_neq_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a < b).
int goc_s_cmp_nlt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a < b).
int goc_s_cmp_nlt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// Scalar conversions execute once per wave and ignore EXEC, including zero;
// d is required and SCC is unchanged. Operands are raw register bits. Sources
// may originate from d storage. instruction_flags must be zero; loose semantics
// only. Errors leave d unchanged. Requires host nearest-even rounding and enabled
// denormals; preserves host rounding but may change exception flags.
// Guest input flushing applies to floating inputs. Output flushing applies to
// FP16 outputs, including tininess detection before subnormal rounding for RNE.
// FP16_OVFL saturates finite RNE narrowing overflow; it does not affect RTZ.
// All other guest FP-state flags accepted here have no effect on integer results
// or integer inputs. Floating conversions quiet signaling NaNs.

// Signed 32-bit integer to FP32, nearest-even.
int goc_s_cvt_f32_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t *excp_flag_user);

// Unsigned 32-bit integer to FP32, nearest-even.
int goc_s_cvt_f32_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t *excp_flag_user);

// FP32 to signed 32-bit integer, truncating and saturating; NaNs yield zero.
int goc_s_cvt_i32_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t *excp_flag_user);

// FP32 to unsigned 32-bit integer, truncating and saturating; NaNs/negatives yield zero.
int goc_s_cvt_u32_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t *excp_flag_user);

// FP32 to FP16, nearest-even; writes a zero upper half.
int goc_s_cvt_f16_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t *excp_flag_user);

// Low FP16 half to FP32.
int goc_s_cvt_f32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t *excp_flag_user);

// High FP16 half to FP32.
int goc_s_cvt_hi_f32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);

// Two FP32 values to low/high FP16 halves, truncating toward zero.
// Finite overflow saturates to the largest finite half; infinities remain infinite.
int goc_s_cvt_pk_rtz_f16_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t b, uint32_t *excp_flag_user);

// Scalar rounding ignores EXEC, including zero EXEC; d is required. Inputs and
// outputs are raw IEEE bits. FP16 reads the low half and writes a zero upper
// half. Signed zero and NaN payload/sign bits survive; signaling NaNs are quieted.
// Both loose and empirical exact semantics are supported. instruction_flags must
// be zero. Guest input flushing applies; output flushing and FP16_OVFL have no
// effect. Preserves all host FP state and SCC, independently of host rounding.
// Inputs may originate from d storage. Errors leave d unchanged.
// Exact mode with non-NULL excp_flag_user accumulates INVALID for signaling NaNs
// and INPUT_DENORM for preserved subnormals. Discarding a fractional part does
// not raise INEXACT. Loose mode leaves excp_flag_user unchanged.

// Round to an integral value toward positive infinity.
int goc_s_ceil_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                   uint32_t *excp_flag_user);

// Round to an integral value toward positive infinity.
int goc_s_ceil_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                   uint32_t *excp_flag_user);

// Round to an integral value toward negative infinity.
int goc_s_floor_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);

// Round to an integral value toward negative infinity.
int goc_s_floor_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);

// Round to an integral value toward zero.
int goc_s_trunc_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);

// Round to an integral value toward zero.
int goc_s_trunc_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);

// Round to an integral value to nearest, with ties to even.
int goc_s_rndne_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);

// Round to an integral value to nearest, with ties to even.
int goc_s_rndne_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);

// Scalar fused multiply-add ignores EXEC, including zero EXEC; d is required.
// Operands and literals are raw IEEE bits. FP16 reads low halves and writes a
// zero upper half. Rounds once to the destination format. SCC is unchanged.
// instruction_flags must be zero. Loose semantics only; NaN signs/payloads are
// unspecified. Guest input/output flushing and FP16_OVFL are supported, with
// tininess detected before destination subnormal rounding. Requires host nearest-
// even rounding and enabled denormals; preserves rounding but may change host
// exception flags. Errors leave d unchanged. Inputs may come from d storage.

// Fused a * b + old *d.
int goc_s_fmac_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t *excp_flag_user);

// Fused a * b + old *d.
int goc_s_fmac_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t *excp_flag_user);

// Fused a * b + literal.
int goc_s_fmaak_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                    uint32_t literal, uint32_t *excp_flag_user);

// Fused a * literal + c.
int goc_s_fmamk_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t literal, uint32_t c, uint32_t *excp_flag_user);

// Scalar FP16/FP32 binary arithmetic executes once per wave and ignores EXEC,
// including zero EXEC; d is required. Inputs are raw IEEE bits. FP16 reads the
// low halves and writes a zero upper half. instruction_flags must be zero.
// Loose semantics only; NaN signs/payloads are unspecified. Requires host
// nearest-even rounding and enabled denormals. Preserves host rounding but may
// change exception flags. SCC is unchanged. Errors leave d unchanged.
// Guest input flushing applies to every operation. Guest output flushing applies
// to add/subtract/multiply; min/max ignore it. FP16_OVFL saturates finite FP16
// overflow. Multiplication flushes tiny results before destination subnormal
// rounding, so a value that would round up to the smallest normal may flush.

// Sum of a and b.
int goc_s_add_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *excp_flag_user);

// Sum of a and b.
int goc_s_add_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *excp_flag_user);

// Difference a - b.
int goc_s_sub_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *excp_flag_user);

// Difference a - b.
int goc_s_sub_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *excp_flag_user);

// Product of a and b.
int goc_s_mul_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *excp_flag_user);

// Product of a and b.
int goc_s_mul_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *excp_flag_user);

// Minimum; a single NaN selects the numeric operand; -0 sorts below +0.
int goc_s_min_num_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// Minimum; a single NaN selects the numeric operand; -0 sorts below +0.
int goc_s_min_num_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// Maximum; a single NaN selects the numeric operand; -0 sorts below +0.
int goc_s_max_num_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// Maximum; a single NaN selects the numeric operand; -0 sorts below +0.
int goc_s_max_num_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// Minimum; either NaN yields NaN; -0 sorts below +0.
int goc_s_minimum_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// Minimum; either NaN yields NaN; -0 sorts below +0.
int goc_s_minimum_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// Maximum; either NaN yields NaN; -0 sorts below +0.
int goc_s_maximum_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// Maximum; either NaN yields NaN; -0 sorts below +0.
int goc_s_maximum_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *excp_flag_user);

// Scalar bitfields and counts ignore EXEC, including zero EXEC; output pointers
// are required. Both loose and empirical exact semantics are supported, and
// instruction_flags must be zero. All host FP state is preserved. SCC outputs
// are written after d and win on overlap, including within a 64-bit destination.
// Errors leave all outputs unchanged. Instructions without SCC outputs preserve SCC.

// Extract unsigned field: offset=b[4:0], width=b[22:16].
// Width clips at bit 32; zero width yields zero. SCC is result != 0.
int goc_s_bfe_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

// Extract signed field: offset=b[4:0], width=b[22:16].
// Width clips at bit 32; zero width yields zero. SCC is result != 0.
int goc_s_bfe_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

// Extract unsigned field: offset=b[5:0], width=b[22:16].
// Width clips at bit 64; zero width yields zero. SCC is result != 0.
int goc_s_bfe_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint32_t b,
                  uint32_t *scc);

// Extract signed field: offset=b[5:0], width=b[22:16].
// Width clips at bit 64; zero width yields zero. SCC is result != 0.
int goc_s_bfe_i64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint32_t b,
                  uint32_t *scc);

// Low 32 bits of ((1 << (a modulo 32)) - 1) << (b modulo 32).
int goc_s_bfm_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b);

// Low 64 bits of ((1 << (a modulo 64)) - 1) << (b modulo 64).
int goc_s_bfm_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint32_t a, uint32_t b);

// Count zero bits in 32-bit a; SCC is result != 0.
int goc_s_bcnt0_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t *scc);

// Count zero bits in 64-bit a; SCC is result != 0.
int goc_s_bcnt0_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
                        uint32_t *scc);

// Count one bits in 32-bit a; SCC is result != 0.
int goc_s_bcnt1_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t *scc);

// Count one bits in 64-bit a; SCC is result != 0.
int goc_s_bcnt1_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
                        uint32_t *scc);

// Count trailing zero bits in 32-bit a; return UINT32_MAX for zero.
int goc_s_ctz_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// Count trailing zero bits in 64-bit a; return UINT32_MAX for zero.
int goc_s_ctz_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a);

// Count leading zero bits in 32-bit a; return UINT32_MAX for zero.
int goc_s_clz_i32_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// Count leading zero bits in 64-bit a; return UINT32_MAX for zero.
int goc_s_clz_i32_u64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a);

// Count leading sign bits in 32-bit a, including the sign bit;
// return UINT32_MAX for zero or all-ones input.
int goc_s_cls_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// Count leading sign bits in 64-bit a, including the sign bit;
// return UINT32_MAX for zero or all-ones input.
int goc_s_cls_i32_i64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a);

// Clear bit b modulo 32 in old *d.
int goc_s_bitset0_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t b);

// Clear bit b modulo 64 in old *d.
int goc_s_bitset0_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint32_t b);

// Set bit b modulo 32 in old *d.
int goc_s_bitset1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t b);

// Set bit b modulo 64 in old *d.
int goc_s_bitset1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint32_t b);

// Scalar bitwise and shift instructions ignore EXEC, including zero EXEC.
// Output pointers are required; all host FP state is preserved. Both loose and
// empirical exact semantics are supported. instruction_flags must be zero.
// SCC, when present, is written after d and wins on overlapping storage, including
// a 32-bit word within a 64-bit destination. Errors leave all outputs unchanged.

// 32-bit and; SCC is result != 0.
int goc_s_and_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

// 64-bit and; SCC is result != 0.
int goc_s_and_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint64_t b,
                  uint32_t *scc);

// 32-bit or; SCC is result != 0.
int goc_s_or_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                 uint32_t *scc);

// 64-bit or; SCC is result != 0.
int goc_s_or_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint64_t b,
                 uint32_t *scc);

// 32-bit xor; SCC is result != 0.
int goc_s_xor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

// 64-bit xor; SCC is result != 0.
int goc_s_xor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint64_t b,
                  uint32_t *scc);

// 32-bit nand; SCC is result != 0.
int goc_s_nand_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t *scc);

// 64-bit nand; SCC is result != 0.
int goc_s_nand_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint64_t b,
                   uint32_t *scc);

// 32-bit nor; SCC is result != 0.
int goc_s_nor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

// 64-bit nor; SCC is result != 0.
int goc_s_nor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint64_t b,
                  uint32_t *scc);

// 32-bit xnor; SCC is result != 0.
int goc_s_xnor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t *scc);

// 64-bit xnor; SCC is result != 0.
int goc_s_xnor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint64_t b,
                   uint32_t *scc);

// 32-bit and_not1; SCC is result != 0.
int goc_s_and_not1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

// 64-bit and_not1; SCC is result != 0.
int goc_s_and_not1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint64_t b, uint32_t *scc);

// 32-bit or_not1; SCC is result != 0.
int goc_s_or_not1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *scc);

// 64-bit or_not1; SCC is result != 0.
int goc_s_or_not1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                      uint64_t b, uint32_t *scc);

// 32-bit not; SCC is result != 0.
int goc_s_not_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                  uint32_t *scc);

// 64-bit not; SCC is result != 0.
int goc_s_not_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                  uint32_t *scc);

// 32-bit bit reversal; leaves SCC unchanged.
int goc_s_brev_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// 64-bit bit reversal; leaves SCC unchanged.
int goc_s_brev_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a);

// 32-bit left shift by b modulo 32; SCC is result != 0.
int goc_s_lshl_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t *scc);

// 64-bit left shift by b modulo 64; SCC is result != 0.
int goc_s_lshl_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint32_t b,
                   uint32_t *scc);

// 32-bit logical right shift by b modulo 32; SCC is result != 0.
int goc_s_lshr_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t *scc);

// 64-bit logical right shift by b modulo 64; SCC is result != 0.
int goc_s_lshr_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint32_t b,
                   uint32_t *scc);

// 32-bit arithmetic right shift by b modulo 32; SCC is result != 0.
int goc_s_ashr_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t *scc);

// 64-bit arithmetic right shift by b modulo 64; SCC is result != 0.
int goc_s_ashr_i64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint32_t b,
                   uint32_t *scc);

// Low 32 bits of (a << 1) + b; SCC indicates that the full sum exceeds UINT32_MAX.
int goc_s_lshl1_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc);

// Low 32 bits of (a << 2) + b; SCC indicates that the full sum exceeds UINT32_MAX.
int goc_s_lshl2_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc);

// Low 32 bits of (a << 3) + b; SCC indicates that the full sum exceeds UINT32_MAX.
int goc_s_lshl3_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc);

// Low 32 bits of (a << 4) + b; SCC indicates that the full sum exceeds UINT32_MAX.
int goc_s_lshl4_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc);

// Scalar-register integer arithmetic executes once per wave and ignores EXEC,
// including zero EXEC. Scalar output pointers are required. Both loose and
// empirical exact semantics are supported; instruction_flags must be zero.
// All host FP state is preserved. Results wrap to the destination width.
// SCC outputs contain 0 or 1 and are written after d; if they alias, SCC wins.
// Errors leave all outputs unchanged. Inputs passed by value may originate
// from destination storage. Instructions without SCC outputs leave SCC alone.

// Unsigned sum; SCC is carry-out.
int goc_s_add_co_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b, uint32_t *scc);

// Unsigned difference; SCC is borrow-out.
int goc_s_sub_co_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b, uint32_t *scc);

// Signed sum; SCC is signed overflow.
int goc_s_add_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b, uint32_t *scc);

// Signed difference; SCC is signed overflow.
int goc_s_sub_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b, uint32_t *scc);

// Unsigned sum with input_scc bit 0; SCC is carry-out.
int goc_s_add_co_ci_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc, uint32_t input_scc);

// Unsigned difference minus input_scc bit 0; SCC is borrow-out.
int goc_s_sub_co_ci_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc, uint32_t input_scc);

// RDNA3 scalar add/sub spellings. Unsigned SCC is carry/borrow; signed SCC
// is overflow. ADDC/SUBB consume input_scc bit 0. ADDK sign-extends immediate
// and adds to old *d. Both semantics are exact; flags and alias rules above apply.
int goc_s_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

int goc_s_sub_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

int goc_s_add_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

int goc_s_sub_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

int goc_s_addc_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t input_scc, uint32_t *scc);

int goc_s_subb_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t input_scc, uint32_t *scc);

int goc_s_addk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate,
                   uint32_t *scc);

// Sign-extend the low 8 bits to 32 bits; preserves SCC.
int goc_s_sext_i32_i8(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// Sign-extend the low 16 bits to 32 bits; preserves SCC.
int goc_s_sext_i32_i16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// Absolute value modulo 2^32; SCC is result != 0.
int goc_s_abs_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                  uint32_t *scc);

// Absolute value of the wrapped 32-bit difference; SCC is result != 0.
int goc_s_absdiff_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *scc);

// Signed minimum; SCC is a < b.
int goc_s_min_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

// Unsigned minimum; SCC is a < b.
int goc_s_min_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

// Signed maximum; SCC is a > b.
int goc_s_max_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

// Unsigned maximum; SCC is a > b.
int goc_s_max_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc);

// Low 32 bits of the product.
int goc_s_mul_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b);

// High 32 bits of the unsigned product.
int goc_s_mul_hi_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b);

// High 32 bits of the signed product.
int goc_s_mul_hi_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b);

// Sum modulo 2^64.
int goc_s_add_nc_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                     uint64_t b);

// Difference modulo 2^64.
int goc_s_sub_nc_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                     uint64_t b);

// Product modulo 2^64.
int goc_s_mul_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint64_t b);

// Add sign-extended immediate to old *d; SCC is signed overflow.
int goc_s_addk_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate,
                      uint32_t *scc);

// Multiply old *d by sign-extended immediate, modulo 2^32.
int goc_s_mulk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate);

// RDNA4 WAVE_EXCP_FLAG_USER status bits currently produced by GoC.
static const uint32_t GOC_EXCEPTION_INVALID = 1U << 0;
static const uint32_t GOC_EXCEPTION_INPUT_DENORM = 1U << 1;
static const uint32_t GOC_EXCEPTION_FLOAT_DIV0 = 1U << 2;
static const uint32_t GOC_EXCEPTION_OVERFLOW = 1U << 3;
static const uint32_t GOC_EXCEPTION_UNDERFLOW = 1U << 4;
static const uint32_t GOC_EXCEPTION_INEXACT = 1U << 5;
static const uint32_t GOC_EXCEPTION_INT_DIV0 = 1U << 6;

// Reciprocal with sticky integer divide-by-zero status. Supports ABS_A, NEG_A,
// OMOD and CLAMP. Input/output subnormals always flush, independently of guest
// flush flags. Active zero or subnormal inputs set INT_DIV0 unless CLAMP is set.
// Existing excp_flag_user bits, including a pre-existing INT_DIV0, survive.
// DPP8/DPP16 permutes A before source modifiers. DPP-filtered destination lanes
// add no exception; a zero-filled source in an active lane can set INT_DIV0.
// excp_flag_user is optional. Unlike the general no-overlap requirement, this
// instruction also permits it to alias any operand word: its initial value is
// read before VGPR stores and its final write takes precedence on overlap.
// Zero EXEC permits null VGPR pointers and leaves all outputs unchanged.
// Errors leave all outputs unchanged. Loose semantics only; strict exact is
// rejected. This instruction implements exception accumulation on all CPU paths.
int goc_v_rcp_iflag_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// Pseudo-scalar transcendental math on raw SGPR values. Executes once regardless
// of EXEC, including zero. a is a scalar value; d must always be writable.
// FP16 reads a's low half, ignores its high half, and zeros d's upper half.
// Supports ABS_A, NEG_A, OMOD and CLAMP, with ABS before NEG. No half selectors.
// Loose semantics only; strict empirical-exact requests fail without writing d.
// FP32 always flushes input/output denormals. FP16 follows the guest input/output
// flush flags and FP16_OVFL. Rounding precedes OMOD; nonzero OMOD flushes tiny
// values before and after scaling. A zero created by negative underflow retains
// its sign; pre-existing zero becomes positive with OMOD. CLAMP maps NaN and
// negative results to zero. Requires host nearest-even rounding and enabled
// denormals. Host rounding is preserved; exception flags may change.
int goc_v_s_exp_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);
int goc_v_s_exp_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);
int goc_v_s_log_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);
int goc_v_s_log_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);
int goc_v_s_rcp_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);
int goc_v_s_rcp_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);
int goc_v_s_rsq_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);
int goc_v_s_rsq_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user);
int goc_v_s_sqrt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t *excp_flag_user);
int goc_v_s_sqrt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t *excp_flag_user);

// Floating comparisons write a scalar condition mask (CMP) or replacement EXEC
// mask (CMPX) to d. Inactive bits are zero, including for empty EXEC; d must
// always be writable and may alias any input word. Zero EXEC permits null VGPR
// pointers. Errors leave d unchanged. A/B use one VGPR for FP16/FP32 or a low/high
// pair for FP64. Supports ABS_A/B, NEG_A/B and FP16 HIGH_A/B. ABS precedes NEG.
// FP16/FP32 support DPP8/DPP16 on A before modifiers and half selection; B
// stays in its original lane, and filtered result bits are zero. FP64 rejects DPP.
// Supports loose and empirical-exact semantics. GOC_FP_FLUSH_INPUT_DENORMALS
// flushes input subnormals to signed zero after modifiers; otherwise they are
// preserved. Signed zeros compare equal. NaNs make ordered relations false and
// their negations true. O/U test ordered/unordered. All host FP state is preserved.
// CLAMP selects signaling comparison: any NaN raises INVALID when reporting.
// Without CLAMP, only signaling NaNs raise INVALID. Comparison results are unchanged.
// Exact reporting accumulates INPUT_DENORM for
// finite operand pairs containing a preserved subnormal. Only participating
// lanes contribute, after DPP filtering. Loose mode leaves excp_flag_user unchanged.
int goc_v_cmp_lt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_eq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_le_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_gt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_lg_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_ge_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_o_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_u_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nge_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nlg_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_ngt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nle_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_neq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nlt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_lt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_eq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_le_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_gt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_lg_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_ge_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_o_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_u_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_nge_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_nlg_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_ngt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_nle_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_neq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_nlt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmp_lt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_eq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_le_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_gt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_lg_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_ge_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_o_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_u_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nge_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nlg_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_ngt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nle_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_neq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nlt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_lt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_eq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_le_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_gt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_lg_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_ge_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_o_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_u_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_nge_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_nlg_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_ngt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_nle_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_neq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_nlt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmp_lt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_eq_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_le_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_gt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_lg_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_ge_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_o_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_u_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nge_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nlg_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_ngt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nle_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_neq_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmp_nlt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_lt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_eq_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_le_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_gt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_lg_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_ge_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_o_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_u_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user);
int goc_v_cmpx_nge_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_nlg_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_ngt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_nle_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_neq_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);
int goc_v_cmpx_nlt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);

// RDNA3 constant integer comparisons. F writes zero; T writes participating
// EXEC after DPP filtering. CMP produces the condition mask; CMPX produces
// replacement EXEC in d. A/B values are ignored and their pointers may be null.
// Both semantics are exact. I32/U32 support DPP8/DPP16; I64/U64 reject DPP.
// All other instruction_flags are rejected. d is required even for zero EXEC.
int goc_v_cmp_f_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmp_t_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmpx_f_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmpx_t_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmp_f_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmp_t_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmpx_f_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmpx_t_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmp_f_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmp_t_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmpx_f_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmpx_t_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmp_f_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmp_t_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmpx_f_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cmpx_t_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);

// Integer comparisons write a scalar condition mask (CMP) or replacement EXEC
// mask (CMPX) to d. Inactive bits are zero, including for empty EXEC. d is always
// required and may alias any input word. Zero EXEC permits null VGPR pointers.
// A/B use one VGPR for 16/32 bits or a low/high pair for 64 bits. The 16-bit
// forms support HIGH_A/HIGH_B. The 16/32-bit forms support DPP8/DPP16 on A
// before half selection, with B in its original lane; filtered output bits are
// zero. The 64-bit forms reject DPP. Other instruction flags are rejected. Errors
// leave d unchanged. Supports loose and empirical-exact semantics. All host FP
// state is preserved; inputs are interpreted as two's-complement or unsigned.
int goc_v_cmp_lt_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_eq_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_le_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_gt_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ne_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ge_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_lt_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_eq_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_le_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_gt_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ne_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ge_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_lt_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_eq_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_le_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_gt_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ne_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ge_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_lt_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_eq_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_le_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_gt_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ne_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ge_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_lt_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_eq_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_le_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_gt_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ne_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ge_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_lt_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_eq_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_le_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_gt_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ne_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ge_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_lt_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_eq_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_le_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_gt_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ne_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ge_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_lt_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_eq_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_le_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_gt_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ne_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ge_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_lt_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_eq_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_le_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_gt_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ne_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ge_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_lt_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_eq_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_le_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_gt_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ne_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ge_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_lt_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_eq_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_le_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_gt_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ne_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_ge_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_lt_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_eq_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_le_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_gt_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ne_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_ge_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b);

// Floating-point class-mask bits; any combination may be supplied in B.
static const uint32_t GOC_CLASS_SNAN = 1U << 0;
static const uint32_t GOC_CLASS_QNAN = 1U << 1;
static const uint32_t GOC_CLASS_NEG_INF = 1U << 2;
static const uint32_t GOC_CLASS_NEG_NORMAL = 1U << 3;
static const uint32_t GOC_CLASS_NEG_SUBNORMAL = 1U << 4;
static const uint32_t GOC_CLASS_NEG_ZERO = 1U << 5;
static const uint32_t GOC_CLASS_POS_ZERO = 1U << 6;
static const uint32_t GOC_CLASS_POS_SUBNORMAL = 1U << 7;
static const uint32_t GOC_CLASS_POS_NORMAL = 1U << 8;
static const uint32_t GOC_CLASS_POS_INF = 1U << 9;

// Classify A's raw encoding and test the corresponding bit of B's class mask.
// CMP writes a scalar condition mask to d; CMPX writes the replacement EXEC
// mask to d. Inactive bits are zero, including for zero EXEC. d must always be
// writable and may alias any input word; all inputs are read before that write.
// Zero EXEC permits null VGPR pointers. Errors leave d unchanged.
// A uses one VGPR for FP16/FP32 or a low/high pair for FP64; B uses one VGPR.
// Supports ABS_A/NEG_A. FP16 also supports HIGH_A and HIGH_B, selecting source
// halves. Only B's low ten selected bits matter. Signaling NaNs are not quieted.
// FP16/FP32 support DPP8/DPP16 on A before modifiers and half selection;
// B stays in its lane and filtered result bits are zero. FP64 rejects DPP.
// Supports loose and empirical-exact semantics, independently of denormal
// controls. All host FP state is preserved.
int goc_v_cmp_class_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_class_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmp_class_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_class_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_class_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);
int goc_v_cmpx_class_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, const uint32_t *const *a, const uint32_t *const *b);

// Interpolation wait-count field (0..7). Accepted for every interpolation
// instruction; it has no effect on synchronous CPU execution.
static const uint32_t GOC_INTERP_WAIT_EXP_SHIFT = 13;
static const uint32_t GOC_INTERP_WAIT_EXP_MASK = 7U << 13;

// Quad-local interpolation. For lane L and Q=L&~3, P10 computes
// fma(A[Q+1],B[L],C[Q]); P2 computes fma(A[Q+2],B[L],C[L]). Source lanes are
// read even if inactive in EXEC. Each operand uses one VGPR; whole VGPRs may
// alias. Supports NEG_A/B/C, CLAMP and WAIT_EXP, with no ABS or OMOD.
// Loose semantics only. Requires host nearest-even rounding and denormals
// enabled; exception flags may change. NaN payloads are unspecified.
int goc_v_interp_p10_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);
int goc_v_interp_p2_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);

// Mixed FP16 interpolation uses the same quad broadcasts as the FP32 forms.
// A is FP16, B is FP32. P10 reads FP16 C and writes FP32 D; P2 reads FP32 C
// and writes FP16 D, preserving the other destination half. Supports NEG_A/B/C,
// CLAMP, WAIT_EXP and HIGH_A; also HIGH_C for P10 or HIGH_D for P2. No ABS/OMOD.
// RTZ forms round toward zero; other forms round nearest-even. GOC_FP16_OVFL
// saturates finite P2 overflow; RTZ P2 already saturates finite overflow.
// Input infinities remain infinite unless clamped. Loose semantics only;
// host nearest-even rounding and enabled denormals are required. Host rounding
// mode is preserved; exception flags may change. NaN payloads are unspecified.
int goc_v_interp_p10_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c);
int goc_v_interp_p2_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);
int goc_v_interp_p10_rtz_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);
int goc_v_interp_p2_rtz_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// Select B where the corresponding condition bit is set, A otherwise. Each
// operand uses one VGPR. ABS_A/B clear source sign bits, then NEG_A/B toggle
// them; all other payload bits, including signaling NaNs, are preserved.
// The B16 form also supports HIGH_A/B/D: selected source halves are written to
// the selected destination half, preserving the other half. No OMOD or CLAMP.
// DPP8/DPP16 permutes A before source modifiers and half selection; B and
// the condition mask remain in their original lanes.
// Loose semantics only; all host FP state is preserved.
int goc_v_cndmask_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t condition);
int goc_v_cndmask_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t condition);

// Select each output byte using the corresponding byte of C. Selectors 0..7
// select bytes of the concatenation A:B (B supplies the low four bytes).
// Selectors 8..11 replicate the sign bit of its four 16-bit halves; 12 selects
// zero, and 13..255 select 0xff. Each operand uses one VGPR. Supports DPP8/DPP16
// on A; low 32 instruction-flag bits must be zero. Loose semantics only; preserves
// all host FP state.
int goc_v_perm_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c);

// Trigonometric range-reduction table lookup. A is an FP64 VGPR pair, B is
// one integer VGPR (only its low five bits select the segment), and D is an
// FP64 VGPR pair. Only A's encoded exponent affects the lookup, including
// for infinities and NaNs. ABS_A/NEG_A are accepted and have no effect.
// Supports OMOD and CLAMP; OMOD flushes tiny results before and after scaling.
// Supports loose and empirical-exact semantics. Preserves all host FP state.
// If destination VGPRs alias, the high word is written last.
int goc_v_trig_preop_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user);

// Lighting multiply. After ABS/NEG source modifiers, return -FLT_MAX if B is
// -FLT_MAX, -infinity or NaN, or C is nonpositive or NaN. Otherwise return +0
// if either factor is zero, or A*B. OMOD scales the result, flushing tiny
// unscaled values to +0 and tiny scaled values to signed zero; CLAMP then
// maps to [0,1], including NaN/-0 to +0. Each operand uses one VGPR.
// Supports all A/B/C ABS/NEG, OMOD and CLAMP modifiers; loose semantics only.
// DPP8/DPP16 permute A before source modifiers. B and C retain their lanes;
// inactive or DPP-filtered destinations remain unchanged.
// Host nearest-even rounding with denormals enabled is required; exception
// flags may change. NaN payloads are unspecified.
int goc_v_mullit_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c, uint32_t *excp_flag_user);

// Saturate both signed I16 halves of A to U8, pack them low byte first, and
// write the selected half of D, preserving the other half. HIGH_D selects the
// upper destination half. DPP8/DPP16 permutes A before saturation; no other
// instruction modifiers apply. A/D use one
// VGPR each. Loose semantics only; all host FP state is preserved.
int goc_v_sat_pk_u8_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a);

// Pack selected FP16 bits from A into D's low half and B into its high half.
// HIGH_A/B select source halves; ABS_A/B clear their sign bits before NEG_A/B
// toggles them. Signaling NaNs are quieted; other payload bits and subnormals
// are preserved.
// DPP8/DPP16 permutes A before half selection; B stays in its original lane.
// No other instruction modifiers apply. Each operand uses one VGPR. Loose
// semantics only; all host FP state is preserved.
int goc_v_pack_b32_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user);

// Bitwise equivalence: D = ~(A ^ B). Each operand uses one VGPR. Supports
// DPP8/DPP16 on A; low 32 instruction-flag bits must be zero. Loose semantics
// only; host FP state is preserved.
int goc_v_xnor_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// Extract the low 32 bits of the concatenation A:B shifted right by C & 31
// bits (ALIGNBIT), or (C & 3) bytes (ALIGNBYTE). Each operand uses one VGPR.
// Supports DPP8/DPP16 on A; low 32 instruction-flag bits must be zero.
// Loose semantics only; host FP state is preserved.
int goc_v_alignbit_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c);

int goc_v_alignbyte_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);

// Sparse integer 2:4 WMMA with signed 32-bit in/out accumulator D (8 VGPRs).
// A/B use 2/4 VGPRs for K=32 IU8 and K=64 IU4, or 1/2 for K=32 IU4;
// index uses one VGPR. Metadata pairs must select strictly increasing positions.
// SIGNED_A/B select signed factors. CLAMP saturates after each of two product
// groups: compressed positions 0..7 then 8..15 for K=32; positions 0..7 and
// 16..23, then 8..15 and 24..31 for K=64. Without CLAMP results wrap modulo 2^32.
// K=32 accepts GOC_SWMMAC_INDEX_KEY_1; K=64 consumes all 32 metadata bits and
// rejects it. Supports loose and empirical exact semantics; host FP state is
// preserved. GOC_FP16_OVFL has no effect. All inputs and D are read before
// ascending destination-register stores; last store wins aliases. EXEC is ignored;
// every lane is read and written, and operand pointers must be valid.
// Errors leave destinations unchanged.
int goc_v_swmmac_i32_16x16x32_iu8(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                  const uint32_t *const *a, const uint32_t *const *b,
                                  const uint32_t *const *index);

int goc_v_swmmac_i32_16x16x32_iu4(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                  const uint32_t *const *a, const uint32_t *const *b,
                                  const uint32_t *const *index);

int goc_v_swmmac_i32_16x16x64_iu4(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                  const uint32_t *const *a, const uint32_t *const *b,
                                  const uint32_t *const *index);

// Sparse FP8/BF8 2:4 matrix multiply-accumulate into FP32 D. A uses two
// VGPRs, B four, index one, and in/out D eight. FP8 is E4M3FN; BF8 is E5M2.
// Metadata pairs must contain strictly increasing positions in each group of
// four. GOC_SWMMAC_INDEX_KEY_1 selects the upper 16 metadata bits per lane;
// no other instruction modifiers apply. Loose semantics use FP32 FMA with
// host nearest-even rounding and denormals enabled. Exception flags may change.
// Exact semantics are unsupported; GOC_FP16_OVFL has no effect. All inputs and
// D are read before ascending destination-register stores; last store wins
// aliases. EXEC is ignored; every lane is read and written, and operand pointers
// must be valid. Errors leave destinations unchanged.
int goc_v_swmmac_f32_16x16x32_fp8_fp8(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *index);

int goc_v_swmmac_f32_16x16x32_fp8_bf8(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *index);

int goc_v_swmmac_f32_16x16x32_bf8_fp8(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *index);

int goc_v_swmmac_f32_16x16x32_bf8_bf8(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *index);

// Sparse 2:4 matrix multiply-accumulate. D is both the initial accumulator and
// destination; A holds 16 compressed elements per row, B the dense 32x16 matrix,
// and index holds packed 2-bit positions. Within each four-element K group the
// two selected positions must be strictly increasing. A uses four VGPRs, B eight,
// index one, and D eight for FP32 or four for packed FP16/BF16 output.
// GOC_SWMMAC_INDEX_KEY_1 selects the upper instead of lower 16 metadata bits in
// each lane. Supports GOC_WMMA_NEG_LO_A/B and GOC_WMMA_NEG_HI_A/B; these negate
// the first/second member of each selected pair, including B after selection.
// No C modifiers or CLAMP apply. Loose semantics accumulate selected products
// with FP32 FMA, then round packed outputs to nearest-even. Host FP state must
// provide nearest-even rounding with denormals enabled; exception flags may change.
// GOC_FP16_OVFL saturates finite overflow when narrowing to FP16; otherwise it
// has no effect. Exact semantics are unsupported. All inputs and D are read
// before ascending destination-register stores; the last store wins aliases.
// EXEC is ignored; every lane is read and written, and operand pointers must
// be valid. Errors leave all destinations unchanged.
int goc_v_swmmac_f32_16x16x32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                  const uint32_t *const *a, const uint32_t *const *b,
                                  const uint32_t *const *index);

int goc_v_swmmac_f32_16x16x32_bf16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *index);

int goc_v_swmmac_f16_16x16x32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                  const uint32_t *const *a, const uint32_t *const *b,
                                  const uint32_t *const *index);

int goc_v_swmmac_bf16_16x16x32_bf16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                    const uint32_t *const *a, const uint32_t *const *b,
                                    const uint32_t *const *index);

// Multiply two 32-bit lanes and add a 64-bit accumulator. A/B use one VGPR;
// C/D use low/high pairs. The scalar output contains bit 64 of the full sum:
// unsigned carry for U64, or the sign of the mathematical 65-bit sum for I64.
// CLAMP saturates to the unsigned/signed 64-bit range without changing that mask.
// Supports CLAMP only, loose and empirical exact semantics, every EXEC mask and
// whole-register alias. All inputs are read before D0, then D1, then carry are
// written; D1 wins if D0/D1 alias. Inactive scalar bits are cleared, even for zero
// EXEC. carry must always be writable; zero EXEC permits null VGPR pointers.
// Errors leave all destinations unchanged. Host FP state is preserved and
// GOC_FP16_OVFL has no effect.
int goc_v_mad_co_u64_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c);

int goc_v_mad_co_i64_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c);

// RDNA3 spelling of unsigned 32x32+64 multiply-add. Same carry, CLAMP,
// aliasing and execution-mask contract as goc_v_mad_co_u64_u32.
int goc_v_mad_u64_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                      const uint32_t *const *b, const uint32_t *const *c);

// RDNA3 signed spelling. The scalar result is bit 64 of the full signed sum,
// with the same contract as goc_v_mad_co_i64_i32 (not a signed-overflow bit).
int goc_v_mad_i64_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                      const uint32_t *const *b, const uint32_t *const *c);

// Unsigned add/subtract with a scalar carry/borrow output. CI forms also consume
// one input carry/borrow bit per lane. SUBREV computes B-A-input_borrow.
// CLAMP saturates overflow to UINT32_MAX for addition and underflow to zero for
// subtraction; carry/borrow bits still describe the unsaturated result.
// Supports CLAMP and DPP8/DPP16: ADD/SUB permute A, SUBREV permutes B. Input
// carry bits stay in their original lanes. Supports loose and empirical exact
// semantics, all EXEC masks and whole-register aliases. Inactive and DPP-filtered
// scalar output bits are cleared, even at zero
// EXEC. The scalar output must always be writable and is written after VGPR D;
// zero EXEC permits null VGPR pointers. Errors leave all destinations unchanged.
// Host FP state is preserved; GOC_FP16_OVFL has no effect.
int goc_v_add_co_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                     const uint32_t *const *b);

int goc_v_sub_co_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                     const uint32_t *const *b);

int goc_v_subrev_co_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                        const uint32_t *const *b);

int goc_v_add_co_ci_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                        const uint32_t *const *b, uint32_t input_carry);

int goc_v_sub_co_ci_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                        const uint32_t *const *b, uint32_t input_carry);

int goc_v_subrev_co_ci_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                           const uint32_t *const *b, uint32_t input_carry);

// Fused division post-scaling: compute A*B+C and apply a power-of-two scale
// before the single nearest-even rounding. A set lane bit in condition (the
// implicit wave32 VCC input) selects +64/+128 when C's modified encoded exponent
// exceeds its bias, and -64/-128 otherwise, for FP32/FP64 respectively.
// Supports all source ABS/NEG, OMOD and CLAMP, loose and empirical exact semantics.
// Inputs preserve denormals. Active OMOD rounds at normal precision before
// flushing tiny results to +0, then applies the output scale and clamp.
// FP64 operands use low/high VGPR pairs, with D1 winning when D0/D1 alias.
// Supports every EXEC mask and whole-register alias; zero effective EXEC permits
// null VGPR pointers. Results are independent of host FP state and preserve it.
// GOC_FP16_OVFL has no effect.
int goc_v_div_fmas_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t condition, uint32_t *excp_flag_user);

int goc_v_div_fmas_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t condition, uint32_t *excp_flag_user);

// Division pre-scaling. A must equal B (denominator) or C (numerator) after
// source NEG modifiers. Writes the pre-scaled value to D and the per-lane
// post-scaling condition to condition (the wave32 SDST operand).
// Supports source NEG, OMOD and CLAMP; ABS is not encoded by these instructions.
// Condition bits for inactive lanes are cleared. Even zero EXEC writes zero to
// condition, so that pointer must always be writable; VGPR pointers may then be
// null. The condition is written after VGPR outputs. Errors leave both unchanged.
// FP32 uses one VGPR per operand; FP64 uses low/high pairs, with D1 winning if
// D0/D1 alias. Supports all whole-register aliases, loose and empirical exact
// semantics, nearest-even rounding and denormals. Results preserve host FP state
// and do not depend on it. GOC_FP16_OVFL has no effect.
int goc_v_div_scale_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);

int goc_v_div_scale_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);

// Division fixup: A is a provisional quotient, B the original denominator,
// C the original numerator. Repairs sign, propagates C/B NaNs in that order,
// and handles zero/infinity cases and extreme FP32/FP64 exponent underflow.
// FP16/FP32 operands hold one VGPR each; FP64 operands use low/high VGPR pairs.
// FP16 supports HIGH_A/B/C/D and preserves the unselected D half. FP16 also
// supports DPP8/DPP16 on A before half selection and source modifiers; B/C
// retain their lanes. FP32/FP64 reject DPP. FP64 writes
// D[0] before D[1], so D[1] wins if both destination pointers alias.
// Supports all source ABS/NEG, OMOD and CLAMP. Nonzero OMOD flushes subnormals
// before scaling, maps existing zeros to +0 and preserves signed underflow zero.
// FP16_OVFL saturates FP16 provisional-quotient overflow before OMOD and finite
// scaling overflow; exceptional B/C cases retain their infinity/NaN behavior.
// Loose and empirical exact semantics use fixed nearest-even rounding with
// input/output denormals enabled. All EXEC masks and whole-register aliases
// are supported; results are independent of host FP state and preserve it.
int goc_v_div_fixup_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_div_fixup_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_div_fixup_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user);

// Cube-map face ID, S/T coordinates and signed doubled major axis, respectively.
// A/B/C contain X/Y/Z in one VGPR each; D holds one VGPR. Z wins magnitude ties,
// then Y, then X. Comparisons flush subnormals; SC/TC copy selected source bits
// and quiet NaNs. A zero or unordered major axis is treated as nonnegative.
// MA doubles the signed major axis with nearest-even overflow and +0 for zeros.
// DPP8/DPP16 permutes A before source modifiers; B/C retain their lanes.
// Supports all source ABS/NEG, OMOD and CLAMP. Nonzero OMOD flushes subnormal
// inputs/outputs, maps input zeros to +0 and preserves NaN sign/payload.
// Supports loose and empirical exact semantics.
// Full EXEC masking and all whole-register aliases are supported. Host FP state
// is preserved and does not affect results; GOC_FP16_OVFL has no effect.
int goc_v_cubeid_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c);

int goc_v_cubesc_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c);

int goc_v_cubetc_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c);

int goc_v_cubema_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c);

// DPP8/DPP16 permute A before modifiers; B remains in its original lane,
// including when B is the stochastic seed.
// Narrow FP32 to OCP E4M3FN (FP8) / E5M2 (BF8). Each operand is one VGPR.
// PK rounds A/B to nearest-even into the low/high bytes of the destination half
// selected by HIGH_D, preserving the other half. Supports A/B ABS/NEG.
// SR converts A using B as the stochastic seed, replaces D's GOC_CVT_BYTE_*
// byte and preserves the others. Supports A ABS/NEG. Subnormal SR alignment
// discards low significand bits before adding the seed's high 20/21 bits.
// GOC_FP16_OVFL saturates finite overflow to signed max finite; infinities
// remain signed FP8 NaNs / BF8 infinities. Input NaNs become 0xff / 0xfe.
// Supports loose semantics, full EXEC masking and every whole-register alias.
// Results do not depend on host rounding and preserve all host FP state.
int goc_v_cvt_pk_fp8_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cvt_pk_bf8_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cvt_sr_fp8_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cvt_sr_bf8_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// DPP8/DPP16 permute A before nibble selection.
// Convert A's signed low nibble to FP32 divided by 16, then apply OMOD/CLAMP.
// Each operand holds one VGPR; higher source bits are ignored. Supports loose
// semantics, full EXEC masking and whole-register aliasing. Results are exact
// under all host rounding modes, preserving host FP state.
int goc_v_cvt_off_f32_i4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a);

// DPP8/DPP16 permute A; byte selection B and preserved bytes C stay in their
// original lanes. Convert FP32 A to an unsigned byte with nearest-even rounding, saturation
// to [0,255] and NaN-to-zero. Insert it into byte (B & 3) of C and write D;
// each operand holds one VGPR. Supports A ABS/NEG; CLAMP is accepted without
// numeric effect. Loose semantics, full EXEC masking and all whole-register
// aliases are supported. Host rounding does not affect results; exceptions may
// change. GOC_FP16_OVFL has no effect.
int goc_v_cvt_pk_u8_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user);

// DPP8/DPP16 permute A before half selection or saturation; B stays in its
// original lane. Integer widening and saturating packing use one VGPR per
// operand. Widening selects the low/high A half with HIGH_A, then sign- or zero-extends to D.
// Packing saturates each 32-bit source to the signed/unsigned 16-bit range,
// placing A in D's low half and B in its high half; no numeric modifiers
// are accepted.
// Supports loose semantics, full EXEC masking and whole-register aliases.
// Preserves all host FP state. GOC_FP16_OVFL has no effect.

int goc_v_cvt_i32_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_u32_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_pk_i16_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_cvt_pk_u16_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// DPP8/DPP16 permute A before modifiers and half selection; B stays in its
// original lane. Normalized conversions scale by 32767 (signed) or 65535 (unsigned), round
// once to nearest-even, and saturate to [-32767,32767] or [0,65535]. NaNs map
// to zero. All operands hold one VGPR. Packed forms put A/B in D's low/high
// halves; unary forms preserve the unselected D half. Floating sources support
// ABS/NEG; FP16 sources support HIGH_A/B, unary destinations support HIGH_D.
// CLAMP is accepted without numeric effect; unary forms also accept and ignore
// OMOD. GOC_FP16_OVFL has no effect. Supports loose semantics, all EXEC masks
// and whole-register aliases. Host FP exception flags may change.

int goc_v_cvt_pk_norm_i16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int goc_v_cvt_pk_norm_u16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int goc_v_cvt_pk_norm_i16_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *excp_flag_user);

int goc_v_cvt_pk_norm_u16_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *excp_flag_user);

int goc_v_cvt_norm_i16_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_norm_u16_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

// DPP8/DPP16 permute A before conversion and modifiers; B stays in its
// original lane. Pack two FP32 source VGPRs into one 16-bit-pair destination: A goes to the
// low half, B to the high half. Supports ABS/NEG on A/B, full EXEC masking and
// whole-register aliases. Loose semantics only. CLAMP is accepted without
// numeric effect. The FP16 RTZ form also accepts and ignores OMOD, truncates
// toward zero, saturates finite overflow and quiets NaNs while retaining payload
// bits. Integer forms truncate and saturate to the destination range, with NaNs
// mapping to zero. GOC_FP16_OVFL has no effect. Results do not depend on host
// rounding mode; host FP exception flags may change.

int goc_v_cvt_pk_rtz_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user);

int goc_v_cvt_pk_i16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user);

int goc_v_cvt_pk_u16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user);

// FP8/BF8 single-result conversion byte selector, an enumeration encoded in
// instruction_flags bits 16-17. Byte 0 is the least significant byte of A for
// widening, or of D for stochastic narrowing.
static const uint32_t GOC_CVT_BYTE_0 = 0U << 16;
static const uint32_t GOC_CVT_BYTE_1 = 1U << 16;
static const uint32_t GOC_CVT_BYTE_2 = 2U << 16;
static const uint32_t GOC_CVT_BYTE_3 = 3U << 16;

// Single-result forms support DPP8/DPP16, permuting A before byte selection.
// Packed two-result forms have no RDNA4 DPP encoding and reject DPP flags.
// FP8 (OCP E4M3FN) / BF8 (OCP E5M2) to FP32, loose semantics. A holds one
// VGPR. Single-result forms select one byte with GOC_CVT_BYTE_* and write D[0].
// Packed forms select the low/high A half with GOC_ALU_HIGH_A and write its
// two bytes to D[0]/D[1], respectively. If D halves alias, D[1] wins. Supports
// whole-register source/destination aliases and EXEC masking. No ABS/NEG,
// OMOD or CLAMP. Finite results are exact; NaNs become sign-preserving quiet
// NaNs. All host rounding modes give the same bits and preserve FP state.

int goc_v_cvt_f32_fp8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_f32_bf8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_pk_f32_fp8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_pk_f32_bf8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a);

// DPP8/DPP16 permute A before byte selection.
// Unsigned byte-to-FP32 conversions: one VGPR per operand. The mnemonic's
// byte index selects bits [8*index, 8*index+7] of A. Supports OMOD and CLAMP;
// source ABS/NEG and half selectors are invalid. All results are exactly
// representable, independent of host rounding, and preserve host FP state.
// Supports loose semantics, full EXEC masking and whole-register A/D aliasing.

int goc_v_cvt_f32_ubyte0(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_f32_ubyte1(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_f32_ubyte2(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_f32_ubyte3(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a);

// FP16 conversions, one VGPR per operand; loose semantics only. HIGH_A selects
// a 16-bit source half, HIGH_D selects a 16-bit destination half, preserving the
// other half. The corresponding selector is invalid for a full FP32 operand.
// Floating inputs support ABS/NEG; integer inputs reject them. Floating outputs
// support OMOD/CLAMP. FP16 results round before OMOD, using nearest-even and
// FP16_OVFL at both rounding stages. Nonzero OMOD flushes initially tiny
// results and either initial zero to +0; newly tiny scaled results become
// signed zero. NaNs are quieted. Integer outputs
// truncate, saturate, map NaNs to zero, and ignore CLAMP/OMOD numerically.
// All six forms support DPP8/DPP16 on A before source-half selection.

int goc_v_cvt_f16_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_cvt_f16_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_cvt_i16_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_cvt_u16_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_cvt_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_cvt_f32_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// FP64 conversions: FP64 operands use two VGPRs, low word first; FP32 and
// integer operands use one. Loose semantics only. Integer inputs reject ABS/NEG;
// floating inputs support ABS/NEG. Floating outputs round to the destination
// format before OMOD and CLAMP. Host nearest-even rounding and enabled denormals
// are required. Integer outputs truncate, saturate overflow, map NaNs to zero,
// and accept CLAMP/OMOD without numeric effect; GPU exceptions are not modeled.
// Destination halves may alias each other; the high word wins in active lanes.

int goc_v_cvt_f64_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_f64_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_i32_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_cvt_u32_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_cvt_f64_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_cvt_f32_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// FP32/integer conversions, one VGPR per operand. Loose semantics only.
// Integer-to-FP32 conversion uses host nearest-even rounding, followed by OMOD
// and CLAMP; source ABS/NEG are invalid. Float-to-integer conversion saturates
// to the destination range and supports ABS/NEG. Truncating forms map NaNs to
// zero; nearest/floor forms map them to the signed limit selected by the NaN sign
// after source modifiers. CLAMP is accepted but does not affect integer results.
// Truncating forms also accept OMOD without numeric scaling. GPU exceptions are
// not modeled.
// All six forms support DPP8/DPP16 on A.

int goc_v_cvt_f32_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a);

int goc_v_cvt_f32_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a);

// Convert toward zero.
int goc_v_cvt_i32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_cvt_u32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// Round to nearest integer, breaking ties toward positive infinity.
int goc_v_cvt_nearest_i32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a);

// Round toward negative infinity.
int goc_v_cvt_floor_i32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a);

// Combined three-input integer operations, one VGPR per operand. Shift
// counts wrap modulo 32; all addition and shifting wrap to 32 bits. Supports
// DPP8/DPP16 on A; low 32 instruction-flag bits must be zero. Loose semantics
// only; preserves all host FP state.

// (A << (B & 31)) + C.
int goc_v_lshl_add_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c);

// (A + B) << (C & 31).
int goc_v_add_lshl_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c);

// (A << (B & 31)) | C.
int goc_v_lshl_or_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c);

// (A & B) | C.
int goc_v_and_or_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c);

// A | B | C.
int goc_v_or3_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);

// A ^ B ^ C.
int goc_v_xor3_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c);

// (A ^ B) + C.
int goc_v_xad_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);

// Each result byte is floor((A_byte + B_byte + (C_byte & 1)) / 2).
// Higher C-byte bits are ignored. Carries do not cross byte boundaries.
int goc_v_lerp_u8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);

// Sum of unsigned absolute differences between packed fields of A and B, plus
// C. SAD_U8 sums four byte differences, SAD_U16 two halfword differences, and
// SAD_U32 one full-word difference. SAD_HI_U8 shifts the byte sum left by 16
// before accumulation. MSAD_U8 omits differences where the corresponding B
// byte is zero. These forms use one VGPR for each operand.
// GOC_ALU_CLAMP saturates the final unsigned accumulation. DPP8/DPP16 permute
// A before arithmetic; B and C stay in their original lanes. Other instruction
// flags are invalid. Loose semantics only; preserves all host FP state.
int goc_v_sad_u8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);
int goc_v_sad_hi_u8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                    const uint32_t *const *c);
int goc_v_sad_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);
int goc_v_sad_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);
int goc_v_msad_u8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);

// Quad SAD compares four overlapping four-byte windows of A (starting at
// byte offsets 0, 1, 2 and 3) against B. A has two VGPRs, low word first; B has
// one. QSAD uses all B bytes; MQSAD omits zero B bytes. PK_U16 packs four
// independent 16-bit accumulations into two C/D VGPRs. U32 uses four C/D VGPRs.
// GOC_ALU_CLAMP saturates each accumulation to its destination width; otherwise
// each wraps independently. The high byte of A's second VGPR is unused.
// All other instruction flags are invalid. Loose semantics only; preserves
// all host floating-point state.
int goc_v_qsad_pk_u16_u8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);
int goc_v_mqsad_pk_u16_u8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);
int goc_v_mqsad_u32_u8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c);

// Reverse shifts: shift B by the count in A. A is one VGPR. B and D are one
// VGPR for 32-bit forms or two VGPRs (low word first) for 64-bit forms. Counts
// wrap modulo 32 or 64. ASHR replicates the sign bit; logical shifts insert
// zero bits. The 32-bit forms support DPP8/DPP16 on A (the shift count); their
// low 32 instruction-flag bits must be zero. The 64-bit forms require zero
// instruction_flags. Loose semantics only; preserves all host floating-point state.
int goc_v_lshlrev_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);
int goc_v_lshrrev_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);
int goc_v_ashrrev_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);
int goc_v_lshlrev_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);
int goc_v_lshrrev_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);
int goc_v_ashrrev_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// FP32 sine/cosine of inputs measured in turns: sin(2*pi*A), cos(2*pi*A).
// Supports ABS_A, NEG_A, OMOD and CLAMP. Empirical exact semantics use the
// captured RDNA3/4 integer model with denormals preserved and NaNs quieted;
// all host floating-point state is preserved. Output scaling uses nearest-even
// rounding, followed by CLAMP. Active OMOD flushes subnormal outputs and both
// signed zeros to +0. Loose SIMD semantics approximate the captured polynomial
// and require host nearest-even rounding with denormals enabled.
// DPP8/DPP16 permutes A before source modifiers.
int goc_v_sin_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);
int goc_v_cos_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// FP16 sine/cosine of inputs measured in turns. Supports ABS_A, NEG_A, OMOD,
// CLAMP and HIGH_A/D; preserves the unselected D half. Rounds to FP16 before
// OMOD. Active OMOD flushes tiny inputs to scaling and tiny scaled results to
// +0. Empirical exact semantics use rocjitsu's captured model with denormals
// preserved; all host floating-point state is preserved.
// Loose SIMD semantics require host nearest-even rounding with denormals enabled.
// GOC_FP16_OVFL is accepted and has no effect on the bounded finite results.
// DPP8/DPP16 permutes A before source modifiers and half selection.
int goc_v_sin_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);
int goc_v_cos_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// FP16 fused multiply-add: selected halves of A/B/C/D, all ALU source/output
// modifiers and GOC_FP16_OVFL. Preserves the other D half and inactive lanes.
// Arithmetic rounds to FP16 before OMOD; active OMOD flushes tiny arithmetic
// results to +0 and newly tiny scaled results to signed zero. CLAMP is last.
// Exact empirical semantics follow rocjitsu's RNE/denormal-preserving model,
// including NaN payload priority, and preserve the host floating-point environment.
// Loose semantics require host nearest-even arithmetic with denormals enabled.
int goc_v_fma_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c, uint32_t *excp_flag_user);

// FMA supports all GOC_ALU source/output modifiers. Exact semantics requests
// fall back to loose unless GOC_SEMANTICS_STRICT is set.
int goc_v_fma_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c, uint32_t *excp_flag_user);

// FP16 fused multiply-accumulate into D. Supports A/B ABS/NEG, OMOD, CLAMP,
// HIGH_A/B/D and GOC_FP16_OVFL, with the same rounding and exact-semantics
// contract as FP16 FMA. HIGH_D selects both the accumulator and result half;
// preserves the other half. C modifiers (including HIGH_C) are invalid.
int goc_v_fmac_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   uint32_t *excp_flag_user);

// FP32 fused multiply-accumulate into D. Supports A/B ABS/NEG, OMOD and CLAMP.
// C modifiers and half selectors are invalid. Loose semantics only.
int goc_v_fmac_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   uint32_t *excp_flag_user);

// Literal FP16 FMA: FMAMK computes A * literal + B; FMAAK computes A * B +
// literal. The literal is a raw FP16 encoding. HIGH_A/B/D select the two VGPR
// inputs and destination half; the other D half is preserved. Other instruction
// flags are invalid. Supports GOC_FP16_OVFL and empirical exact semantics with
// the FP16 FMA rounding and host-environment contract.
int goc_v_fmamk_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                    const uint32_t *const *b, uint32_t *excp_flag_user);

int goc_v_fmaak_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                    uint16_t literal, uint32_t *excp_flag_user);

// Literal FP32 FMA: FMAMK computes A * literal + B; FMAAK computes A * B +
// literal. The literal is a raw FP32 encoding. No instruction flags are valid.
// Loose semantics only; requires host nearest-even rounding and denormals enabled.
int goc_v_fmamk_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                    const uint32_t *const *b, uint32_t *excp_flag_user);

int goc_v_fmaak_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                    uint32_t literal, uint32_t *excp_flag_user);

// DX9 FMA: one VGPR per operand, all ALU source/output modifiers, loose semantics.
// Flush source/result subnormals. If either modified factor is signed zero,
// add a positive-zero product to modified C before output scaling/CLAMP;
// otherwise compute a fused multiply-add. DPP8/DPP16 permute A before source
// modifiers; B and C retain their lanes.
int goc_v_fma_dx9_zero_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

// RDNA3 DX9 fused multiply-accumulate into D, with the same flushing and
// zero-product rules as DX9 FMA. Supports A/B ABS/NEG, OMOD, CLAMP and DPP
// routing of A. C modifiers and half selectors are invalid. Loose semantics only.
int goc_v_fmac_dx9_zero_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// Packed FP16 FMA: two independent fused results per lane. All GOC_PK_* flags
// below are supported. With no flags, corresponding input halves are multiplied
// and added. Each result rounds to FP16; CLAMP applies last. Supports
// GOC_FP16_OVFL and empirical exact semantics with the FP16 FMA environment
// contract. Both halves use the original inputs, including when D aliases A/B/C.
int goc_v_pk_fma_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c, uint32_t *excp_flag_user);

// Packed FP16 FMAC: component-wise A * B + D. No instruction flags are valid.
// Supports GOC_FP16_OVFL and the same loose/exact semantics as packed FMA.
int goc_v_pk_fmac_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

// Packed FP16 binary arithmetic: supports GOC_PK_* negation and half selectors
// for A/B, plus GOC_PK_CLAMP and GOC_FP16_OVFL. Flags for C are invalid. Each
// result rounds to FP16; the two results use original inputs even with aliases.
// Number min/max ignore a lone NaN; minimum/maximum propagate NaNs. Both order
// -0 below +0. Loose semantics require host nearest-even with denormals enabled.
int goc_v_pk_add_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     uint32_t *excp_flag_user);

int goc_v_pk_mul_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     uint32_t *excp_flag_user);

int goc_v_pk_min_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user);

int goc_v_pk_max_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user);

int goc_v_pk_minimum_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user);

int goc_v_pk_maximum_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user);

// Ordinary 16-bit integer binary arithmetic: one VGPR per operand. HIGH_A/B/D
// select the input and output halves; the other destination half is preserved.
// ADD/SUB accept GOC_ALU_CLAMP for signed/unsigned saturation; otherwise they
// wrap. Other forms do not accept CLAMP. Multiply retains its low 16 bits.
// Shifts use B as the value and A modulo 16 as the count. All forms support
// DPP8/DPP16 on A before half selection. Other instruction flags are invalid.
// Loose semantics only; independent of host FP state.
int goc_v_add_nc_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_sub_nc_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_add_nc_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_sub_nc_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_min_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_max_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_min_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_max_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_mul_lo_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_lshlrev_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_lshrrev_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_ashrrev_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// Integer multiply-add with 32-bit C/D: one VGPR per operand. The 16-bit
// forms support HIGH_A/B; the 24-bit forms discard A/B's upper byte. Signed
// forms sign-extend the selected factors and interpret C as signed 32-bit.
// CLAMP saturates the full product plus C to the result's 32-bit range;
// otherwise results wrap. Supports DPP8/DPP16 on A before half selection. Other
// instruction flags are invalid. Loose semantics only; independent of host FP
// state. Whole-register aliases are supported.
int goc_v_mad_u32_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c);

int goc_v_mad_i32_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c);

int goc_v_mad_u32_u24(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c);

int goc_v_mad_i32_i24(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c);

// Ordinary 16-bit ternary arithmetic: one VGPR per operand. HIGH_A/B/C/D
// select source and destination halves; the other D half is preserved.
// MAD supports CLAMP after full-precision A * B + C, otherwise wrapping.
// MIN3/MAX3 select the smallest/largest of all three signed/unsigned inputs.
// MED3 selects the middle value. CLAMP is invalid for MIN3/MAX3/MED3.
// Supports DPP8/DPP16 on A before half selection; other instruction flags are
// invalid. Loose semantics only; independent of host floating-point state.
int goc_v_mad_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);

int goc_v_mad_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);

int goc_v_min3_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c);

int goc_v_min3_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c);

int goc_v_max3_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c);

int goc_v_max3_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c);

int goc_v_med3_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c);

int goc_v_med3_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c);

// Packed integer binary arithmetic: one VGPR per operand, two 16-bit results
// per lane. Supports GOC_PK_* half selectors for A/B and GOC_PK_CLAMP. CLAMP
// saturates ADD/SUB to the signed/unsigned 16-bit range; min/max and multiply
// ignore it. Without CLAMP, ADD/SUB wrap. Multiply keeps the low 16 bits.
// Negation and C flags are invalid. Both results read original inputs even
// when D aliases A/B. Loose semantics only; independent of host FP state.
int goc_v_pk_add_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_pk_sub_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_pk_add_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_pk_sub_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_pk_min_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_pk_max_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_pk_min_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_pk_max_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_pk_mul_lo_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// Packed 16-bit shifts: A supplies counts modulo 16; B supplies values.
// Supports GOC_PK_* half selectors for A/B. CLAMP is accepted and ignored;
// negation and C flags are invalid. ASHR sign-extends each selected B half.
// Two results per lane, one VGPR per operand, loose semantics only.
int goc_v_pk_lshlrev_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_pk_lshrrev_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_pk_ashrrev_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// Packed integer multiply-add: two results per lane, one VGPR per operand.
// All GOC_PK_* half selectors for A/B/C are supported; negation is invalid.
// CLAMP saturates the full A * B + C result to the signed/unsigned 16-bit range;
// without it, results wrap. Both halves read original sources before D is written.
// Loose semantics only; independent of host floating-point state.
int goc_v_pk_mad_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c);

int goc_v_pk_mad_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c);

// Packed source negation: independent for the low and high result calculations.
static const uint32_t GOC_PK_NEG_LO_A = 1U << 0;
static const uint32_t GOC_PK_NEG_LO_B = 1U << 1;
static const uint32_t GOC_PK_NEG_LO_C = 1U << 2;
static const uint32_t GOC_PK_NEG_HI_A = 1U << 3;
static const uint32_t GOC_PK_NEG_HI_B = 1U << 4;
static const uint32_t GOC_PK_NEG_HI_C = 1U << 5;

// Packed output clamp: floats to [0, 1], with NaNs and -0 to +0;
// integer ADD/SUB/MAD saturate to their result range.
static const uint32_t GOC_PK_CLAMP = 1U << 6;

// Packed half selectors flip the default choice for each result calculation.
// Zero flags use low inputs for the low result and high inputs for the high one.
// These flags can swap or replicate halves independently for each source.
static const uint32_t GOC_PK_LO_A_HIGH = 1U << 7;
static const uint32_t GOC_PK_LO_B_HIGH = 1U << 8;
static const uint32_t GOC_PK_LO_C_HIGH = 1U << 9;
static const uint32_t GOC_PK_HI_A_LOW = 1U << 10;
static const uint32_t GOC_PK_HI_B_LOW = 1U << 11;
static const uint32_t GOC_PK_HI_C_LOW = 1U << 12;

// Floating ALU source modifiers: ABS precedes NEG.
static const uint32_t GOC_ALU_NEG_A = 1U << 0;
static const uint32_t GOC_ALU_NEG_B = 1U << 1;
static const uint32_t GOC_ALU_NEG_C = 1U << 2;
static const uint32_t GOC_ALU_ABS_A = 1U << 3;
static const uint32_t GOC_ALU_ABS_B = 1U << 4;
static const uint32_t GOC_ALU_ABS_C = 1U << 5;

// Floating ALU output scaling precedes CLAMP. OMOD is a two-bit enumeration:
// none, multiply by 2, multiply by 4, divide by 2.
static const uint32_t GOC_ALU_OMOD_2 = 1U << 6;
static const uint32_t GOC_ALU_OMOD_4 = 2U << 6;
static const uint32_t GOC_ALU_OMOD_HALF = 3U << 6;

// Result clamping where supported: floating results to [0,1] (NaNs become zero),
// integer results to the representable signed/unsigned destination range.
static const uint32_t GOC_ALU_CLAMP = 1U << 8;

// Binary FP32 arithmetic: one VGPR per operand. OMOD flushes unscaled tiny
// results to +0; halving a normal result below twice minimum normal gives
// signed zero. Supports ABS/NEG for A/B, OMOD, CLAMP, and DPP8/DPP16.
// Flags for C and half selection are invalid. Loose semantics only.
int goc_v_add_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_sub_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_subrev_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     uint32_t *excp_flag_user);

int goc_v_mul_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

// FP32 min/max: one VGPR per operand; supports A/B ABS/NEG, OMOD and CLAMP.
// Loose semantics only. Number variants prefer numeric operands over NaNs;
// minimum/maximum propagate NaNs, preferring signaling NaNs and quieting them.
// Both families order -0 below +0.
int goc_v_min_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

int goc_v_max_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

int goc_v_minimum_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

int goc_v_maximum_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

// RDNA3 floating min/max/median spellings. Share the corresponding *_num_*
// loose implementations, operand widths, modifiers, alias rules and SIMD paths.
// Number selection ignores a lone NaN and orders -0 below +0; exact IEEE-mode
// signaling-NaN behavior is not implemented. Strict exact requests fail; non-null
// exception reporting in non-loose semantics fails with UNSUPPORTED_GLOBAL_STATE.
// Loose semantics leave excp_flag_user unchanged. Binary F64 and packed forms
// reject DPP; the other forms route A with DPP8/DPP16.
int goc_v_min_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_max_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_min_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_max_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_min_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_max_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_min3_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_max3_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_minmax_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_maxmin_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_med3_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_min3_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_max3_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_minmax_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_maxmin_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_med3_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_pk_min_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     uint32_t *excp_flag_user);

int goc_v_pk_max_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     uint32_t *excp_flag_user);

// Three-input FP32 min/max: one VGPR per operand; supports all A/B/C ABS/NEG,
// OMOD and CLAMP, with loose semantics. First select between A/B, then between
// that result and C; output modifiers apply only after both selections.
// MIN3/MAX3 repeat the same selection; MINMAX/MAXMIN apply opposite selections.
// Number/propagating variants follow the binary NaN and signed-zero rules above.
int goc_v_min3_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_max3_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_minmax_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_maxmin_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_minimum3_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_maximum3_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_minimummaximum_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_maximumminimum_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user);

// DX9 multiplication: one VGPR per operand, with A/B ABS/NEG, OMOD and CLAMP.
// Either signed-zero input forces a positive-zero product, including with NaN
// or infinity as the other input. Loose semantics only.
int goc_v_mul_dx9_zero_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *excp_flag_user);

// FP32 fractional part: x - floor(x), capped at the largest FP32 value below one
// before output scaling/CLAMP. Supports A ABS/NEG, OMOD and CLAMP, loose semantics.
// One VGPR per operand; infinite inputs produce NaN.
int goc_v_fract_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// FP32 median: one VGPR per operand; all A/B/C ABS/NEG, OMOD and CLAMP, loose
// semantics. Any NaN selects minimumNumber across all three inputs. Otherwise
// remove the first input numerically equal to the maximum and select the maximum
// of the remaining two inputs, following the ISA's signed-zero tie behavior.
int goc_v_med3_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

// FP64 arithmetic: two VGPRs per operand, holding each lane's low/high words.
// Supports ALU ABS/NEG on present sources, OMOD and CLAMP, with loose semantics.
// Host nearest-even rounding and enabled denormals are required. Whole VGPR
// aliases may cross operand halves; sources are read before destination writes.
int goc_v_add_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_mul_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_fma_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c, uint32_t *excp_flag_user);

// FP64 min/max uses two VGPRs per operand, with A/B ABS/NEG, OMOD and CLAMP.
// Number variants prefer numeric operands over NaNs; minimum/maximum propagate
// NaNs, preferring signaling NaNs and quieting them. Both order -0 below +0.
// Loose semantics and the same FP64 alias/host-FP-state contract apply.
int goc_v_min_num_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

int goc_v_max_num_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

int goc_v_minimum_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

int goc_v_maximum_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

// Unary FP64 arithmetic uses the same low/high VGPR layout. Supports A ABS/NEG,
// OMOD and CLAMP, with loose semantics. RNDNE rounds ties to even; FRACT computes
// x - floor(x), capped at 0x3fefffffffffffff before output modifiers.
int goc_v_trunc_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_ceil_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_rndne_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_floor_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_fract_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_sqrt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_rcp_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_rsq_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// Binary-significand extraction: finite nonzero inputs produce magnitude in
// [0.5,1); zeros, infinities and NaN bits pass through before output modifiers.
// Supports A ABS/NEG, OMOD and CLAMP, with loose semantics. FP32 uses one VGPR
// per operand; FP64 uses low/high VGPR pairs with the FP64 alias contract.
int goc_v_frexp_mant_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_frexp_mant_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// Binary exponent extraction into one signed 32-bit VGPR. A uses one FP32
// VGPR or an FP64 low/high pair. Returns zero for zeros, infinities and NaNs;
// finite nonzero inputs satisfy A = FREXP_MANT(A) * 2^D, including subnormals.
// A ABS/NEG, OMOD and CLAMP are accepted but do not change the integer result.
// Supports loose semantics; D may alias either whole source VGPR.
// The FP32 form supports DPP8/DPP16 on A and preserves the host FP environment.
int goc_v_frexp_exp_i32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_frexp_exp_i32_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// Scale A by 2^B, with a signed 32-bit integer exponent in one B VGPR.
// FP32 A/D each use one VGPR; FP64 A/D use low/high pairs. Supports A ABS/NEG,
// OMOD and CLAMP with loose semantics and gradual underflow. Integer B has no
// sign modifiers. D may alias any whole source VGPR; FP64 writes low then high.
// The FP32 form supports DPP8/DPP16 on A; B is not permuted.
int goc_v_ldexp_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                    uint32_t *excp_flag_user);

int goc_v_ldexp_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                    uint32_t *excp_flag_user);

// Signed/unsigned 32-bit integer selection. Each operand uses one VGPR;
// D may alias any whole source VGPR. These instructions have no arithmetic
// modifiers. Supports DPP8/DPP16 on A; low 32 instruction-flag bits must be
// zero. Supports loose semantics.
// MINMAX computes max(min(A, B), C); MAXMIN computes min(max(A, B), C).
int goc_v_min_i32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b);

int goc_v_max_i32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b);

int goc_v_min3_i32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

int goc_v_max3_i32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

int goc_v_minmax_i32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

int goc_v_maxmin_i32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

int goc_v_med3_i32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

int goc_v_min_u32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b);

int goc_v_max_u32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b);

int goc_v_min3_u32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

int goc_v_max3_u32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

int goc_v_minmax_u32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

int goc_v_maxmin_u32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

int goc_v_med3_u32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

// Integer multiplication into one 32-bit VGPR. Each source uses one VGPR;
// D may alias A or B. The 24-bit forms discard the upper byte of each input,
// then sign-extend signed inputs. High forms select bits 63:32 of the product.
// MUL_I32_I24 and MUL_U32_U24 support GOC_ALU_CLAMP to saturate to the signed
// or unsigned 32-bit range; otherwise low results wrap. All four 24-bit forms
// support DPP8/DPP16 on A; other instruction flags are invalid. The 32-bit forms
// require zero instruction_flags. Loose semantics; preserves the host FP environment.
int goc_v_mul_lo_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_mul_hi_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_mul_hi_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_mul_i32_i24(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_mul_hi_i32_i24(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_mul_u32_u24(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_mul_hi_u32_u24(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// Non-carry 32-bit integer addition/subtraction. Each operand uses one VGPR;
// D may alias any whole source VGPR. The two-input forms accept GOC_ALU_CLAMP
// to saturate in the signed/unsigned result domain; otherwise results wrap.
// ADD3 wraps modulo 2^32 and requires zero low instruction-flag bits. All forms
// support DPP8/DPP16 on A, except SUBREV_NC_U32 which permutes B. Supports loose
// semantics and preserves the host FP environment. No carry mask is produced.
int goc_v_add_nc_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_sub_nc_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_subrev_nc_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_add_nc_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_sub_nc_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_add3_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   const uint32_t *const *c);

// True16 source/destination half selectors. Zero selects the low half.
// DOT2 consumes both A/B halves and accepts only the C and D selectors.
static const uint32_t GOC_ALU_HIGH_A = 1U << 9;
static const uint32_t GOC_ALU_HIGH_B = 1U << 10;
static const uint32_t GOC_ALU_HIGH_C = 1U << 11;
static const uint32_t GOC_ALU_HIGH_D = 1U << 12;

// Mixed FMA source formats: unset means FP32; set means FP16 in the half
// selected by GOC_ALU_HIGH_A/B/C. Half selectors are ignored for FP32 sources.
static const uint32_t GOC_MIX_F16_A = 1U << 13;
static const uint32_t GOC_MIX_F16_B = 1U << 14;
static const uint32_t GOC_MIX_F16_C = 1U << 15;

// Wave32 mixed FMA: one VGPR per operand; supports source ABS/NEG, source
// format/half selectors, and CLAMP. DPP8/DPP16 permute A before format/half
// selection and modifiers; B and C retain their lanes. OMOD and HIGH_D are
// invalid. MIX_F32
// produces FP32; MIXLO/MIXHI round directly to FP16 and preserve the other
// destination half. GOC_FP16_OVFL saturates finite FP16 overflow only.
// Loose semantics require host nearest-even arithmetic with denormals enabled.
// FP16-output forms also support empirical-exact scalar semantics and preserve
// host FP state in that mode. EXEC masks and whole-register aliases are supported.
int goc_v_fma_mix_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_fma_mixlo_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_fma_mixhi_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user);

// Binary FP16 arithmetic: one VGPR per operand, with independently selected
// source and destination halves. Preserves the other destination half and all
// inactive lanes; D may alias a whole source VGPR. Supports source ABS/NEG,
// OMOD then CLAMP before nearest-even FP16 narrowing, and GOC_FP16_OVFL.
// Loose semantics require host nearest-even arithmetic with denormals enabled.
int goc_v_add_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_sub_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_subrev_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     uint32_t *excp_flag_user);

int goc_v_mul_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  uint32_t *excp_flag_user);

int goc_v_min_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

int goc_v_max_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

int goc_v_minimum_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

int goc_v_maximum_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user);

// Three-input FP16 min/max and median: independently selected halves of A/B/C/D,
// with all source ABS/NEG, OMOD, CLAMP and GOC_FP16_OVFL. Output modifiers apply
// after the final selection. Preserves the other D half and inactive lanes;
// any whole-VGPR aliases are allowed. MINMAX/MAXMIN select A/B first, then C.
// Number variants ignore lone NaNs; MINIMUM/MAXIMUM propagate NaNs. MED3 uses
// minimumNumber(A,B,C) if any input is NaN. Loose semantics require host
// nearest-even arithmetic with denormals enabled.
int goc_v_min3_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_max3_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_minmax_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_maxmin_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_minimum3_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_maximum3_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_minimummaximum_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_maximumminimum_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_med3_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

// FP16 LDEXP: scale the selected half of A by 2 raised to the signed int16_t
// exponent in the selected half of B. Supports A ABS/NEG, HIGH_A/B/D, OMOD,
// CLAMP, GOC_FP16_OVFL, and DPP8/DPP16 on A. Other source modifiers are invalid.
// Preserves the other D half and inactive lanes; D may alias a whole source VGPR.
// Loose semantics require host nearest-even arithmetic with denormals enabled.
int goc_v_ldexp_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                    uint32_t *excp_flag_user);

// FP16 FREXP exponent: read the selected A half and write a signed int16_t
// exponent into the selected D half. Zero, infinity and NaN return zero.
// Preserves the other D half, inactive lanes and the host FP environment;
// whole-register A/D aliasing is allowed. ABS_A/NEG_A, OMOD and CLAMP are
// accepted but do not change the result. Supports DPP8/DPP16 on A and loose semantics.
int goc_v_frexp_exp_i16_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// Unary FP16: one independently selected half per A/D VGPR; supports ABS_A,
// NEG_A, HIGH_A/D, OMOD, CLAMP and GOC_FP16_OVFL. Preserves the unselected D
// half and inactive lanes; whole-register A/D aliasing is allowed. EXP/LOG
// use base two, RNDNE rounds ties to even, and FRACT is capped below one.
// Supports DPP8/DPP16 on A. Loose semantics require host nearest-even arithmetic
// with denormals enabled. EXP/LOG round to FP16 before OMOD; other forms apply
// output modifiers before final FP16 narrowing.
int goc_v_trunc_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_ceil_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_rndne_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_floor_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_sqrt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_rcp_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_rsq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_exp_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_log_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_fract_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_frexp_mant_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// DPP8/DPP16 permute the complete A word before modifiers; B/C stay in their
// original lanes.
// True16 DOT2: A/B each hold two packed factors; C supplies one selected half.
// D replaces only its selected half, preserving the other half. Supports all
// six GOC_ALU ABS/NEG flags and HIGH_C/HIGH_D, without OMOD or CLAMP.
// Loose semantics use FP32 products and sums followed by nearest-even narrowing.
// BF16 flushes input/output denormals; F16 honors GOC_FP16_OVFL.
int goc_v_dot2_f16_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_dot2_bf16_bf16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c, uint32_t *excp_flag_user);

// Unary FP32: one VGPR each for A/D. Supports NEG_A, ABS_A, OMOD and CLAMP;
// modifiers for absent operands are invalid. CLAMP maps NaNs to +0 and clamps
// to [0, 1]. EXP and LOG use base 2; RSQ computes reciprocal square root.
// RNDNE rounds ties to even. Only loose semantics are implemented.
int goc_v_trunc_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_ceil_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_rndne_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_floor_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_sqrt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_rcp_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_rsq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_exp_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

int goc_v_log_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user);

// Floating DOT2 sign modifiers act after selecting each packed half.
static const uint32_t GOC_DOT_NEG_LO_A = 1U << 0;
static const uint32_t GOC_DOT_NEG_LO_B = 1U << 1;
static const uint32_t GOC_DOT_NEG_C = 1U << 2;
static const uint32_t GOC_DOT_NEG_HI_A = 1U << 3;
static const uint32_t GOC_DOT_NEG_HI_B = 1U << 4;

// Floating DOT2 half selection: defaults are low for term 0 and high for term 1.
// These flags override those defaults; unlike raw op_sel_hi, zero means default.
static const uint32_t GOC_DOT_LO_A_HIGH = 1U << 7;
static const uint32_t GOC_DOT_LO_B_HIGH = 1U << 8;
static const uint32_t GOC_DOT_HI_A_LOW = 1U << 9;
static const uint32_t GOC_DOT_HI_B_LOW = 1U << 10;

// DPP8/DPP16 permute the complete A word before sign and half selection;
// B/C remain in their original lanes. Supported in loose and exact semantics.
// RDNA4 DOT2: A/B each hold two packed 16-bit factors in one VGPR;
// C/D each hold one FP32 value per lane. Loose and exact modes are supported.
// Supports all floating DOT2 sign/half-selection flags. GOC_DOT_CLAMP is
// accepted but has no effect on these floating DOT2 forms (as in rocjitsu).
int goc_v_dot2_f32_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_dot2_f32_bf16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *excp_flag_user);

// RDNA3 DOT2 has the same register layout, sign/half selection and ignored
// CLAMP, but different empirical accumulation. Exact result bits are supported;
// exact requests with non-null excp_flag_user return UNSUPPORTED_GLOBAL_STATE.
// Loose semantics share the RDNA4 SIMD path and leave optional state unchanged.
int goc_v_dot2_f32_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user);

int goc_v_dot2_f32_bf16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user);

// RDNA3 DOT2ACC adds the packed FP16 dot product to old D (one FP32 VGPR).
// Supports DPP8/DPP16 routing of A, but no sign or half-selection modifiers.
// Exact arithmetic and reporting restrictions match RDNA3 DOT2 above. Whole
// VGPR aliasing is allowed; all results use the original accumulator/source bits.
int goc_v_dot2acc_f32_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                          uint32_t *excp_flag_user);

// FP8/BF8 DOT4 accepts NEG_C and ABS_C; ABS precedes NEG. A/B modifiers,
// half selection, output scaling and CLAMP are not supported.
static const uint32_t GOC_DOT_ABS_C = 1U << 5;

// Wave32 DOT4: one VGPR each for A/B/C/D, four packed bytes per A/B lane;
// C/D are FP32. FP8 is OCP E4M3FN, BF8 is OCP E5M2. Loose semantics only.
int goc_v_dot4_f32_fp8_fp8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int goc_v_dot4_f32_fp8_bf8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int goc_v_dot4_f32_bf8_fp8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int goc_v_dot4_f32_bf8_bf8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

// Integer DOT modifiers: SIGNED selects signed factors for I32_IU forms.
// U32_U forms accept only CLAMP. CLAMP saturates the final accumulator to its
// signed/unsigned 32-bit range; otherwise arithmetic wraps modulo 2^32.
static const uint32_t GOC_DOT_SIGNED_A = 1U << 0;
static const uint32_t GOC_DOT_SIGNED_B = 1U << 1;
static const uint32_t GOC_DOT_CLAMP = 1U << 6;

// Integer DOT wave32: one VGPR each for A/B/C/D. A/B contain four packed
// bytes or eight packed nibbles. C/D are signed for I32_IU, unsigned for U32_U.
// Loose and exact semantics return the same integer result.
int goc_v_dot4_i32_iu8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c);

int goc_v_dot4_u32_u8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c);

int goc_v_dot8_i32_iu4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c);

int goc_v_dot8_u32_u4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c);

// WMMA modifier layout follows neg_lo[0:2], then neg_hi[0:2]. For C,
// neg_hi means absolute value, applied before neg_lo negation.
static const uint32_t GOC_WMMA_NEG_LO_A = 1U << 0;
static const uint32_t GOC_WMMA_NEG_LO_B = 1U << 1;
static const uint32_t GOC_WMMA_NEG_C = 1U << 2;
static const uint32_t GOC_WMMA_NEG_HI_A = 1U << 3;
static const uint32_t GOC_WMMA_NEG_HI_B = 1U << 4;
static const uint32_t GOC_WMMA_ABS_C = 1U << 5;

// Sparse WMMA metadata selector. This bit selects the upper half of each index VGPR lane.
static const uint32_t GOC_SWMMAC_INDEX_KEY_1 = 1U << 7;

// RDNA3 packed WMMA OPSEL selects the same half of C and D. The other half
// of D is preserved. This is not a packed adjacent-row output as on RDNA4.
static const uint32_t GOC_WMMA_HIGH_C_D = 1U << 8;

// RDNA3 FP16/BF16 WMMA with FP32 accumulators. A/B each hold 8 VGPRs,
// with the full 16-element K vector in every lane. Replicate A/B across the
// two (Wave32) or four (Wave64) groups of 16 lanes. C/D hold 8 or 4 VGPRs:
// row = reg * (wave_size / 16) + lane / 16, column = lane % 16.
// EXEC is ignored and all operands must be valid. Whole-VGPR aliasing is allowed.
// Supports loose and empirical exact semantics and all six NEG/NEG_HI modifiers.
// Exact semantics use integer arithmetic and preserve the host FP environment.
// Loose FP32 arithmetic requires host nearest-even rounding and enabled denormals.
// GOC_FP16_OVFL has no effect on FP32 output.
int goc_v_wmma_f32_16x16x16_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c);

int goc_v_wmma_f32_16x16x16_f16_wave64(uint64_t flags, uint64_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_f32_16x16x16_bf16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c);

int goc_v_wmma_f32_16x16x16_bf16_wave64(uint64_t flags, uint64_t instruction_flags,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c);

// RDNA3 packed-output and integer WMMA use the same interleaved C/D layout
// above: 8 VGPRs in Wave32, 4 in Wave64. Packed A/B use 8 VGPRs each;
// IU8 uses 4 each and IU4 uses 2 each. All lanes participate and whole VGPRs may alias.
// Packed forms support all six NEG/NEG_HI bits and GOC_WMMA_HIGH_C_D, narrow
// after each DOT2 step, and honor GOC_FP16_OVFL for finite FP16 overflow.
// Integer forms support SIGNED_A/B and CLAMP (final signed saturation after
// all 16 products); without CLAMP they wrap modulo 2^32. Both semantics are
// supported. Packed and integer paths preserve the host FP environment.
int goc_v_wmma_f16_16x16x16_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c);

int goc_v_wmma_f16_16x16x16_f16_wave64(uint64_t flags, uint64_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_bf16_16x16x16_bf16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                  const uint32_t *const *a, const uint32_t *const *b,
                                  const uint32_t *const *c);

int goc_v_wmma_bf16_16x16x16_bf16_wave64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_i32_16x16x16_iu8(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c);

int goc_v_wmma_i32_16x16x16_iu8_wave64(uint64_t flags, uint64_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_i32_16x16x16_iu4(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c);

int goc_v_wmma_i32_16x16x16_iu4_wave64(uint64_t flags, uint64_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c);

// RDNA4 Wave32 16x16x16 WMMA: A/B each contain 4 VGPRs of packed 16-bit
// elements; C/D each contain 8 VGPRs of FP32 elements. WMMA and SWMMAC ignore
// architectural EXEC, including zero, and write every destination lane.
// Both loose and empirical exact semantics are supported for these WMMA forms.
int goc_v_wmma_f32_16x16x16_f16_rdna4(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_f32_16x16x16_bf16_rdna4(uint64_t flags, uint64_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c);

// Wave64 variants: 64 words per VGPR; 2 VGPRs for A/B, 4 for C/D.
// Both semantics and all WMMA modifiers are supported through scalar paths.
int goc_v_wmma_f32_16x16x16_f16_rdna4_wave64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_f32_16x16x16_bf16_rdna4_wave64(uint64_t flags, uint64_t instruction_flags,
                                              uint32_t *const *d, const uint32_t *const *a,
                                              const uint32_t *const *b, const uint32_t *const *c);

// Packed-output WMMA: A/B hold 4 VGPRs (wave32) or 2 (wave64);
// C/D hold 4 or 2 VGPRs respectively, with adjacent rows in low/high halves.
// Both semantics and all six floating WMMA modifiers are supported. Packed
// results narrow after each four-product step; GOC_FP16_OVFL controls FP16 overflow.
int goc_v_wmma_f16_16x16x16_f16_rdna4(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_bf16_16x16x16_bf16_rdna4(uint64_t flags, uint64_t instruction_flags,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_f16_16x16x16_f16_rdna4_wave64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_bf16_16x16x16_bf16_rdna4_wave64(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c);

// Wave32 FP8/BF8 WMMA: A/B each hold 2 VGPRs, C/D each hold 8.
// FP8 is OCP E4M3FN; BF8 is OCP E5M2. Only loose semantics are implemented;
// strict exact requests return GOC_ERROR_UNSUPPORTED_SEMANTICS. Supported
// modifiers are GOC_WMMA_NEG_C and GOC_WMMA_ABS_C.
int goc_v_wmma_f32_16x16x16_fp8_fp8(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                    const uint32_t *const *a, const uint32_t *const *b,
                                    const uint32_t *const *c);

int goc_v_wmma_f32_16x16x16_fp8_bf8(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                    const uint32_t *const *a, const uint32_t *const *b,
                                    const uint32_t *const *c);

int goc_v_wmma_f32_16x16x16_bf8_fp8(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                    const uint32_t *const *a, const uint32_t *const *b,
                                    const uint32_t *const *c);

int goc_v_wmma_f32_16x16x16_bf8_bf8(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                    const uint32_t *const *a, const uint32_t *const *b,
                                    const uint32_t *const *c);

// Integer WMMA modifiers: NEG[0:1] select signed interpretation of A/B;
// CLAMP saturates signed accumulation at instruction-specific stage boundaries;
// without CLAMP, results wrap modulo 2^32.
static const uint32_t GOC_WMMA_SIGNED_A = 1U << 0;
static const uint32_t GOC_WMMA_SIGNED_B = 1U << 1;
static const uint32_t GOC_WMMA_CLAMP = 1U << 6;

// Wave32 integer WMMA: C/D each hold 8 VGPRs. A/B each hold 2 VGPRs for
// IU8 and K=32 IU4, or 1 VGPR for K=16 IU4. Supports loose and exact semantics,
// signed/unsigned factors, and CLAMP. Accumulators are signed 32-bit integers.
// CLAMP applies after products with (k / 8) even, then after those with
// (k / 8) odd: K=16 uses 0..7 then 8..15; K=32 uses 0..7 plus 16..23,
// then 8..15 plus 24..31.
int goc_v_wmma_i32_16x16x16_iu8_rdna4(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_i32_16x16x16_iu4_rdna4(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c);

int goc_v_wmma_i32_16x16x32_iu4(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c);

// Bit counts: every operand holds one VGPR. CLZ/CTZ return 0xffffffff for zero;
// CLS counts leading sign bits (including the sign bit) and returns 0xffffffff
// if all bits match. BCNT returns popcount(A) + B, wrapping modulo 2^32.
// MBCNT_LO counts A bits below min(lane, 32); MBCNT_HI counts A bits below
// max(lane - 32, 0). Both add B with wrapping. The lane index is physical and
// independent of EXEC. Wave64 forms use 64 lane words per VGPR.
// Wave32 forms support DPP8/DPP16 on A; low 32 instruction-flag bits must be zero.
// Wave64 forms require zero instruction_flags. Only loose semantics are implemented;
// all host FP state is preserved.
int goc_v_clz_i32_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a);

int goc_v_ctz_i32_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a);

int goc_v_cls_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a);

int goc_v_bcnt_u32_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_mbcnt_lo_u32_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_mbcnt_hi_u32_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_mbcnt_lo_u32_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int goc_v_mbcnt_hi_u32_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// Wave32 Boolean operations: each operand holds one VGPR. All forms support
// DPP8/DPP16 on A. B32 forms have no arithmetic modifiers. B16 forms select A/B/D
// halves with HIGH_A/B/D after DPP and leave the other D half unchanged. Other
// modifiers are invalid; NOT has no B.
// Only loose semantics are implemented. All host FP state is preserved.
int goc_v_and_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_or_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b);

int goc_v_xor_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_not_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a);

int goc_v_and_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_or_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b);

int goc_v_xor_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_not_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a);

// Wave32 bit operations: every operand holds one VGPR. Supports DPP8/DPP16 on A;
// low 32 instruction-flag bits must be zero. Only loose semantics are implemented.
// BFE offsets and widths, and BFM widths and offsets, use their low five bits;
// a zero width produces zero.
// Signed BFE sign-extends A before extraction and sign-extends the extracted field.
// BFI selects B where A has a set bit, C otherwise. BFREV reverses all 32 bits.
int goc_v_bfe_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);

int goc_v_bfe_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);

int goc_v_bfi_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c);

int goc_v_bfm_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int goc_v_bfrev_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a);

#ifdef __cplusplus
} // extern "C"
#endif

#endif

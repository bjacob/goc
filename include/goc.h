// SPDX-License-Identifier: MIT

#ifndef GOC_H_
#define GOC_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU levels are an enumeration, not independently OR-able feature bits.
static const uint64_t GOC_CPU_MASK = UINT64_C(0xffff);
static const uint64_t GOC_CPU_BASELINE = UINT64_C(0);
static const uint64_t GOC_CPU_X86_64_V3 = UINT64_C(1);
static const uint64_t GOC_CPU_X86_64_V4 = UINT64_C(2);
static const uint64_t GOC_CPU_ZEN4 = UINT64_C(3);
static const uint64_t GOC_SEMANTICS_MASK = (UINT64_C(3) << 16);
static const uint64_t GOC_SEMANTICS_LOOSE = UINT64_C(0);
static const uint64_t GOC_SEMANTICS_EXACT = (UINT64_C(1) << 16);
static const uint64_t GOC_SEMANTICS_STRICT = (UINT64_C(1) << 18);

static const int GOC_SUCCESS = 0;
static const int GOC_ERROR_UNSUPPORTED_SEMANTICS = 1;
static const int GOC_ERROR_INVALID_FLAGS = 2;

// Returns runtime-usable CPU capabilities only. Callers may select a lower CPU
// level for testing, but must not claim features unavailable on the calling CPU.
uint64_t goc_init_cpu_flags(void);

// Wave32 entry points. High exec_mask bits are ignored. Each pointer names 32
// contiguous uint32_t lane words. Pointer arrays and backing storage must be valid.
// Whole VGPRs may alias; distinct VGPR addresses must not overlap. Sources are
// conceptually read before writes. Inactive destination lanes and all destinations
// on error are unchanged. Host FP mode must be nearest-even with denormals enabled.
// Initially only instruction_flags == 0 is supported. Exact semantics requests
// fall back to loose unless GOC_SEMANTICS_STRICT is set.
int goc_rdna4_v_fma_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, uint32_t *const *a, uint32_t *const *b,
                        uint32_t *const *c);
int goc_rdna4_v_log_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, uint32_t *const *a);
// RDNA4 DOT2: A/B each hold two packed 16-bit factors in one VGPR;
// C/D each hold one FP32 value per lane. Loose and exact modes are supported.
int goc_rdna4_v_dot2_f32_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                             uint32_t *const *d, uint32_t *const *a, uint32_t *const *b,
                             uint32_t *const *c);
int goc_rdna4_v_dot2_f32_bf16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                              uint32_t *const *d, uint32_t *const *a, uint32_t *const *b,
                              uint32_t *const *c);

// WMMA modifier layout follows neg_lo[0:2], then neg_hi[0:2]. For C,
// neg_hi means absolute value, applied before neg_lo negation.
static const uint32_t GOC_WMMA_NEG_LO_A = (UINT32_C(1) << 0);
static const uint32_t GOC_WMMA_NEG_LO_B = (UINT32_C(1) << 1);
static const uint32_t GOC_WMMA_NEG_C = (UINT32_C(1) << 2);
static const uint32_t GOC_WMMA_NEG_HI_A = (UINT32_C(1) << 3);
static const uint32_t GOC_WMMA_NEG_HI_B = (UINT32_C(1) << 4);
static const uint32_t GOC_WMMA_ABS_C = (UINT32_C(1) << 5);
// Wave32 16x16x16 WMMA: A/B each contain 4 VGPRs of packed 16-bit
// elements; C/D each contain 8 VGPRs of FP32 elements. GoC applies exec_mask
// to destination writes, including WMMA. All source lanes remain readable.
// Both loose and empirical exact semantics are supported for these WMMA forms.
int goc_rdna4_v_wmma_f32_16x16x16_f16(uint64_t flags, uint64_t exec_mask,
                                      uint32_t instruction_flags, uint32_t *const *d,
                                      uint32_t *const *a, uint32_t *const *b, uint32_t *const *c);
int goc_rdna4_v_wmma_f32_16x16x16_bf16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       uint32_t *const *a, uint32_t *const *b, uint32_t *const *c);
// Wave64 variants: 64 words per VGPR; 2 VGPRs for A/B, 4 for C/D.
// Both semantics and all WMMA modifiers are supported through scalar paths.
int goc_rdna4w64_v_wmma_f32_16x16x16_f16(uint64_t flags, uint64_t exec_mask,
                                         uint32_t instruction_flags, uint32_t *const *d,
                                         uint32_t *const *a, uint32_t *const *b,
                                         uint32_t *const *c);
int goc_rdna4w64_v_wmma_f32_16x16x16_bf16(uint64_t flags, uint64_t exec_mask,
                                          uint32_t instruction_flags, uint32_t *const *d,
                                          uint32_t *const *a, uint32_t *const *b,
                                          uint32_t *const *c);

#ifdef __cplusplus
}
#endif

#endif

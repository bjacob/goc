// SPDX-License-Identifier: MIT
#ifndef GOC_H_
#define GOC_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// CPU levels are an enumeration, not independently OR-able feature bits.
#define GOC_CPU_MASK UINT64_C(0xffff)
#define GOC_CPU_BASELINE UINT64_C(0)
#define GOC_CPU_X86_64_V3 UINT64_C(1)
#define GOC_CPU_X86_64_V4 UINT64_C(2)
#define GOC_CPU_ZEN4 UINT64_C(3)
#define GOC_SEMANTICS_MASK (UINT64_C(3) << 16)
#define GOC_SEMANTICS_LOOSE UINT64_C(0)
#define GOC_SEMANTICS_EXACT (UINT64_C(1) << 16)
#define GOC_SEMANTICS_STRICT (UINT64_C(1) << 18)

enum { GOC_SUCCESS = 0, GOC_ERROR_UNSUPPORTED_SEMANTICS = 1, GOC_ERROR_INVALID_FLAGS = 2 };

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
#ifdef __cplusplus
}
#endif
#endif

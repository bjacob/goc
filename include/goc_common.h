// SPDX-License-Identifier: MIT

#ifndef GOC_COMMON_H_
#define GOC_COMMON_H_

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

#ifdef __cplusplus
}
#endif

#endif

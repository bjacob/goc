// SPDX-License-Identifier: MIT

#ifndef GOC_COMMON_H_
#define GOC_COMMON_H_

#include "goc_export.h"

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

// Semantics are an enumeration, not independently OR-able feature bits.
// Values 2 and 3 are currently reserved for future semantics.
static const uint64_t GOC_SEMANTICS_MASK = (UINT64_C(3) << 16);
static const uint64_t GOC_SEMANTICS_LOOSE = UINT64_C(0);
static const uint64_t GOC_SEMANTICS_EXACT_EMPIRICAL = (UINT64_C(1) << 16);

// Require support for the selected semantics instead of falling back to loose.
static const uint64_t GOC_SEMANTICS_STRICT = (UINT64_C(1) << 18);

// Status codes returned by instruction emulation functions.
static const int GOC_SUCCESS = 0;
static const int GOC_ERROR_UNSUPPORTED_SEMANTICS = 1;
static const int GOC_ERROR_INVALID_FLAGS = 2;

// Returns CPU capability flags for features usable on this machine, accounting
// for CPU and operating-system support. All non-CPU flag bits are zero.
GOC_API uint64_t goc_init_cpu_flags(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif

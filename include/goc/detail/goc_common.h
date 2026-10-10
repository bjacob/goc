// SPDX-License-Identifier: MIT

#ifndef GOC_COMMON_H_
#define GOC_COMMON_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU levels are an enumeration, not independently OR-able feature bits.
static const uint64_t GOC_CPU_MASK = 0xffffULL;
static const uint64_t GOC_CPU_BASELINE = 0ULL;
static const uint64_t GOC_CPU_X86_64_V3 = 1ULL;
static const uint64_t GOC_CPU_X86_64_V4 = 2ULL;
static const uint64_t GOC_CPU_ZEN4 = 3ULL;

// Semantics are an enumeration, not independently OR-able feature bits.
// Values 2 and 3 are currently reserved for future semantics.
static const uint64_t GOC_SEMANTICS_MASK = 3ULL << 16;
static const uint64_t GOC_SEMANTICS_LOOSE = 0ULL;
static const uint64_t GOC_SEMANTICS_EXACT_EMPIRICAL = 1ULL << 16;

// Require support for the selected semantics instead of falling back to loose.
static const uint64_t GOC_SEMANTICS_STRICT = 1ULL << 18;

// GPU MODE.FP16_OVFL: saturate finite overflow to the largest finite value for
// supporting FP16/FP8/BF8 instructions. Otherwise use the format's overflow encoding.
// Does not change input-infinity handling or BF16/FP32 results, and is independent
// of the host floating-point environment.
static const uint64_t GOC_FP16_OVFL = 1ULL << 19;

// Flush guest input subnormals to signed zero. Supported by floating comparisons
// and pseudo-scalar FP16 math, plus FP32 binary arithmetic/min/max and ordinary FP16
// ADD/SUB/SUBREV/MUL and FP64 ADD/MUL/FMA/min/max. CLASS accepts it without changing
// classification; pseudo-scalar FP32 math, RCP_IFLAG, and DX9 FMA always flush. Other instructions
// reject it. Independent of host FP settings; zero preserves inputs where configurable.
static const uint64_t GOC_FP_FLUSH_INPUT_DENORMALS = 1ULL << 20;

// Flush guest output subnormals to signed zero. Supported by pseudo-scalar FP16
// math and FP32 binary arithmetic/min/max and ordinary FP16 ADD/SUB/SUBREV/MUL and FP64
// ADD/MUL/FMA. Pseudo-scalar FP32 math, RCP_IFLAG, and DX9 FMA always flush; floating and CLASS
// comparisons accept this bit without effect because they produce integer masks. FP32/FP64 min/max
// ignore output flushing except through OMOD. Other instructions currently reject it. Independent
// of host FTZ; zero preserves outputs where configurable. Nonzero OMOD may require flushing
// independently.
static const uint64_t GOC_FP_FLUSH_OUTPUT_DENORMALS = 1ULL << 21;

// Guest MODE.IEEE for the FP32 binary arithmetic/min/max and ordinary FP16 ADD/SUB/SUBREV/MUL and
// FP64 ADD/MUL/FMA/min/max APIs. On RDNA3 this disables OMOD and changes MIN/MAX NaN selection;
// RDNA4 accepts it without changing OMOD. These APIs support only loose arithmetic; exact
// arithmetic and exception reporting remain unsupported. Other APIs currently reject this flag. It
// never changes the host FP environment.
static const uint64_t GOC_FP_IEEE_MODE = 1ULL << 22;

// Status codes returned by instruction emulation functions.
static const int GOC_SUCCESS = 0;
static const int GOC_ERROR_UNSUPPORTED_SEMANTICS = 1;
static const int GOC_ERROR_INVALID_FLAGS = 2;

// Faithful updates to a requested implicit architectural register are unavailable.
// Applies only to non-loose semantics with a non-NULL global-state output pointer.
static const int GOC_ERROR_UNSUPPORTED_GLOBAL_STATE = 3;

// Returns CPU capability flags for features usable on this machine, accounting
// for CPU and operating-system support. All non-CPU flag bits are zero.
uint64_t goc_init_cpu_flags(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif

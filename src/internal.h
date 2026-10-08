// SPDX-License-Identifier: MIT

#ifndef GOC_INTERNAL_H_
#define GOC_INTERNAL_H_

#include "goc/goc.h"

#include <cstring>
#include <stdint.h>

namespace goc {

inline float as_float(uint32_t v) {
  float f;
  std::memcpy(&f, &v, sizeof(f));
  return f;
}

inline uint32_t as_bits(float f) {
  uint32_t v;
  std::memcpy(&v, &f, sizeof(v));
  return v;
}

inline int validate(uint64_t flags, uint32_t instruction_flags, bool supports_exact = false) {
  constexpr uint64_t known =
      GOC_CPU_MASK | GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT | GOC_FP16_OVFL;
  if ((flags & ~known) || (flags & GOC_CPU_MASK) > GOC_CPU_ZEN4 || instruction_flags)
    return GOC_ERROR_INVALID_FLAGS;
  if ((flags & GOC_SEMANTICS_MASK) &&
      !(supports_exact && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL) &&
      (flags & GOC_SEMANTICS_STRICT))
    return GOC_ERROR_UNSUPPORTED_SEMANTICS;
  return GOC_SUCCESS;
}

// Snapshot of CPUID and XCR0 for deterministic OS-state/feature decoder tests.
struct CpuState {
  uint32_t leaf1_ecx = 0, leaf7_ebx = 0, leaf7_ecx = 0, leaf71_eax = 0, ext1_ecx = 0;
  uint64_t xcr0 = 0;
};

uint64_t decode_cpu(const CpuState &s);

} // namespace goc

#endif

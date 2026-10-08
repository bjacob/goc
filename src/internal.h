// SPDX-License-Identifier: MIT

#ifndef GOC_INTERNAL_H_
#define GOC_INTERNAL_H_

#include "goc_common.h"

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
  constexpr uint64_t known = GOC_CPU_MASK | GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT;
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

#if defined(GOC_HAVE_AVX2)
void fma_avx2(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);
#endif

#if defined(GOC_HAVE_AVX512F)
void fma_avx512f(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                 const uint32_t *c);
#endif

#if defined(GOC_HAVE_AVX512BF16)
void wmma_avx512bf16(uint32_t mask, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c);
#endif

} // namespace goc

#endif

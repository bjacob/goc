// SPDX-License-Identifier: MIT

#include "goc_common.h"
#include "internal.h"

#include <stdint.h>

#if defined(GOC_X86_64_QUERY)
#include <cpuid.h>
#endif

namespace goc {

// CPUID/XCR0 gating follows hrx-system/runtime/src/iree/base/internal/cpu_x86_64.c.
// Masks below coalesce the relevant architectural features into GoC's CPU levels.
uint64_t decode_cpu(const CpuState &s) {
  constexpr uint32_t v3_1 = (1u << 0) | (1u << 9) | (1u << 12) | (1u << 13) | (1u << 19) |
                            (1u << 20) | (1u << 22) | (1u << 23) | (1u << 26) | (1u << 27) |
                            (1u << 28) | (1u << 29);
  constexpr uint32_t v3_7 = (1u << 3) | (1u << 5) | (1u << 8);
  if ((s.leaf1_ecx & v3_1) != v3_1 || (s.leaf7_ebx & v3_7) != v3_7 || (s.ext1_ecx & 0x21) != 0x21 ||
      (s.xcr0 & 0x6) != 0x6)
    return GOC_CPU_BASELINE;

  constexpr uint32_t v4_7 = (1u << 16) | (1u << 17) | (1u << 28) | (1u << 30) | (1u << 31);
  if ((s.leaf7_ebx & v4_7) != v4_7 || (s.xcr0 & 0xe6) != 0xe6)
    return GOC_CPU_X86_64_V3;

  constexpr uint32_t zen4_7c = (1u << 1) | (1u << 6) | (1u << 8) | (1u << 9) | (1u << 10) |
                               (1u << 11) | (1u << 12) | (1u << 14);
  if (!(s.leaf7_ebx & (1u << 21)) || (s.leaf7_ecx & zen4_7c) != zen4_7c ||
      !(s.leaf71_eax & (1u << 5)))
    return GOC_CPU_X86_64_V4;
  return GOC_CPU_ZEN4;
}

} // namespace goc

uint64_t goc_init_cpu_flags(void) {
#if defined(GOC_X86_64_QUERY)
  goc::CpuState s;
  uint32_t a, b, c, d;
  const uint32_t max_leaf = __get_cpuid_max(0, nullptr);
  if (max_leaf >= 1) {
    __cpuid_count(1, 0, a, b, c, d);
    s.leaf1_ecx = c;
  }

  if (max_leaf >= 7) {
    __cpuid_count(7, 0, a, b, c, d);
    s.leaf7_ebx = b;
    s.leaf7_ecx = c;
    if (a >= 1) {
      __cpuid_count(7, 1, a, b, c, d);
      s.leaf71_eax = a;
    }
  }

  if (__get_cpuid_max(0x80000000u, nullptr) >= 0x80000001u) {
    __cpuid_count(0x80000001u, 0, a, b, c, d);
    s.ext1_ecx = c;
  }

  if ((s.leaf1_ecx & ((1u << 26) | (1u << 27))) == ((1u << 26) | (1u << 27))) {
    __asm__ volatile("xgetbv" : "=a"(a), "=d"(d) : "c"(0));
    s.xcr0 = (uint64_t(d) << 32) | a;
  }

  return goc::decode_cpu(s);
#else
  return GOC_CPU_BASELINE;
#endif
}

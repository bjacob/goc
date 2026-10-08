// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

#if defined(GOC_X86_64_QUERY)
#include <cpuid.h>
#include <immintrin.h>
#endif

#if defined(GOC_X86_64_QUERY)
namespace {

// Returns XCR0. Requires CPUID to report both XSAVE and OSXSAVE support.
[[gnu::target("xsave")]] uint64_t read_xcr0(void) { return _xgetbv(0); }

} // namespace
#endif

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

  // CPUID(1, 0).ECX: XSAVE (26) and OSXSAVE (27) make XGETBV available.
  constexpr uint32_t xgetbv_leaf1_ecx = 0x0c000000;
  if ((s.leaf1_ecx & xgetbv_leaf1_ecx) == xgetbv_leaf1_ecx)
    s.xcr0 = read_xcr0();

  return goc::decode_cpu(s);
#else
  return GOC_CPU_BASELINE;
#endif
}

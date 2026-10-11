// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <cfenv>
#include <cstdint>
#include <gtest/gtest.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace goc_test {

// The saved MODE values use identical F32 and F16/F64 denormal policies.
// MODE rounding selectors have no GoC API counterpart; borrowed matrix models
// ignore them. IEEE-on is an additional model-invariance check, not a capture.
inline uint64_t matrix_capture_flags(uint32_t mode, unsigned host_case) {
  EXPECT_EQ((mode >> 4) & 3, (mode >> 6) & 3);
  return ((mode & (1U << 4)) ? 0 : GOC_FP_FLUSH_INPUT_DENORMALS) |
         ((mode & (1U << 5)) ? 0 : GOC_FP_FLUSH_OUTPUT_DENORMALS) |
         ((mode & (1U << 23)) ? GOC_FP16_OVFL : 0) |
         ((mode & (1U << 9)) || (host_case & 1) ? GOC_FP_IEEE_MODE : 0);
}

// Test-only host setup: four rounding directions with FTZ/DAZ off and on.
// The caller owns a ScopedFpEnvironment so assertion failures restore state.
inline void matrix_test_host(unsigned state) {
  const int modes[] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
  ASSERT_EQ(std::fesetround(modes[state % 4]), 0);
#if defined(__x86_64__) || defined(_M_X64)
  unsigned csr = _mm_getcsr() & ~0x8040U;
  _mm_setcsr(csr | (state & 4 ? 0x8040U : 0));
#endif
}

inline unsigned matrix_host_controls() {
#if defined(__x86_64__) || defined(_M_X64)
  return _mm_getcsr() & ~63U; // Ordinary arithmetic may raise sticky flags.
#else
  return unsigned(std::fegetround());
#endif
}

} // namespace goc_test

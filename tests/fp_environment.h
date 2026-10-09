// SPDX-License-Identifier: MIT

#pragma once

#include <cfenv>
#include <gtest/gtest.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace goc_test {

// Restore the saved host FP environment on scope exit, including early returns
// and the full MXCSR register on x86-64.
// Callers must check saved() before changing the environment. A failed save is
// never restored; a failed restoration is reported as a test failure.
class ScopedFpEnvironment {
public:
  ScopedFpEnvironment() : saved_(std::fegetenv(&environment_) == 0) {
#if defined(__x86_64__) || defined(_M_X64)
    mxcsr_ = _mm_getcsr();
#endif
  }

  ~ScopedFpEnvironment() {
    if (saved_) {
      EXPECT_EQ(std::fesetenv(&environment_), 0);
#if defined(__x86_64__) || defined(_M_X64)
      _mm_setcsr(mxcsr_);
#endif
    }
  }

  ScopedFpEnvironment(const ScopedFpEnvironment &) = delete;
  ScopedFpEnvironment &operator=(const ScopedFpEnvironment &) = delete;

  bool saved() const { return saved_; }

private:
  std::fenv_t environment_;
  bool saved_;
#if defined(__x86_64__) || defined(_M_X64)
  unsigned mxcsr_;
#endif
};

} // namespace goc_test

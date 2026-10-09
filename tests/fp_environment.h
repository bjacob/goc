// SPDX-License-Identifier: MIT

#pragma once

#include <cfenv>
#include <gtest/gtest.h>

namespace goc_test {

// Restore the saved host FP environment on scope exit, including early returns.
// Callers must check saved() before changing the environment. A failed save is
// never restored; a failed restoration is reported as a test failure.
class ScopedFpEnvironment {
public:
  ScopedFpEnvironment() : saved_(std::fegetenv(&environment_) == 0) {}

  ~ScopedFpEnvironment() {
    if (saved_) {
      EXPECT_EQ(std::fesetenv(&environment_), 0);
    }
  }

  ScopedFpEnvironment(const ScopedFpEnvironment &) = delete;
  ScopedFpEnvironment &operator=(const ScopedFpEnvironment &) = delete;

  bool saved() const { return saved_; }

private:
  std::fenv_t environment_;
  bool saved_;
};

} // namespace goc_test

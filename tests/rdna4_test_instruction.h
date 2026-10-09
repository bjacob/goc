// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// Holds either wave width for test tables that exercise both API signatures.
template <typename Function> struct WaveInstruction;

template <typename... Operands>
struct WaveInstruction<int (*)(uint64_t, uint32_t, uint64_t, Operands...)> {
  using Wave32 = int (*)(uint64_t, uint32_t, uint64_t, Operands...);
  using Wave64 = int (*)(uint64_t, uint64_t, uint64_t, Operands...);
  Wave32 wave32 = nullptr;
  Wave64 wave64 = nullptr;

  constexpr WaveInstruction(Wave32 fn) : wave32(fn) {}

  constexpr WaveInstruction(Wave64 fn) : wave64(fn) {}

  int operator()(uint64_t flags, uint64_t mask, uint64_t mode, Operands... operands) const {
    return wave32 ? wave32(flags, uint32_t(mask), mode, operands...)
                  : wave64(flags, mask, mode, operands...);
  }
};

} // namespace goc_test

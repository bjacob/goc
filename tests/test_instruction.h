// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>
#include <type_traits>

namespace goc_test {

// Numerical tests and benchmarks opt out of exception recording explicitly.
template <typename Function, typename... Args>
int without_exceptions(Function function, Args... args) {
  if constexpr (std::is_invocable_v<Function, Args..., uint32_t *>)
    return function(args..., nullptr);
  else
    return function(args...);
}

// Test tables may mix wave widths and instructions with/without exception output.
// Function names supplied as the template argument must have no exception output.
template <typename Function> struct WaveInstruction;

template <typename... Operands>
struct WaveInstruction<int (*)(uint64_t, uint32_t, uint64_t, Operands...)> {
  using Wave32 = int (*)(uint64_t, uint32_t, uint64_t, Operands...);
  using Wave64 = int (*)(uint64_t, uint64_t, uint64_t, Operands...);
  using ReportingWave32 = int (*)(uint64_t, uint32_t, uint64_t, Operands..., uint32_t *);
  using ReportingWave64 = int (*)(uint64_t, uint64_t, uint64_t, Operands..., uint32_t *);
  ReportingWave32 reporting_wave32 = nullptr;
  ReportingWave64 reporting_wave64 = nullptr;
  Wave32 wave32 = nullptr;
  Wave64 wave64 = nullptr;

  constexpr WaveInstruction(Wave32 fn) : wave32(fn) {}

  constexpr WaveInstruction(Wave64 fn) : wave64(fn) {}

  constexpr WaveInstruction(ReportingWave32 fn) : reporting_wave32(fn) {}

  constexpr WaveInstruction(ReportingWave64 fn) : reporting_wave64(fn) {}

  int operator()(uint64_t flags, uint64_t exec_mask, uint64_t mode, Operands... operands,
                 uint32_t *excp_flag_user = nullptr) const {
    if (reporting_wave32)
      return reporting_wave32(flags, uint32_t(exec_mask), mode, operands..., excp_flag_user);
    if (reporting_wave64)
      return reporting_wave64(flags, exec_mask, mode, operands..., excp_flag_user);
    if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
      return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
    return wave32 ? wave32(flags, uint32_t(exec_mask), mode, operands...)
                  : wave64(flags, exec_mask, mode, operands...);
  }
};

} // namespace goc_test

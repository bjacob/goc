// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

template <typename... Operands>
void check_high_flags(const char *name,
                      int (*instruction)(uint64_t, uint64_t, uint64_t, Operands...)) {
  SCOPED_TRACE(name);
  for (unsigned bit = 32; bit < 64; ++bit)
    for (uint64_t exec : {UINT64_C(0), UINT64_MAX}) {
      // Invalid flags must be rejected before reading even required scalar outputs.
      EXPECT_EQ(instruction(0, exec, UINT64_C(1) << bit, Operands{}...), GOC_ERROR_INVALID_FLAGS)
          << bit;
    }
}

} // namespace

TEST(InstructionFlags, EveryEntryPointRejectsUnsupportedHighBits) {
#define GOC_RDNA4_API(name) check_high_flags(#name, name);
#include "rdna4_api_symbols.inc"
#undef GOC_RDNA4_API
}

// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

template <typename T> using Fn = int (*)(uint64_t, uint64_t, T *, T, T, T *, uint32_t *);

template <typename T> void check(const Fn<T> *functions) {
  const T values[] = {0,
                      T(~T(0)),
                      1,
                      T(1) << (sizeof(T) * 8 - 1),
                      T(0xaaaaaaaaaaaaaaaaULL),
                      T(0x5555555555555555ULL)};
  for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
    for (T a : values)
      for (T old : values)
        for (unsigned op = 0; op < 12; ++op) {
          // Independent bit-by-bit truth tables, in wrapper order.
          T wanted = 0;
          const unsigned truth[] = {8, 14, 6, 7, 1, 9, 4, 13, 2, 11, 4, 2};
          for (unsigned bit = 0; bit < sizeof(T) * 8; ++bit) {
            unsigned index = unsigned((a >> bit) & 1) | (unsigned((old >> bit) & 1) << 1);
            wanted |= T((truth[op] >> index) & 1) << bit;
          }
          T d = 0, exec = old;
          uint32_t scc = 7;
          ASSERT_EQ(functions[op](semantics, 0, &d, a, old, &exec, &scc), GOC_SUCCESS);
          ASSERT_EQ(d, op < 10 ? old : wanted);
          ASSERT_EQ(exec, wanted);
          ASSERT_EQ(scc, unsigned(wanted != 0));
          ASSERT_EQ(functions[op](semantics, 0, &d, a, old, nullptr, &scc), GOC_SUCCESS);
          ASSERT_EQ(d, op < 10 ? old : wanted);
          ASSERT_EQ(scc, unsigned(wanted != 0));
          ASSERT_EQ(functions[op](semantics, 0, &d, a, old, &d, &scc), GOC_SUCCESS);
          ASSERT_EQ(d, wanted);
          T alias = 0;
          ASSERT_EQ(functions[op](semantics, 0, &alias, a, old, &alias,
                                  reinterpret_cast<uint32_t *>(&alias)),
                    GOC_SUCCESS);
          uint32_t low;
          std::memcpy(&low, &alias, sizeof low);
          ASSERT_EQ(low, unsigned(wanted != 0));
          for (unsigned bit = 0; bit < 64; ++bit)
            EXPECT_EQ(functions[op](semantics, 1ULL << bit, nullptr, a, old, nullptr, nullptr),
                      GOC_ERROR_INVALID_FLAGS);
        }
}

} // namespace

TEST(ScalarExec, B32TruthTablesAliasesAndValidation) {
  const Fn<uint32_t> functions[] = {
      goc_s_and_saveexec_b32,      goc_s_or_saveexec_b32,      goc_s_xor_saveexec_b32,
      goc_s_nand_saveexec_b32,     goc_s_nor_saveexec_b32,     goc_s_xnor_saveexec_b32,
      goc_s_and_not0_saveexec_b32, goc_s_or_not0_saveexec_b32, goc_s_and_not1_saveexec_b32,
      goc_s_or_not1_saveexec_b32,  goc_s_and_not0_wrexec_b32,  goc_s_and_not1_wrexec_b32};
  check<uint32_t>(functions);
}

TEST(ScalarExec, B64TruthTablesAliasesAndValidation) {
  const Fn<uint64_t> functions[] = {
      goc_s_and_saveexec_b64,      goc_s_or_saveexec_b64,      goc_s_xor_saveexec_b64,
      goc_s_nand_saveexec_b64,     goc_s_nor_saveexec_b64,     goc_s_xnor_saveexec_b64,
      goc_s_and_not0_saveexec_b64, goc_s_or_not0_saveexec_b64, goc_s_and_not1_saveexec_b64,
      goc_s_or_not1_saveexec_b64,  goc_s_and_not0_wrexec_b64,  goc_s_and_not1_wrexec_b64};
  check<uint64_t>(functions);
}

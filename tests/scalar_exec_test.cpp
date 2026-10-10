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

template <typename T>
void nested_branches(Fn<T> and_save, Fn<T> xor_save, Fn<T> or_save, Fn<T> and_not1_write) {
  uint64_t seed = 0x923456789abcdef1ULL;
  auto random = [&]() {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return T(seed);
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned sample = 0; sample < 512; ++sample) {
        const T original = sample == 0 ? T(0) : sample == 1 ? T(~T(0)) : random();
        const T outer_condition = random(), inner_condition = random();
        const uint64_t flags = cpu | semantics | GOC_SEMANTICS_STRICT;
        T exec_mask = original, outer_saved = 0, inner_saved = 0, scratch = 0;
        uint32_t scc = 99;
        ASSERT_EQ(and_save(flags, 0, &outer_saved, outer_condition, exec_mask, &exec_mask, &scc),
                  GOC_SUCCESS);
        const T outer_then = exec_mask;
        EXPECT_EQ(outer_saved, original);
        EXPECT_EQ(scc, unsigned(exec_mask != 0));
        ASSERT_EQ(and_save(flags, 0, &inner_saved, inner_condition, exec_mask, &exec_mask, &scc),
                  GOC_SUCCESS);
        const T inner_then = exec_mask;
        EXPECT_EQ(inner_saved, outer_then);
        EXPECT_EQ(scc, unsigned(exec_mask != 0));
        ASSERT_EQ(and_not1_write(flags, 0, &scratch, inner_saved, exec_mask, &exec_mask, &scc),
                  GOC_SUCCESS);
        const T inner_else = exec_mask;
        EXPECT_EQ(scratch, inner_else);
        EXPECT_EQ(scc, unsigned(exec_mask != 0));
        ASSERT_EQ(or_save(flags, 0, &scratch, inner_saved, exec_mask, &exec_mask, &scc),
                  GOC_SUCCESS);
        EXPECT_EQ(scratch, inner_else);
        EXPECT_EQ(exec_mask, outer_then);
        ASSERT_EQ(xor_save(flags, 0, &scratch, outer_saved, exec_mask, &exec_mask, &scc),
                  GOC_SUCCESS);
        const T outer_else = exec_mask;
        EXPECT_EQ(scratch, outer_then);
        EXPECT_EQ(scc, unsigned(exec_mask != 0));
        // Check lane membership independently of the word-wide Boolean operations.
        for (unsigned lane = 0; lane < sizeof(T) * 8; ++lane) {
          bool entered = (original >> lane) & 1;
          bool outer = (outer_condition >> lane) & 1;
          bool inner = (inner_condition >> lane) & 1;
          EXPECT_EQ(bool((inner_then >> lane) & 1), entered && outer && inner);
          EXPECT_EQ(bool((inner_else >> lane) & 1), entered && outer && !inner);
          EXPECT_EQ(bool((outer_else >> lane) & 1), entered && !outer);
        }
        ASSERT_EQ(or_save(flags, 0, &scratch, outer_saved, exec_mask, &exec_mask, &scc),
                  GOC_SUCCESS);
        EXPECT_EQ(scratch, outer_else);
        EXPECT_EQ(exec_mask, original);
        EXPECT_EQ(scc, unsigned(original != 0));
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

TEST(ScalarExec, B32NestedBranchesRestoreExecAndScc) {
  nested_branches<uint32_t>(goc_s_and_saveexec_b32, goc_s_xor_saveexec_b32, goc_s_or_saveexec_b32,
                            goc_s_and_not1_wrexec_b32);
}

TEST(ScalarExec, B64NestedBranchesRestoreExecAndScc) {
  nested_branches<uint64_t>(goc_s_and_saveexec_b64, goc_s_xor_saveexec_b64, goc_s_or_saveexec_b64,
                            goc_s_and_not1_wrexec_b64);
}

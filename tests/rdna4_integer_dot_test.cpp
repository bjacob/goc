// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_fma_f32);
const Fn functions[] = {goc_rdna4_v_dot4_i32_iu8, goc_rdna4_v_dot4_u32_u8, goc_rdna4_v_dot8_i32_iu4,
                        goc_rdna4_v_dot8_u32_u4};

uint32_t reference(int op, uint32_t flags, uint32_t a, uint32_t b, uint32_t c) {
  const int radix = op < 2 ? 256 : 16, count = op < 2 ? 4 : 8;
  int64_t sum = c;
  if (!(op & 1) && sum > INT32_MAX)
    sum -= INT64_C(4294967296);
  for (int i = 0; i < count; ++i) {
    int x = a % radix, y = b % radix;
    a /= radix;
    b /= radix;
    if ((flags & GOC_DOT_SIGNED_A) && x >= radix / 2)
      x -= radix;
    if ((flags & GOC_DOT_SIGNED_B) && y >= radix / 2)
      y -= radix;
    sum += int64_t(x) * y;
  }
  if (flags & GOC_DOT_CLAMP) {
    if (op & 1)
      sum = std::min(sum, int64_t(UINT32_MAX));
    else
      sum = std::clamp(sum, int64_t(INT32_MIN), int64_t(INT32_MAX));
  }
  return uint32_t(sum);
}

} // namespace

TEST(IntegerDot, AllModifiersCpuLevelsMasksAliasesAndBoundaryInputs) {
  std::mt19937 rng(918273);
  uint32_t inputs[3][32];
  for (auto &reg : inputs)
    for (auto &word : reg)
      word = rng();
  const uint32_t patterns[] = {0,          0xffffffff, 0x80808080, 0x7f7f7f7f,
                               0x88888888, 0x77777777, 0x80ff7f01, 0x0180017f};
  for (int i = 0; i < 16; ++i) {
    inputs[0][i] = patterns[i % 8];
    inputs[1][i] = patterns[(i / 8 ? 7 - i % 8 : i % 8)];
  }
  for (int i = 0; i < 32; ++i)
    inputs[2][i] = i % 4 == 0   ? 0x7ffffff0
                   : i % 4 == 1 ? 0x80000010
                   : i % 4 == 2 ? 0xfffffff0
                                : 0x00000010;
  for (int op = 0; op < 4; ++op)
    for (unsigned mode = 0; mode < 8; ++mode) {
      uint32_t flags = (mode & 3) | (mode & 4 ? GOC_DOT_CLAMP : 0);
      if ((op & 1) && (mode & 3))
        continue;
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t semantics :
             {uint64_t{0}, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
          for (uint32_t mask : rdna4_exec_masks())
            for (int alias = 0; alias < 4; ++alias) {
              SCOPED_TRACE(::testing::Message()
                           << op << "/" << mode << "/" << cpu << "/" << mask << "/" << alias);
              uint32_t storage[4][34], before[32];
              uint32_t *v[4];
              for (int r = 0; r < 4; ++r) {
                std::fill(storage[r], storage[r] + 34, 0xdeadbeef);
                v[r] = storage[r] + 1;
                if (r < 3)
                  std::copy(inputs[r], inputs[r] + 32, v[r]);
              }
              std::copy(v[alias], v[alias] + 32, before);
              ASSERT_EQ(functions[op](cpu | semantics, mask, flags, &v[alias], &v[0], &v[1], &v[2]),
                        GOC_SUCCESS);
              for (int i = 0; i < 32; ++i) {
                uint32_t want = ((mask >> i) & 1)
                                    ? reference(op, flags, inputs[0][i], inputs[1][i], inputs[2][i])
                                    : before[i];
                EXPECT_EQ(v[alias][i], want);
              }
              for (const auto &reg : storage) {
                EXPECT_EQ(reg[0], 0xdeadbeef);
                EXPECT_EQ(reg[33], 0xdeadbeef);
              }
            }
    }
}

TEST(IntegerDot, ClampAfterWholeDotAndNoSaturatingBytePairs) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t a[32] = {}, b[32] = {}, c[32] = {}, d[32] = {};
    auto pa = a, pb = b, pc = c, pd = d;
    // The first product overflows C, but subsequent negative products cancel it.
    a[0] = 0x01010101;
    b[0] = 0xffff0101;
    c[0] = 0x7fffffff;
    // Unsigned byte pairs exceed int16 and must not saturate before summation.
    a[1] = b[1] = 0xffffffff;
    ASSERT_EQ(
        goc_rdna4_v_dot4_i32_iu8(cpu, 3, GOC_DOT_SIGNED_B | GOC_DOT_CLAMP, &pd, &pa, &pb, &pc),
        GOC_SUCCESS);
    EXPECT_EQ(d[0], 0x7fffffffu);
    EXPECT_EQ(d[1], uint32_t(-1020));
    ASSERT_EQ(goc_rdna4_v_dot4_u32_u8(cpu, 2, 0, &pd, &pa, &pb, &pc), GOC_SUCCESS);
    EXPECT_EQ(d[1], 260100u);
    a[0] = 0x11111111;
    b[0] = 0xffff1111;
    ASSERT_EQ(
        goc_rdna4_v_dot8_i32_iu4(cpu, 1, GOC_DOT_SIGNED_B | GOC_DOT_CLAMP, &pd, &pa, &pb, &pc),
        GOC_SUCCESS);
    EXPECT_EQ(d[0], 0x7fffffffu);
  }
}

TEST(IntegerDot, ValidationAndFpEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[32], dest[32];
      std::fill(words, words + 32, 0x7f800001);
      std::fill(dest, dest + 32, 0xdeadbeef);
      auto p = words, d = dest;
      for (uint32_t mask : {0U, UINT32_MAX}) {
        EXPECT_EQ(functions[op](cpu, mask, 1u << 31, &d, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[op](cpu | (2ULL << 16) | GOC_SEMANTICS_STRICT, mask, 0, &d, &p, &p, &p),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
        if (op & 1) {
          EXPECT_EQ(functions[op](cpu, mask, GOC_DOT_SIGNED_A, &d, &p, &p, &p),
                    GOC_ERROR_INVALID_FLAGS);
        }
      }
      for (auto word : dest)
        EXPECT_EQ(word, 0xdeadbeef);
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        std::fesetround(rounding);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
        EXPECT_EQ(functions[op](cpu, UINT32_MAX, GOC_DOT_CLAMP, &d, &p, &p, &p), GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
      }
    }
}

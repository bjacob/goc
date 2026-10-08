// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <vector>

namespace {

using Fn = decltype(&goc_rdna4_v_frexp_exp_i32_f32);
const Fn functions[] = {goc_rdna4_v_frexp_exp_i32_f32, goc_rdna4_v_frexp_exp_i32_f64};

uint32_t mode(int variant) {
  return (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
         ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
}

// Independent integer reference: find the leading significand bit and combine
// its position with the exponent field. Never interpret a NaN as a host float.
uint32_t reference(uint64_t bits, bool fp64) {
  int width = fp64 ? 52 : 23, bias = fp64 ? 1023 : 127;
  uint64_t fraction = bits & ((UINT64_C(1) << width) - 1);
  int field = int(bits >> width) & (2 * bias + 1);
  if (field == 2 * bias + 1 || (field == 0 && fraction == 0))
    return 0;
  if (field)
    return uint32_t(field - bias + 1);
  int leading = 0;
  while (fraction >>= 1)
    ++leading;
  return uint32_t(leading + 2 - bias - width);
}

} // namespace

TEST(FrexpExp, AllModifiersMasksAndAliases) {
  for (int fp64 = 0; fp64 < 2; ++fp64) {
    const int width = fp64 ? 52 : 23, bias = fp64 ? 1023 : 127;
    const uint64_t sign = UINT64_C(1) << (fp64 ? 63 : 31);
    const uint64_t min_normal = UINT64_C(1) << width;
    const uint64_t inf = uint64_t(2 * bias + 1) << width;
    const uint64_t inputs[] = {0,
                               sign,
                               1,
                               sign | 1,
                               min_normal - 1,
                               sign | (min_normal - 1),
                               min_normal,
                               sign | min_normal,
                               min_normal + 1,
                               sign | (min_normal + 1),
                               inf - 1,
                               sign | (inf - 1),
                               inf,
                               sign | inf,
                               inf | 1,
                               sign | inf | 1,
                               inf | (min_normal >> 1),
                               sign | inf | (min_normal >> 1),
                               uint64_t(bias) << width,
                               sign | (uint64_t(bias) << width),
                               uint64_t(bias - 1) << width,
                               sign | (uint64_t(bias - 1) << width),
                               (uint64_t(bias + 5) << width) | 31,
                               (uint64_t(bias - 5) << width) | 31,
                               2,
                               3,
                               4,
                               7,
                               min_normal / 2,
                               min_normal / 2 - 1,
                               inf | 12345,
                               sign | inf | 12345};
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int variant = 0; variant < 32; ++variant)
        for (uint64_t mask : rdna4_exec_masks())
          for (int alias = 0; alias < (fp64 ? 3 : 2); ++alias) {
            SCOPED_TRACE(::testing::Message()
                         << fp64 << "/" << cpu << "/" << variant << "/" << mask << "/" << alias);
            uint32_t storage[3][34];
            for (auto &reg : storage)
              std::fill(reg, reg + 34, 0xdeadbeef);
            uint32_t *a[] = {storage[0] + 1, storage[1] + 1};
            uint32_t *d = storage[alias == 0 ? 2 : alias - 1] + 1;
            for (int lane = 0; lane < 32; ++lane) {
              a[0][lane] = uint32_t(inputs[lane]);
              a[1][lane] = uint32_t(inputs[lane] >> 32);
            }
            std::array<uint32_t, 32> before;
            std::copy(d, d + 32, before.begin());
            ASSERT_EQ(functions[fp64](cpu, mask, mode(variant), &d, a), GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              EXPECT_EQ(d[lane], (mask >> lane & 1) ? reference(inputs[lane], fp64) : before[lane]);
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
  }
}

TEST(FrexpExp, EveryExponentAndSubnormalLeadingBit) {
  for (int fp64 = 0; fp64 < 2; ++fp64) {
    int width = fp64 ? 52 : 23, max_field = fp64 ? 2047 : 255;
    uint64_t sign = UINT64_C(1) << (fp64 ? 63 : 31);
    uint64_t fraction = (UINT64_C(1) << width) - 1;
    std::vector<uint64_t> inputs;
    for (int field = 0; field <= max_field; ++field)
      for (uint64_t tail : {UINT64_C(0), UINT64_C(1), fraction / 2, fraction}) {
        uint64_t value = (uint64_t(field) << width) | tail;
        inputs.push_back(value);
        inputs.push_back(value | sign);
      }
    for (int bit = 0; bit < width; ++bit)
      for (uint64_t value : {UINT64_C(1) << bit, (UINT64_C(1) << bit) - 1}) {
        inputs.push_back(value);
        inputs.push_back(value | sign);
      }
    std::mt19937_64 random(417);
    for (int i = 0; i < 1024; ++i)
      inputs.push_back(random() & (fp64 ? UINT64_MAX : UINT32_MAX));
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int variant = 0; variant < 32; ++variant)
        for (size_t start = 0; start < inputs.size(); start += 32) {
          SCOPED_TRACE(::testing::Message()
                       << fp64 << "/" << cpu << "/" << variant << "/" << start);
          uint32_t a[2][32], d[32];
          uint32_t *pa[] = {a[0], a[1]}, *pd = d;
          for (int lane = 0; lane < 32; ++lane) {
            uint64_t value = inputs[(start + lane) % inputs.size()];
            a[0][lane] = uint32_t(value);
            a[1][lane] = uint32_t(value >> 32);
          }
          ASSERT_EQ(functions[fp64](cpu, UINT32_MAX, mode(variant), &pd, pa), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[lane], reference(inputs[(start + lane) % inputs.size()], fp64));
        }
  }
}

TEST(FrexpExp, LiteralValuesAndValidation) {
  const uint64_t inputs[][8] = {
      {1, 0x007fffff, 0x00800000, 0x3f000000, 0x3f800000, 0x7f7fffff, 0, 0x7f800001},
      {1, 0x000fffffffffffff, 0x0010000000000000, 0x3fe0000000000000, 0x3ff0000000000000,
       0x7fefffffffffffff, 0, 0x7ff0000000000001}};
  const int expected[][8] = {{-148, -126, -125, 0, 1, 128, 0, 0},
                             {-1073, -1022, -1021, 0, 1, 1024, 0, 0}};
  for (int fp64 = 0; fp64 < 2; ++fp64)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[2][32], d[32];
      uint32_t *pa[] = {a[0], a[1]}, *pd = d;
      for (int lane = 0; lane < 32; ++lane) {
        a[0][lane] = uint32_t(inputs[fp64][lane % 8]);
        a[1][lane] = uint32_t(inputs[fp64][lane % 8] >> 32);
      }
      ASSERT_EQ(functions[fp64](cpu, UINT32_MAX, 0, &pd, pa), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(d[lane], uint32_t(expected[fp64][lane % 8]));
      std::fill(d, d + 32, 0xdeadbeef);
      for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
        for (uint32_t invalid : {GOC_ALU_NEG_B, GOC_ALU_ABS_B, GOC_ALU_HIGH_D, UINT32_C(1) << 31})
          EXPECT_EQ(functions[fp64](cpu, mask, invalid, &pd, pa), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[fp64](cpu | (UINT64_C(1) << 63), mask, 0, &pd, pa),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[fp64](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask,
                                  0, &pd, pa),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (uint32_t value : d)
        EXPECT_EQ(value, 0xdeadbeef);
      EXPECT_EQ(functions[fp64](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &pd, pa),
                GOC_SUCCESS);
    }
}

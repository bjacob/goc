// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_reference.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_pk_add_f16);
const Fn functions[] = {goc_rdna4_v_pk_add_f16,     goc_rdna4_v_pk_mul_f16,
                        goc_rdna4_v_pk_min_num_f16, goc_rdna4_v_pk_max_num_f16,
                        goc_rdna4_v_pk_minimum_f16, goc_rdna4_v_pk_maximum_f16};
const uint32_t known = GOC_PK_NEG_LO_A | GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A | GOC_PK_NEG_HI_B |
                       GOC_PK_CLAMP | GOC_PK_LO_A_HIGH | GOC_PK_LO_B_HIGH | GOC_PK_HI_A_LOW |
                       GOC_PK_HI_B_LOW;
const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00,
                           0x3c01, 0x3bff, 0x3800, 0xb800, 0x4000, 0xc000, 0xbc00,
                           0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe12};

uint32_t reference(int op, uint32_t a, uint32_t b, uint32_t mode, bool saturate) {
  uint32_t result = 0;
  for (int half = 0; half < 2; ++half) {
    bool a_high = half ? !(mode & GOC_PK_HI_A_LOW) : bool(mode & GOC_PK_LO_A_HIGH);
    bool b_high = half ? !(mode & GOC_PK_HI_B_LOW) : bool(mode & GOC_PK_LO_B_HIGH);
    double x = goc_test::half_value(uint16_t(a >> (a_high ? 16 : 0)));
    double y = goc_test::half_value(uint16_t(b >> (b_high ? 16 : 0)));
    if (mode & (half ? GOC_PK_NEG_HI_A : GOC_PK_NEG_LO_A))
      x = -x;
    if (mode & (half ? GOC_PK_NEG_HI_B : GOC_PK_NEG_LO_B))
      y = -y;
    double value = goc_test::half_binary(op == 0 ? 0 : op == 1 ? 3 : op + 2, x, y);
    if (mode & GOC_PK_CLAMP)
      value = !(value > 0) ? 0 : std::min(value, 1.0);
    result |= uint32_t(goc_test::half_bits(value, saturate)) << (16 * half);
  }
  return result;
}

void check(uint32_t got, uint32_t want) {
  for (int half = 0; half < 2; ++half) {
    uint16_t actual = uint16_t(got >> (16 * half)), expected = uint16_t(want >> (16 * half));
    if ((expected & 0x7fff) > 0x7c00) {
      EXPECT_GT(actual & 0x7fff, 0x7c00);
    } else {
      EXPECT_EQ(actual, expected);
    }
  }
}

void fill(uint32_t (&words)[3][34]) {
  for (int reg = 0; reg < 3; ++reg) {
    std::fill(words[reg], words[reg] + 34, 0xfacecafe);
    for (int lane = 1; lane <= 32; ++lane)
      words[reg][lane] =
          values[(lane + reg * 7) % 20] | (uint32_t(values[(lane * 3 + reg * 5) % 20]) << 16);
  }
}

void run(int op, uint64_t flags, uint64_t mask, uint64_t mode, int b, int d,
         uint32_t (&words)[3][34]) {
  uint32_t before[3][34];
  std::memcpy(before, words, sizeof(before));
  uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1};
  ASSERT_EQ(functions[op](flags, mask, mode, p + d, p, p + b), GOC_SUCCESS);
  for (int reg = 0; reg < 3; ++reg)
    for (int lane = 0; lane < 34; ++lane) {
      if (reg == d && lane >= 1 && lane <= 32 && ((mask >> (lane - 1)) & 1)) {
        check(words[reg][lane],
              reference(op, before[0][lane], before[b][lane], mode, flags & GOC_FP16_OVFL));
      } else {
        EXPECT_EQ(words[reg][lane], before[reg][lane]);
      }
    }
}

} // namespace

TEST(PackedBinary, AllModifiersAndOverflowPolicies) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true})
        for (uint32_t mode = 0; mode < 8192; ++mode) {
          if (mode & ~known)
            continue;
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << saturate << '/' << mode);
          uint32_t words[3][34];
          fill(words);
          run(op, cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, mode, 1, mode % 3, words);
        }
}

TEST(PackedBinary, SpecialCartesianPairsAndEveryEncoding) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      std::mt19937 random(7231);
      for (unsigned base = 0; base < 512 + 65536; base += 32) {
        SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << base);
        uint32_t words[3][34];
        fill(words);
        for (int lane = 1; lane <= 32; ++lane) {
          unsigned i = base + lane - 1;
          if (base < 512) {
            words[0][lane] = values[i % 20] | (uint32_t(values[(i / 20) % 20]) << 16);
            words[1][lane] = values[(i / 20) % 20] | (uint32_t(values[i % 20]) << 16);
          } else {
            uint32_t code = uint16_t(i - 512);
            words[0][lane] = code | ((65535 - code) << 16);
            words[1][lane] = random();
          }
        }
        run(op, cpu, UINT32_MAX, 0, 1, 2, words);
      }
    }
}

TEST(PackedBinary, MasksAndAllWholeRegisterAliases) {
  const uint32_t modes[] = {0,
                            known,
                            GOC_PK_HI_A_LOW,
                            GOC_PK_HI_B_LOW,
                            GOC_PK_LO_A_HIGH | GOC_PK_HI_A_LOW,
                            GOC_PK_LO_B_HIGH | GOC_PK_HI_B_LOW,
                            GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A,
                            GOC_PK_LO_A_HIGH | GOC_PK_NEG_HI_B | GOC_PK_CLAMP};
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto mode : modes)
        for (auto mask : rdna4_exec_masks())
          for (int b : {0, 1})
            for (int d = 0; d < 3; ++d) {
              SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/' << mask
                                                << '/' << b << '/' << d);
              uint32_t words[3][34];
              fill(words);
              run(op, cpu, mask, mode, b, d, words);
            }
}

TEST(PackedBinary, LiteralHalfSelectionSignedZeroAndNanRules) {
  struct Case {
    uint32_t a, b, mode, want[6];
  };

  const Case cases[] = {
      {0x44004000,
       0x3c004200,
       0,
       {0x45004500, 0x44004600, 0x3c004000, 0x44004200, 0x3c004000, 0x44004200}},
      {0x44004000,
       0x3c004200,
       GOC_PK_HI_A_LOW,
       {0x42004500, 0x40004600, 0x3c004000, 0x40004200, 0x3c004000, 0x40004200}},
      {0x44004000,
       0x3c004200,
       GOC_PK_HI_B_LOW,
       {0x47004500, 0x4a004600, 0x42004000, 0x44004200, 0x42004000, 0x44004200}},
      {0x00008000, 0x80000000, 0, {0, 0x80008000, 0x80008000, 0, 0x80008000, 0}},
      {0x3c007c01,
       0xfe124000,
       0,
       {0x7e007e00, 0x7e007e00, 0x3c004000, 0x3c004000, 0x7e007e00, 0x7e007e00}},
      {0x3c007c01, 0xfe124000, GOC_PK_CLAMP, {0, 0, 0x3c003c00, 0x3c003c00, 0, 0}}};
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (const auto &test : cases)
        for (int d = 0; d < 3; ++d) {
          uint32_t words[3][32];
          uint32_t *p[] = {words[0], words[1], words[2]};
          std::fill(words[0], words[0] + 32, test.a);
          std::fill(words[1], words[1] + 32, test.b);
          std::fill(words[2], words[2] + 32, 0xfacecafe);
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, test.mode, p + d, p, p + 1), GOC_SUCCESS);
          for (auto word : words[d])
            check(word, test.want[op]);
        }
}

TEST(PackedBinary, ValidationAndSemantics) {
  for (auto fn : functions) {
    uint32_t words[32];
    std::fill(words, words + 32, 0xfacecafe);
    auto p = words;
    for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
      for (unsigned bit = 0; bit < 32; ++bit) {
        if ((UINT32_C(1) << bit) & ~known) {
          EXPECT_EQ(fn(0, mask, UINT32_C(1) << bit, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
        }
      }
      EXPECT_EQ(fn(UINT64_C(1) << 63, mask, 0, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, &p, &p, &p),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : words)
      EXPECT_EQ(word, 0xfacecafe);
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_C(0xffffffff00000000), 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p, &p), GOC_SUCCESS);
  }
}

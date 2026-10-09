// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_reference.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_ldexp_f16);

int frexp_exp(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
              const uint32_t *const *a, const uint32_t *const *) {
  return goc_rdna4_v_frexp_exp_i16_f16(flags, mask, mode, d, a);
}

const Fn functions[] = {goc_rdna4_v_ldexp_f16, frexp_exp};
const uint32_t common_modes = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP |
                              GOC_ALU_HIGH_A | GOC_ALU_HIGH_D;

uint16_t reference(int op, uint32_t a, uint32_t b, uint32_t mode, bool saturate) {
  double x = goc_test::half_value(uint16_t(a >> (mode & GOC_ALU_HIGH_A ? 16 : 0)));
  if (op == 1) {
    int exponent = 0;
    if (x != 0 && std::isfinite(x))
      std::frexp(x, &exponent);
    return uint16_t(exponent);
  }
  if (mode & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (mode & GOC_ALU_NEG_A)
    x = -x;
  uint16_t raw = uint16_t(b >> (mode & GOC_ALU_HIGH_B ? 16 : 0));
  int exponent = raw <= INT16_MAX ? int(raw) : int(raw) - 65536;
  double result = std::ldexp(x, exponent);
  // Preserve finite-overflow provenance through subsequent output scaling.
  if (std::isfinite(x) && std::abs(result) > 1e100)
    result = std::copysign(1e100, result);
  const double scales[] = {1, 2, 4, 0.5};
  if (mode & GOC_ALU_OMOD_HALF) {
    if (std::abs(result) < 0x1p-14)
      result = 0;
    else if ((mode & GOC_ALU_OMOD_HALF) == GOC_ALU_OMOD_HALF && std::abs(result) < 0x1p-13)
      result = std::copysign(0.0, result);
  }
  if (mode & GOC_ALU_OMOD_HALF)
    result = goc_test::half_value(goc_test::half_bits(result, saturate));
  result *= scales[(mode >> 6) & 3];
  if (mode & GOC_ALU_CLAMP)
    result = !(result > 0) ? 0 : std::min(result, 1.0);
  return goc_test::half_bits(result, saturate);
}

void check(int op, uint32_t actual, uint32_t before, uint16_t want, uint32_t mode) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  EXPECT_EQ((actual ^ before) & ~(UINT32_C(0xffff) << shift), 0u);
  uint16_t got = uint16_t(actual >> shift);
  if (op == 0 && (want & 0x7fff) > 0x7c00) {
    EXPECT_GT(got & 0x7fff, 0x7c00);
  } else {
    EXPECT_EQ(got, want);
  }
}

uint32_t modifiers(unsigned variant) {
  return (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
         ((variant & 28) << 4) | (variant & 32 ? GOC_ALU_HIGH_A : 0) |
         (variant & 64 ? GOC_ALU_HIGH_D : 0) | (variant & 128 ? GOC_ALU_HIGH_B : 0);
}

} // namespace

TEST(HalfExponent, EveryHalfEncodingAndBoundaryExponents) {
  const int exponents[] = {0, -32768, -65, -25, -24, -15, -1, 1, 15, 24, 25, 65, 32767};
  for (int op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true})
        for (bool high : {false, true})
          for (int exponent : exponents) {
            if (op == 1 && exponent != 0)
              continue;
            uint32_t mode = high ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | GOC_ALU_OMOD_HALF |
                                       (op == 0 ? GOC_ALU_HIGH_B : 0)
                                 : 0;
            for (unsigned base = 0; base < 65536; base += 32) {
              SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << saturate << '/'
                                                << mode << '/' << exponent << '/' << base);
              uint32_t words[3][32];
              uint32_t *p[] = {words[0], words[1], words[2]};
              for (int lane = 0; lane < 32; ++lane) {
                auto bits = base + lane;
                words[0][lane] = high ? (bits << 16) | 0xbeef : 0xdead0000 | bits;
                words[1][lane] = high ? (uint32_t(uint16_t(exponent)) << 16) | 0x1234
                                      : 0xabcd0000 | uint16_t(exponent);
                words[2][lane] = 0xdeadbeef;
              }
              ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, mode, p + 2,
                                      p, p + 1),
                        GOC_SUCCESS);
              for (int lane = 0; lane < 32; ++lane)
                check(op, words[2][lane], 0xdeadbeef,
                      reference(op, words[0][lane], words[1][lane], mode, saturate), mode);
            }
          }
}

TEST(HalfExponent, EverySignedExponent) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3bff, 0x3c00,
                             0xbc00, 0x4001, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool saturate : {false, true})
      for (bool high : {false, true}) {
        uint32_t mode = high ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D : 0;
        for (unsigned base = 0; base < 65536; base += 32) {
          uint32_t words[3][32];
          uint32_t *p[] = {words[0], words[1], words[2]};
          for (int lane = 0; lane < 32; ++lane) {
            auto bits = values[lane % 16];
            words[0][lane] = high ? (uint32_t(bits) << 16) | 0xbeef : 0xdead0000 | bits;
            words[1][lane] = high ? ((base + lane) << 16) | 0x8000 : 0x7fff0000 | (base + lane);
            words[2][lane] = 0xdeadbeef;
          }
          ASSERT_EQ(goc_rdna4_v_ldexp_f16(cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, mode,
                                          p + 2, p, p + 1),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            check(0, words[2][lane], 0xdeadbeef,
                  reference(0, words[0][lane], words[1][lane], mode, saturate), mode);
        }
      }
}

TEST(HalfExponent, AllModifiersMasksAndAliases) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3800, 0xb800,
                             0x3bff, 0x3c00, 0x3e00, 0xbe00, 0x4100, 0xc100, 0x4200, 0x4400,
                             0x4c00, 0xcc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};
  const int exponents[] = {-32768, -65, -25, -24, -15, -1, 0, 1, 15, 24, 25, 32767};
  for (int op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < (op == 0 ? 256u : 128u); ++variant)
        for (bool saturate : {false, true})
          for (uint32_t mask : rdna4_exec_masks())
            for (int layout = 0; layout < (op == 0 ? 5 : 2); ++layout) {
              auto mode = modifiers(variant);
              SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/'
                                                << saturate << '/' << mask << '/' << layout);
              uint32_t words[3][34], before[3][34];
              for (int reg = 0; reg < 3; ++reg) {
                std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  words[reg][lane] =
                      reg == 1 ? uint16_t(exponents[lane % 12]) |
                                     (uint32_t(uint16_t(exponents[(lane + 5) % 12])) << 16)
                               : values[(lane + reg * 5) % 24] |
                                     (uint32_t(values[(lane * 7 + reg) % 24]) << 16);
              }
              std::memcpy(before, words, sizeof(words));
              int dest = layout % 3 == 0 ? 2 : layout % 3 - 1;
              int b = layout >= 3 ? 0 : 1;
              uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1};
              ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), mask, mode, p + dest, p,
                                      p + b),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 3; ++reg)
                for (int lane = 0; lane < 34; ++lane) {
                  if (reg == dest && lane >= 1 && lane <= 32 && ((mask >> (lane - 1)) & 1)) {
                    check(op, words[reg][lane], before[reg][lane],
                          reference(op, before[0][lane], before[b][lane], mode, saturate), mode);
                  } else {
                    EXPECT_EQ(words[reg][lane], before[reg][lane]);
                  }
                }
            }
}

TEST(HalfExponent, LiteralRoundingAndOverflow) {
  struct Case {
    uint16_t a, b, result;
    uint32_t mode;
    bool saturate;
  };

  const Case cases[] = {{1, 0xffff, 0, 0, false},
                        {3, 0xffff, 2, 0, false},
                        {0x8001, 0xffff, 0x8000, 0, false},
                        {1, 24, 0x3c00, 0, false},
                        {0x3c00, uint16_t(-24), 1, 0, false},
                        {0x3c00, uint16_t(-25), 0, 0, false},
                        {0x3c00, uint16_t(-25), 0, GOC_ALU_OMOD_2, false},
                        {0x7bff, 1, 0x7c00, 0, false},
                        {0x7bff, 1, 0x7bff, 0, true},
                        {0xfbff, 0x7fff, 0xfbff, 0, true},
                        {0x7bff, 1, 0x7c00, GOC_ALU_OMOD_HALF, false},
                        {0x7c00, 0x8000, 0x7c00, 0, true},
                        {0x8000, 0x7fff, 0x8000, 0, false},
                        {0x7c01, 0x8000, 0, GOC_ALU_CLAMP, false}};
  const uint16_t exponent_cases[][2] = {{1, 0xffe9}, {0x3ff, 0xfff2}, {0x400, 0xfff3}, {0x3c00, 1},
                                        {0xbc00, 1}, {0x7bff, 16},    {0, 0},          {0x8000, 0},
                                        {0x7c00, 0}, {0xfc01, 0}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    for (const auto &test : cases) {
      uint32_t words[3][32];
      uint32_t *p[] = {words[0], words[1], words[2]};
      std::fill(words[0], words[0] + 32, test.a);
      std::fill(words[1], words[1] + 32, test.b);
      std::fill(words[2], words[2] + 32, 0xdeadbeef);
      ASSERT_EQ(goc_rdna4_v_ldexp_f16(cpu | (test.saturate ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                      test.mode, p + 2, p, p + 1),
                GOC_SUCCESS);
      for (auto word : words[2])
        EXPECT_EQ(word, UINT32_C(0xdead0000) | test.result);
    }
    for (const auto &test : exponent_cases) {
      uint32_t a[32], d[32];
      std::fill(a, a + 32, test[0]);
      std::fill(d, d + 32, 0xdeadbeef);
      auto pa = a, pd = d;
      ASSERT_EQ(goc_rdna4_v_frexp_exp_i16_f16(cpu, UINT32_MAX, 0, &pd, &pa), GOC_SUCCESS);
      for (auto word : d)
        EXPECT_EQ(word, UINT32_C(0xdead0000) | test[1]);
    }
  }
}

TEST(HalfExponent, FrexpPreservesFpEnvironment) {
  std::fenv_t saved;
  std::fegetenv(&saved);
  const uint16_t bits[] = {0x7c01, 0xfc01, 0x7c00, 0x8000, 1, 0x3ff, 0x400, 0x7bff};
  const uint16_t golden[] = {0, 0, 0, 0, 0xffe9, 0xfff2, 0xfff3, 16};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      uint32_t input[32], output[32];
      for (int lane = 0; lane < 32; ++lane) {
        input[lane] = (uint32_t(bits[lane % 8]) << 16) | 0xbeef;
        output[lane] = 0xdeadbeef;
      }
      auto a = input, d = output;
      std::fesetround(rounding);
      std::feclearexcept(FE_ALL_EXCEPT);
      std::feraiseexcept(FE_DIVBYZERO);
      EXPECT_EQ(
          goc_rdna4_v_frexp_exp_i16_f16(cpu | GOC_FP16_OVFL, UINT32_MAX, common_modes, &d, &a),
          GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(output[lane], (uint32_t(golden[lane % 8]) << 16) | 0xbeef);
    }
  std::fesetenv(&saved);
}

TEST(HalfExponent, ValidationAndSemantics) {
  for (int op = 0; op < 2; ++op) {
    uint32_t data[32];
    std::fill(data, data + 32, 0xdeadbeef);
    auto p = data;
    uint32_t known = common_modes | (op == 0 ? GOC_ALU_HIGH_B : 0);
    for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
      for (int bit = 0; bit < 32; ++bit) {
        if (!(known & (UINT32_C(1) << bit))) {
          EXPECT_EQ(functions[op](0, mask, UINT32_C(1) << bit, &p, &p, &p),
                    GOC_ERROR_INVALID_FLAGS);
        }
      }
      EXPECT_EQ(functions[op](UINT64_C(1) << 63, mask, 0, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(
          functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, &p, &p, &p),
          GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : data)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p, &p), GOC_SUCCESS);
  }
}

TEST(HalfExponent, DppMasksAndAliases) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3800, 0xb800,
                             0x3bff, 0x3c00, 0x3e00, 0xbe00, 0x4100, 0xc100, 0x4200, 0x4400,
                             0x4c00, 0xcc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};
  const int exponents[] = {-32768, -65, -25, -24, -15, -1, 0, 1, 15, 24, 25, 32767};
  for (int op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant : {0u, 127u, op == 0 ? 255u : 73u})
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (bool saturate : {false, true})
            for (uint32_t mask : rdna4_exec_masks())
              for (int layout = 0; layout < (op == 0 ? 5 : 2); ++layout) {
                auto mode = modifiers(variant);
                SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/'
                                                  << saturate << '/' << mask << '/' << layout);
                uint32_t words[3][34], before[3][34];
                for (int reg = 0; reg < 3; ++reg) {
                  std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                  for (int lane = 1; lane <= 32; ++lane)
                    words[reg][lane] =
                        reg == 1 ? uint16_t(exponents[lane % 12]) |
                                       (uint32_t(uint16_t(exponents[(lane + 5) % 12])) << 16)
                                 : values[(lane + reg * 5) % 24] |
                                       (uint32_t(values[(lane * 7 + reg) % 24]) << 16);
                }
                std::memcpy(before, words, sizeof(words));
                int dest = layout % 3 == 0 ? 2 : layout % 3 - 1;
                int b = layout >= 3 ? 0 : 1;
                uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1};
                ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), mask,
                                        descriptor | mode, p + dest, p, p + b),
                          GOC_SUCCESS);
                for (int reg = 0; reg < 3; ++reg)
                  for (int lane = 0; lane < 34; ++lane) {
                    int source = -1;
                    if (reg == dest && lane >= 1 && lane <= 32 &&
                        goc_test::dpp_source(descriptor, mask, lane - 1, source)) {
                      check(op, words[reg][lane], before[reg][lane],
                            reference(op, source < 0 ? 0 : before[0][source + 1], before[b][lane],
                                      mode, saturate),
                            mode);
                    } else {
                      EXPECT_EQ(words[reg][lane], before[reg][lane]);
                    }
                  }
              }
}

TEST(HalfExponent, HardwareRoundingCorpus) {
  // GFX1201: every FP16 encoding, five exponents, four OMOD values, both
  // overflow settings. Canonicalize NaN payloads, preserving the other half.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (bool saturate : {false, true})
      for (int exponent : {-25, -1, 0, 1, 24})
        for (unsigned omod = 0; omod < 4; ++omod)
          for (unsigned base = 0; base < 65536; base += 32) {
            uint32_t a[32], b[32], d[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              a[lane] = base + lane;
              b[lane] = uint16_t(exponent);
              d[lane] = 0xdeadbeef;
            }
            auto pa = a, pb = b, pd = d;
            ASSERT_EQ(goc_rdna4_v_ldexp_f16(cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                            omod << 6, &pd, &pa, &pb),
                      GOC_SUCCESS);
            for (uint32_t word : d) {
              if ((word & 0x7fff) > 0x7c00)
                word = (word & 0xffff0000u) | 0x7e00;
              hash = (hash ^ word) * UINT64_C(1099511628211);
            }
          }
    EXPECT_EQ(hash, UINT64_C(0x62b2fe854d088725));
  }
}

TEST(HalfExponent, DppHardwareCorpus) {
  const uint32_t values[] = {0x00018001, 0x03ff83ff, 0x04008400, 0x04018401,
                             0x08008800, 0x3c00bc00, 0x00008000, 0x7bfffbff};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint32_t mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned op = 0; op < 2; ++op)
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (uint32_t mode : {0u, 1u, 8u, 9u, 64u, 128u, 192u, 256u, 512u, 4096u, 4608u, 5065u}) {
            uint32_t a[32], b[32], d[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              a[lane] = values[lane % 8];
              b[lane] = lane % 5 - 2;
              d[lane] = 0xdead0000u + lane;
            }
            auto pa = a, pb = b, pd = d;
            ASSERT_EQ(functions[op](cpu, mask, descriptor | mode, &pd, &pa, &pb), GOC_SUCCESS);
            for (auto word : d)
              hash = (hash ^ word) * UINT64_C(1099511628211);
          }
    EXPECT_EQ(hash, UINT64_C(0x1af7c9e0f5f41bf6));
  }
}

TEST(HalfExponent, DppEveryModifier) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < (op == 0 ? 256u : 128u); ++variant)
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (bool saturate : {false, true}) {
            uint32_t a[32], b[32], d[32];
            auto mode = modifiers(variant);
            for (unsigned lane = 0; lane < 32; ++lane) {
              a[lane] = (lane * 2039u) | ((65535u - lane * 2039u) << 16);
              b[lane] = uint16_t(int(lane) - 16) | (uint32_t(uint16_t(16 - int(lane))) << 16);
              d[lane] = 0xdeadbeef;
            }
            auto pa = a, pb = b, pd = d;
            ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                    descriptor | mode, &pd, &pa, &pb),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              int source;
              if (goc_test::dpp_source(descriptor, UINT32_MAX, lane, source))
                check(op, d[lane], 0xdeadbeef,
                      reference(op, source < 0 ? 0 : a[source], b[lane], mode, saturate), mode);
              else
                EXPECT_EQ(d[lane], 0xdeadbeef);
            }
          }
}

TEST(HalfExponent, DppValidation) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(functions[op](0, 0, descriptor, nullptr, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(functions[op](0, UINT32_MAX, descriptor | GOC_ALU_NEG_B, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(
          functions[op](0, UINT32_MAX, descriptor | (UINT64_C(1) << 36), nullptr, nullptr, nullptr),
          GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                              descriptor, nullptr, nullptr, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
}

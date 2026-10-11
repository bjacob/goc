// SPDX-License-Identifier: MIT

#include "exec_masks.h"
#include "fp_environment.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "loose_tolerances.h"
#include "omod_reference.h"
#include "test_instruction.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn = decltype(&goc_v_sqrt_f64_rdna4);
const Fn functions[] = {goc_v_trunc_f64_rdna4, goc_v_ceil_f64_rdna4,  goc_v_rndne_f64_rdna4,
                        goc_v_floor_f64_rdna4, goc_v_fract_f64_rdna4, goc_v_sqrt_f64_rdna4,
                        goc_v_rcp_f64_rdna4,   goc_v_rsq_f64_rdna4,   goc_v_frexp_mant_f64_rdna4};
const decltype(&goc_v_sqrt_f64_rdna4_wave64) wave64[] = {
    goc_v_trunc_f64_rdna4_wave64, goc_v_ceil_f64_rdna4_wave64,  goc_v_rndne_f64_rdna4_wave64,
    goc_v_floor_f64_rdna4_wave64, goc_v_fract_f64_rdna4_wave64, goc_v_sqrt_f64_rdna4_wave64,
    goc_v_rcp_f64_rdna4_wave64,   goc_v_rsq_f64_rdna4_wave64,   goc_v_frexp_mant_f64_rdna4_wave64};

uint64_t bits(double value) {
  uint64_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

double number(uint64_t value) {
  double result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

double reference(int op, uint64_t input, uint32_t mode) {
  long double x = number(input);
  if (mode & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (mode & GOC_ALU_NEG_A)
    x = -x;
  long double y = 0;
  switch (op) {
  case 8: {
    int exponent;
    y = std::isfinite(x) ? std::frexp(x, &exponent) : x;
    break;
  }
  case 0:
    y = std::trunc(x);
    break;
  case 1:
    y = std::ceil(x);
    break;
  case 2:
    y = x;
    if (std::isfinite(x) && x != 0) {
      long double lo = std::floor(x), fraction = x - lo;
      y = lo + (fraction > 0.5L || (fraction == 0.5L && std::fmod(lo, 2.0L) != 0));
      y = std::copysign(std::abs(y), x);
    }
    break;
  case 3:
    y = std::floor(x);
    break;
  case 4:
    y = x - std::floor(x);
    if (y > number(0x3fefffffffffffffULL))
      y = number(0x3fefffffffffffffULL);
    break;
  case 5:
    y = std::sqrt(x);
    break;
  case 6:
    y = 1 / x;
    break;
  case 7:
    y = 1 / std::sqrt(x);
    break;
  }
  double result = double(y);
  result = goc_test::omod_f64_reference(result, mode);
  if (mode & GOC_ALU_CLAMP)
    result = !(result > 0) ? 0 : std::min(result, 1.0);
  return result;
}

} // namespace

TEST(Fp64Unary, AllModifiersMasksAndCrossHalfAliases) {
  const uint64_t inputs[] = {0,
                             0x8000000000000000,
                             1,
                             0x8000000000000001,
                             0x0010000000000000,
                             0x8010000000000000,
                             0x3fe0000000000000,
                             0xbfe0000000000000,
                             0x3ff8000000000000,
                             0xbff8000000000000,
                             0x4004000000000000,
                             0xc004000000000000,
                             0x400c000000000000,
                             0xc00c000000000000,
                             0x3fefffffffffffff,
                             0x3ff0000000000001,
                             0x7fefffffffffffff,
                             0xffefffffffffffff,
                             0x7ff0000000000000,
                             0xfff0000000000000,
                             0x7ff8000000001234,
                             0xfff8000000001234,
                             0x7ff0000000001234,
                             0xfff0000000001234,
                             0x4330000000000001,
                             0xc330000000000001,
                             0x3fe0000000000001,
                             0x3fdfffffffffffff,
                             0x4010000000000000,
                             0x3fd0000000000000,
                             0x000fffffffffffff,
                             0x800fffffffffffff};
  const int aliases[][2] = {{2, 3}, {0, 1}, {1, 0}, {1, 2}, {2, 0}, {2, 2}};
  for (int op = 0; op < 9; ++op)
    for (uint32_t variant = 0; variant < 32; ++variant) {
      uint32_t mode = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
                      ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
      double expected[32];
      for (int lane = 0; lane < 32; ++lane)
        expected[lane] = reference(op, inputs[lane], mode);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        // Identical output buffers retain only the high word, so compare that
        // alias case to a disjoint execution. A few allowed numeric ULPs can
        // straddle a high-word boundary in the higher-precision reference.
        uint32_t source[2][32], separate[2][32];
        uint32_t *pa[] = {source[0], source[1]}, *pd[] = {separate[0], separate[1]};
        for (int lane = 0; lane < 32; ++lane) {
          source[0][lane] = uint32_t(inputs[lane]);
          source[1][lane] = uint32_t(inputs[lane] >> 32);
        }
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, mode, pd, pa, nullptr), GOC_SUCCESS);
        for (uint32_t exec_mask : exec_masks())
          for (const auto &alias : aliases) {
            SCOPED_TRACE(::testing::Message() << op << "/" << mode << "/" << cpu << "/" << exec_mask
                                              << "/" << alias[0] << "/" << alias[1]);
            uint32_t storage[4][34], before[2][32], *v[4];
            for (int reg = 0; reg < 4; ++reg) {
              std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
              if (reg < 2)
                for (int lane = 0; lane < 32; ++lane)
                  v[reg][lane] = uint32_t(inputs[lane] >> (32 * reg));
            }
            uint32_t *d[] = {v[alias[0]], v[alias[1]]};
            for (int reg = 0; reg < 2; ++reg)
              std::copy(d[reg], d[reg] + 32, before[reg]);
            ASSERT_EQ(functions[op](cpu, exec_mask, mode, d, v, nullptr), GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              if (!((exec_mask >> lane) & 1)) {
                EXPECT_EQ(d[0][lane], before[0][lane]);
                EXPECT_EQ(d[1][lane], before[1][lane]);
                continue;
              }
              uint64_t got = d[0][lane] | (uint64_t(d[1][lane]) << 32);
              if (std::isnan(expected[lane])) {
                EXPECT_TRUE(std::isnan(number(got)));
              } else if (alias[0] == alias[1]) {
                EXPECT_EQ(d[0][lane], separate[1][lane]);
              } else if (op < 5 || expected[lane] == 0 || std::isinf(expected[lane])) {
                EXPECT_EQ(got, bits(expected[lane]));
              } else {
                EXPECT_TRUE(
                    goc_test::fuzzy(number(got), expected[lane], goc_test::unary_f64_tolerance));
              }
            }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
      }
    }
}

TEST(Fp64Unary, LiteralBoundaries) {
  struct Case {
    int op;
    uint64_t input, want;
  };

  const Case cases[] = {
      {2, 0x3fe0000000000000, 0},
      {2, 0xbfe0000000000000, 0x8000000000000000},
      {2, 0x3ff8000000000000, 0x4000000000000000},
      {2, 0x4004000000000000, 0x4000000000000000},
      {2, 0xc004000000000000, 0xc000000000000000},
      {2, 0x400c000000000000, 0x4010000000000000},
      {2, 0x7ff0000000001234, 0x7ff8000000001234},
      {4, 0x8000000000000000, 0},
      {4, 0x8000000000000001, 0x3fefffffffffffff},
      {4, 1, 1},
      {4, 0xbfefffffffffffff, 0x3ca0000000000000},
      {5, 0x8000000000000000, 0x8000000000000000},
      {5, 1, 0x1e60000000000000},
      {5, 0x4010000000000000, 0x4000000000000000},
      {6, 0x7ff0000000000000, 0},
      {6, 0xfff0000000000000, 0x8000000000000000},
      {6, 0, 0x7ff0000000000000},
      {6, 0x8000000000000000, 0xfff0000000000000},
      {6, 0x0010000000000000, 0x7fd0000000000000},
      {7, 0x8000000000000000, 0xfff0000000000000},
      {7, 0x4010000000000000, 0x3fe0000000000000},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t a[2][32], d[2][32];
      uint32_t *pa[] = {a[0], a[1]}, *pd[] = {d[0], d[1]};
      std::fill(a[0], a[0] + 32, uint32_t(test.input));
      std::fill(a[1], a[1] + 32, uint32_t(test.input >> 32));
      ASSERT_EQ(functions[test.op](cpu, UINT32_MAX, 0, pd, pa, nullptr), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(d[0][lane] | (uint64_t(d[1][lane]) << 32), test.want)
            << cpu << "/" << test.op << "/" << test.input;
    }
}

TEST(Fp64Unary, Validation) {
  uint32_t a[2][32] = {}, d[2][32];
  uint32_t *pa[] = {a[0], a[1]}, *pd[] = {d[0], d[1]};
  for (auto fn : functions) {
    for (auto &reg : d)
      std::fill(reg, reg + 32, 0xdeadbeef);
    EXPECT_EQ(fn(0, UINT32_MAX, GOC_ALU_NEG_B, pd, pa, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, 0, GOC_ALU_HIGH_D, pd, pa, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, pd, pa, nullptr),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0U, 0, pd, pa, nullptr), GOC_SUCCESS);
    for (auto &reg : d)
      for (uint32_t value : reg)
        EXPECT_EQ(value, 0xdeadbeef);
  }
}

TEST(Fp64Unary, MantissaLiteralSubnormalsAndPassthrough) {
  const uint64_t cases[][2] = {{1, 0x3fe0000000000000},
                               {0x8000000000000001, 0xbfe0000000000000},
                               {0x000fffffffffffff, 0x3feffffffffffffe},
                               {0x0010000000000000, 0x3fe0000000000000},
                               {0x4018000000000000, 0x3fe8000000000000},
                               {0, 0},
                               {0x8000000000000000, 0x8000000000000000},
                               {0x7ff0000000000000, 0x7ff0000000000000},
                               {0xfff0000000000000, 0xfff0000000000000},
                               {0x7ff0000000001234, 0x7ff0000000001234},
                               {0xfff8000000001234, 0xfff8000000001234}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t a[2][32], d[2][32];
      uint32_t *pa[] = {a[0], a[1]}, *pd[] = {d[0], d[1]};
      std::fill(a[0], a[0] + 32, uint32_t(test[0]));
      std::fill(a[1], a[1] + 32, uint32_t(test[0] >> 32));
      ASSERT_EQ(goc_v_frexp_mant_f64_rdna4(cpu, UINT32_MAX, 0, pd, pa, nullptr), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(d[0][lane] | (uint64_t(d[1][lane]) << 32), test[1]);
    }
}

TEST(Fp64Unary, Wave64AllModifiersMasksAndCrossWordAliases) {
  const uint64_t inputs[] = {0,
                             0x8000000000000000,
                             1,
                             0x8000000000000001,
                             0x0010000000000000,
                             0x8010000000000000,
                             0x3fe0000000000000,
                             0xbfe0000000000000,
                             0x3ff8000000000000,
                             0xbff8000000000000,
                             0x4004000000000000,
                             0xc004000000000000,
                             0x400c000000000000,
                             0xc00c000000000000,
                             0x3fefffffffffffff,
                             0x3ff0000000000001,
                             0x7fefffffffffffff,
                             0xffefffffffffffff,
                             0x7ff0000000000000,
                             0xfff0000000000000,
                             0x7ff8000000001234,
                             0xfff8000000001234,
                             0x7ff0000000001234,
                             0xfff0000000001234,
                             0x4330000000000001,
                             0xc330000000000001,
                             0x3fe0000000000001,
                             0x3fdfffffffffffff,
                             0x4010000000000000,
                             0x3fd0000000000000,
                             0x000fffffffffffff,
                             0x800fffffffffffff};
  const int aliases[][2] = {{2, 3}, {0, 1}, {1, 0}, {1, 2}, {2, 0}, {2, 2}};
  for (int op = 0; op < 9; ++op)
    for (uint32_t variant = 0; variant < 32; ++variant) {
      uint32_t mode = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
                      ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
      double expected[64];
      for (int lane = 0; lane < 64; ++lane)
        expected[lane] = reference(op, inputs[(lane + (lane / 32) * 7) % 32], mode);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        // Identical output buffers retain only the high word, so compare that
        // alias case to a disjoint execution. A few allowed numeric ULPs can
        // straddle a high-word boundary in the higher-precision reference.
        uint32_t source[2][64], separate[2][64];
        uint32_t *pa[] = {source[0], source[1]}, *pd[] = {separate[0], separate[1]};
        for (int lane = 0; lane < 64; ++lane) {
          source[0][lane] = uint32_t(inputs[(lane + (lane / 32) * 7) % 32]);
          source[1][lane] = uint32_t(inputs[(lane + (lane / 32) * 7) % 32] >> 32);
        }
        ASSERT_EQ(wave64[op](cpu, UINT64_MAX, mode, pd, pa, nullptr), GOC_SUCCESS);
        for (uint64_t exec_mask :
             {UINT64_MAX, uint64_t(0xffffffff00000000ULL), uint64_t(0x87654321fedcba98ULL)})
          for (const auto &alias : aliases) {
            SCOPED_TRACE(::testing::Message() << op << "/" << mode << "/" << cpu << "/" << exec_mask
                                              << "/" << alias[0] << "/" << alias[1]);
            uint32_t storage[4][66], before[2][64], *v[4];
            for (int reg = 0; reg < 4; ++reg) {
              std::fill(storage[reg], storage[reg] + 66, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
              if (reg < 2)
                for (int lane = 0; lane < 64; ++lane)
                  v[reg][lane] = uint32_t(inputs[(lane + (lane / 32) * 7) % 32] >> (32 * reg));
            }
            uint32_t *d[] = {v[alias[0]], v[alias[1]]};
            for (int reg = 0; reg < 2; ++reg)
              std::copy(d[reg], d[reg] + 64, before[reg]);
            ASSERT_EQ(wave64[op](cpu, exec_mask, mode, d, v, nullptr), GOC_SUCCESS);
            for (int lane = 0; lane < 64; ++lane) {
              if (!((exec_mask >> lane) & 1)) {
                EXPECT_EQ(d[0][lane], before[0][lane]);
                EXPECT_EQ(d[1][lane], before[1][lane]);
                continue;
              }
              uint64_t got = d[0][lane] | (uint64_t(d[1][lane]) << 32);
              if (std::isnan(expected[lane])) {
                EXPECT_TRUE(std::isnan(number(got)));
              } else if (alias[0] == alias[1]) {
                EXPECT_EQ(d[0][lane], separate[1][lane]);
              } else if (op < 5 || expected[lane] == 0 || std::isinf(expected[lane])) {
                EXPECT_EQ(got, bits(expected[lane]));
              } else {
                EXPECT_TRUE(
                    goc_test::fuzzy(number(got), expected[lane], goc_test::unary_f64_tolerance));
              }
            }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[65], 0xdeadbeef);
            }
          }
      }
    }
}

TEST(Fp64Unary, Wave64ReportingAndValidation) {
  for (auto fn : wave64) {
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, GOC_ALU_NEG_B, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    uint32_t exceptions = 0x80000055;
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, 0, nullptr, nullptr, &exceptions),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(exceptions, 0x80000055U);
  }
}

namespace {

using UnaryInstruction = goc_test::WaveInstruction<decltype(&goc_v_cvt_f64_i32)>;
const UnaryInstruction architecture_functions[2][2][9] = {
    {
        {goc_v_trunc_f64, goc_v_ceil_f64, goc_v_rndne_f64, goc_v_floor_f64, goc_v_fract_f64,
         goc_v_sqrt_f64, goc_v_rcp_f64, goc_v_rsq_f64, goc_v_frexp_mant_f64},
        {goc_v_trunc_f64_wave64, goc_v_ceil_f64_wave64, goc_v_rndne_f64_wave64,
         goc_v_floor_f64_wave64, goc_v_fract_f64_wave64, goc_v_sqrt_f64_wave64,
         goc_v_rcp_f64_wave64, goc_v_rsq_f64_wave64, goc_v_frexp_mant_f64_wave64},
    },
    {
        {goc_v_trunc_f64_rdna4, goc_v_ceil_f64_rdna4, goc_v_rndne_f64_rdna4, goc_v_floor_f64_rdna4,
         goc_v_fract_f64_rdna4, goc_v_sqrt_f64_rdna4, goc_v_rcp_f64_rdna4, goc_v_rsq_f64_rdna4,
         goc_v_frexp_mant_f64_rdna4},
        {goc_v_trunc_f64_rdna4_wave64, goc_v_ceil_f64_rdna4_wave64, goc_v_rndne_f64_rdna4_wave64,
         goc_v_floor_f64_rdna4_wave64, goc_v_fract_f64_rdna4_wave64, goc_v_sqrt_f64_rdna4_wave64,
         goc_v_rcp_f64_rdna4_wave64, goc_v_rsq_f64_rdna4_wave64, goc_v_frexp_mant_f64_rdna4_wave64},
    },
};

uint64_t unary_policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

uint32_t unary_modifiers(unsigned m) {
  return (m & 1 ? GOC_ALU_NEG_A : 0) | (m & 2 ? GOC_ALU_ABS_A : 0) | ((m >> 2 & 3) << 6) |
         (m & 16 ? GOC_ALU_CLAMP : 0);
}

} // namespace

TEST(Fp64Unary, ArchitectureLiteralOutputScaling) {
  const uint64_t input[] = {0x400a000000000000ULL, 0x400a000000000000ULL, 0x400a000000000000ULL,
                            0x400a000000000000ULL, 0x400a000000000000ULL, 0x4010000000000000ULL,
                            0x4010000000000000ULL, 0x4010000000000000ULL, 0x4008000000000000ULL};
  const double unscaled[] = {3, 4, 3, 3, .25, 2, .25, .5, .75};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 9; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t a[2][64], d[2][64];
            for (unsigned lane = 0; lane < 64; ++lane) {
              a[0][lane] = uint32_t(input[op]);
              a[1][lane] = uint32_t(input[op] >> 32);
              d[0][lane] = d[1][lane] = 0xdeadbeef;
            }
            const uint32_t *pa[] = {a[0], a[1]};
            uint32_t *pd[] = {d[0], d[1]}, state = 0x80000055;
            ASSERT_EQ(architecture_functions[arch][wide][op](cpu | unary_policy(guest), UINT64_MAX,
                                                             GOC_ALU_OMOD_2, pd, pa, &state),
                      GOC_SUCCESS);
            bool scale = arch || ((guest & 2) && !(guest & 4));
            uint64_t want = bits(unscaled[op] * (scale ? 2 : 1));
            for (unsigned lane = 0; lane < 64; ++lane)
              for (unsigned reg = 0; reg < 2; ++reg)
                EXPECT_EQ(d[reg][lane],
                          lane < (wide ? 64U : 32U) ? uint32_t(want >> (reg * 32)) : 0xdeadbeefU);
            EXPECT_EQ(state, 0x80000055U);
          }
}

TEST(Fp64Unary, ArchitecturePoliciesModifiersAndAliases) {
  const uint64_t inputs[] = {0,
                             0x8000000000000000ULL,
                             1,
                             0x8000000000000001ULL,
                             0xfffffffffffffULL,
                             0x10000000000000ULL,
                             0x10000000000001ULL,
                             0x3ff0000000000001ULL,
                             0x3ff8000000000000ULL,
                             0xbfefffffffffffffULL,
                             0x7fefffffffffffffULL,
                             0x7ff0000000000000ULL,
                             0xfff0000000000000ULL,
                             0x7ff0000000000001ULL,
                             0xfff0000000000001ULL,
                             0x7ff8000000004321ULL};
  const unsigned aliases[][2] = {{2, 3}, {0, 1}, {1, 0}, {2, 2}, {0, 2}};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 9; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned m = 0; m < 32; ++m)
            for (const auto &alias : aliases)
              for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
                uint32_t words[4][66], before[4][66];
                for (unsigned lane = 0; lane < 66; ++lane) {
                  uint64_t input = inputs[(lane + lane / 32) % 16];
                  words[0][lane] = uint32_t(input);
                  words[1][lane] = uint32_t(input >> 32);
                  words[2][lane] = words[3][lane] = 0xdeadbeef;
                }
                std::memcpy(before, words, sizeof(words));
                const uint32_t *a[] = {words[0] + 1, words[1] + 1};
                uint32_t *d[] = {words[alias[0]] + 1, words[alias[1]] + 1};
                const uint64_t exec_mask = 0xf13579bde2468acfULL;
                uint32_t mode = unary_modifiers(m);
                uint32_t effective =
                    !arch && (!(guest & 2) || (guest & 4)) ? mode & ~GOC_ALU_OMOD_HALF : mode;
                uint32_t separate[2][64];
                if (alias[0] == alias[1]) {
                  uint32_t *separate_d[] = {separate[0], separate[1]};
                  ASSERT_EQ(architecture_functions[arch][wide][op](cpu | unary_policy(guest),
                                                                   exec_mask, mode, separate_d, a),
                            GOC_SUCCESS);
                }
                ASSERT_EQ(architecture_functions[arch][wide][op](cpu | unary_policy(guest),
                                                                 exec_mask, mode, d, a),
                          GOC_SUCCESS);
                for (unsigned lane = 0; lane < 64; ++lane) {
                  bool active = lane < (wide ? 64U : 32U) && ((exec_mask >> lane) & 1);
                  if (!active) {
                    EXPECT_EQ(d[0][lane], before[alias[0]][lane + 1]);
                    EXPECT_EQ(d[1][lane], before[alias[1]][lane + 1]);
                    continue;
                  }
                  uint64_t input =
                      uint64_t(before[0][lane + 1]) | (uint64_t(before[1][lane + 1]) << 32);
                  double want = reference(op, input, effective);
                  uint64_t got = uint64_t(d[0][lane]) | (uint64_t(d[1][lane]) << 32);
                  if (alias[0] == alias[1]) {
                    // Compare with a disjoint invocation: the loose SFU result can
                    // differ from the reference by an ULP that crosses a word boundary.
                    EXPECT_EQ(d[0][lane], separate[1][lane]);
                  } else if (std::isnan(want)) {
                    EXPECT_TRUE(std::isnan(number(got)));
                  } else if (op < 5 || want == 0 || std::isinf(want)) {
                    EXPECT_EQ(got, bits(want));
                  } else {
                    EXPECT_TRUE(goc_test::fuzzy(number(got), want, goc_test::unary_f64_tolerance));
                  }
                }
                for (unsigned reg = 0; reg < 4; ++reg) {
                  EXPECT_EQ(words[reg][0], before[reg][0]);
                  EXPECT_EQ(words[reg][65], before[reg][65]);
                  if (reg != alias[0] && reg != alias[1]) {
                    EXPECT_EQ(std::memcmp(words[reg], before[reg], sizeof(words[reg])), 0);
                  }
                }
              }
}

TEST(Fp64Unary, ArchitectureValidationAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (const auto &arch : architecture_functions)
    for (const auto &wave : arch)
      for (auto instruction : wave)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t a[2][64] = {}, d[2][64], state = 0x80000055;
            for (auto &reg : d)
              for (auto &word : reg)
                word = 0xdeadbeef;
            const uint32_t *pa[] = {a[0], a[1]};
            uint32_t *pd[] = {d[0], d[1]};
            for (uint64_t exec_mask : {0ULL, ~0ULL}) {
              EXPECT_EQ(instruction(cpu | unary_policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL,
                                    exec_mask, 0, pd, pa, &state),
                        GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
              EXPECT_EQ(instruction(cpu | unary_policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                        GOC_SEMANTICS_STRICT,
                                    exec_mask, 0, pd, pa),
                        GOC_ERROR_UNSUPPORTED_SEMANTICS);
              EXPECT_EQ(instruction(cpu | unary_policy(guest), exec_mask, GOC_DPP8, pd, pa),
                        GOC_ERROR_INVALID_FLAGS);
              EXPECT_EQ(instruction(cpu | unary_policy(guest), exec_mask, GOC_ALU_NEG_B, pd, pa),
                        GOC_ERROR_INVALID_FLAGS);
            }
            for (const auto &reg : d)
              for (auto word : reg)
                EXPECT_EQ(word, 0xdeadbeefU);
            for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
              ASSERT_EQ(std::fesetround(rounding), 0);
              EXPECT_EQ(instruction(cpu | unary_policy(guest), UINT64_MAX, 0, pd, pa, &state),
                        GOC_SUCCESS);
              EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
              for (unsigned denorm : {0U, 0x40U, 0x8000U, 0x8040U}) {
                unsigned control = (_mm_getcsr() & ~0x8040U) | denorm;
                _mm_setcsr(control | 0x20U);
                EXPECT_EQ(instruction(cpu | unary_policy(guest), UINT64_MAX, 0, pd, pa, &state),
                          GOC_SUCCESS);
                EXPECT_EQ(_mm_getcsr() & ~0x3fU, control & ~0x3fU);
                EXPECT_NE(_mm_getcsr() & 0x20U, 0U);
              }
#endif
            }
            EXPECT_EQ(state, 0x80000055U);
          }
}

namespace {

void check_numerical_corpus(uint32_t seed) {
  std::mt19937 random(seed);
  for (unsigned batch = 0; batch < 32; ++batch) {
    uint32_t inputs[2][64], outputs[2][64];
    for (unsigned lane = 0; lane < 64; ++lane) {
      uint64_t high = random();
      uint32_t low = random();
      uint64_t word = (high << 32) | low;
      if (batch % 3 == 1)
        word = 0x3ff0000000000000ULL + word % 8193 - 4096;
      else if (batch % 3 == 2)
        word &= 0x800fffffffffffffULL;
      inputs[0][lane] = uint32_t(word);
      inputs[1][lane] = uint32_t(word >> 32);
    }
    const uint32_t *a[] = {inputs[0], inputs[1]};
    uint32_t *d[] = {outputs[0], outputs[1]};
    for (unsigned op = 5; op < 8; ++op)
      for (unsigned variant = 0; variant < 32; ++variant) {
        uint32_t mode = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
                        ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
        for (bool wide : {false, true})
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            ASSERT_EQ(wide ? wave64[op](cpu, UINT64_MAX, mode, d, a, nullptr)
                           : functions[op](cpu, UINT32_MAX, mode, d, a, nullptr),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
              SCOPED_TRACE(::testing::Message()
                           << seed << '/' << batch << '/' << op << '/' << variant << '/' << wide
                           << '/' << cpu << '/' << lane);
              uint64_t input = inputs[0][lane] | (uint64_t(inputs[1][lane]) << 32);
              uint64_t output = outputs[0][lane] | (uint64_t(outputs[1][lane]) << 32);
              ASSERT_TRUE(goc_test::fuzzy(number(output), reference(op, input, mode),
                                          goc_test::unary_f64_tolerance));
            }
          }
      }
  }
}

} // namespace

TEST(Fp64Unary, DeterministicNumericalCalibration) { check_numerical_corpus(33457U); }

TEST(Fp64Unary, DeterministicNumericalHoldout) { check_numerical_corpus(81541U); }

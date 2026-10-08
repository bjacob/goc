// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stddef.h>
#include <stdint.h>
#include <utility>
#include <vector>

namespace {

using Fn = decltype(&goc_rdna4_v_ldexp_f32);
const Fn functions[] = {goc_rdna4_v_ldexp_f32, goc_rdna4_v_ldexp_f64};

uint32_t modifiers(int variant) {
  return (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
         ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
}

template <typename T, typename U> U reference(U input, int exponent, uint32_t mode) {
  const U sign = U(1) << (sizeof(U) * 8 - 1);
  if (mode & GOC_ALU_ABS_A)
    input &= ~sign;
  if (mode & GOC_ALU_NEG_A)
    input ^= sign;
  T value;
  std::memcpy(&value, &input, sizeof(value));
  // Long double has enough range for every finite FP32/FP64 boundary. Clamp
  // extreme exponents first; they have already forced overflow or underflow.
  value = T(std::ldexp(static_cast<long double>(value), std::clamp(exponent, -4096, 4096)));
  const T scale[] = {T(1), T(2), T(4), T(0.5)};
  if (mode & GOC_ALU_OMOD_HALF)
    value *= scale[(mode >> 6) & 3];
  if (mode & GOC_ALU_CLAMP)
    value = !(value > 0) ? T(0) : value > 1 ? T(1) : value;
  U result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

uint64_t expected(bool fp64, uint64_t input, int exponent, uint32_t mode) {
  return fp64 ? reference<double>(input, exponent, mode)
              : reference<float>(uint32_t(input), exponent, mode);
}

bool nan_bits(bool fp64, uint64_t bits) {
  return fp64 ? (bits & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000)
              : (bits & 0x7fffffff) > 0x7f800000;
}

} // namespace

TEST(Ldexp, AllModifiersMasksAndAliases) {
  const int aliases[][2] = {{3, 4}, {0, 1}, {2, 0}, {1, 0}, {0, 2}, {2, 1}, {3, 3}};
  for (int fp64 = 0; fp64 < 2; ++fp64) {
    const int width = fp64 ? 52 : 23, bias = fp64 ? 1023 : 127;
    const uint64_t sign = UINT64_C(1) << (fp64 ? 63 : 31);
    const uint64_t normal = UINT64_C(1) << width, inf = uint64_t(2 * bias + 1) << width;
    const uint64_t values[] = {0,
                               sign,
                               1,
                               sign | 1,
                               normal - 1,
                               sign | (normal - 1),
                               normal,
                               sign | normal,
                               normal + 1,
                               sign | (normal + 1),
                               inf - 1,
                               sign | (inf - 1),
                               inf,
                               sign | inf,
                               uint64_t(bias) << width,
                               sign | (uint64_t(bias) << width)};
    const int powers[] = {0,   1,   -1,   2,    -2,    23,        -23,       52,
                          -52, 126, -126, 1022, -1022, INT32_MIN, INT32_MAX, -4096};
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int variant = 0; variant < 32; ++variant)
        for (uint64_t mask : rdna4_exec_masks())
          for (const auto &alias : aliases) {
            SCOPED_TRACE(::testing::Message() << fp64 << "/" << cpu << "/" << variant << "/" << mask
                                              << "/" << alias[0] << "/" << alias[1]);
            uint32_t storage[5][34], want[5][34];
            for (auto &reg : storage)
              std::fill(reg, reg + 34, 0xdeadbeef);
            uint32_t *a[] = {storage[0] + 1, storage[1] + 1}, *b = storage[2] + 1;
            uint32_t *d[] = {storage[alias[0]] + 1, storage[alias[1]] + 1};
            uint64_t results[32];
            for (int lane = 0; lane < 32; ++lane) {
              uint64_t input = values[lane % 16];
              int power = powers[(lane + lane / 16) % 16];
              a[0][lane] = uint32_t(input);
              a[1][lane] = uint32_t(input >> 32);
              b[lane] = uint32_t(power);
              results[lane] = expected(fp64, input, power, modifiers(variant));
            }
            std::memcpy(want, storage, sizeof(want));
            for (int reg = 0; reg <= fp64; ++reg)
              for (int lane = 0; lane < 32; ++lane)
                if (mask >> lane & 1)
                  want[alias[reg]][lane + 1] = uint32_t(results[lane] >> (32 * reg));
            ASSERT_EQ(functions[fp64](cpu, mask, modifiers(variant), d, a, &b), GOC_SUCCESS);
            for (int reg = 0; reg < 5; ++reg)
              for (int lane = 0; lane < 34; ++lane)
                EXPECT_EQ(storage[reg][lane], want[reg][lane]);
          }
  }
}

TEST(Ldexp, EveryExponentBoundaryAndRandomValues) {
  for (int fp64 = 0; fp64 < 2; ++fp64) {
    int width = fp64 ? 52 : 23, bias = fp64 ? 1023 : 127, max_field = 2 * bias + 1;
    uint64_t sign = UINT64_C(1) << (fp64 ? 63 : 31), fraction = (UINT64_C(1) << width) - 1;
    std::vector<std::pair<uint64_t, int>> cases;
    for (int field = 0; field <= max_field; ++field)
      for (uint64_t tail : {UINT64_C(0), UINT64_C(1), fraction})
        for (int target : {-width - 1, -width, -1, 0, 1, max_field - 1, max_field}) {
          uint64_t input = (uint64_t(field) << width) | tail;
          cases.emplace_back(input, target - field);
          cases.emplace_back(input | sign, target - field);
        }
    for (int bit = 0; bit < width; ++bit)
      for (int power : {-1, 0, 1, width, bias, 2 * bias, INT32_MIN, INT32_MAX})
        cases.emplace_back(UINT64_C(1) << bit, power);
    std::mt19937_64 random(905);
    for (int i = 0; i < 2048; ++i)
      cases.emplace_back(random() & (fp64 ? UINT64_MAX : UINT32_MAX),
                         int(random() % (4 * bias)) - 2 * bias);
    // Exercise each output scale across the exponent boundaries. The mask/alias
    // matrix above covers every combination of source and output modifiers.
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int variant : {0, 4, 8, 12, 16, 31})
        for (size_t start = 0; start < cases.size(); start += 32) {
          SCOPED_TRACE(::testing::Message()
                       << fp64 << "/" << cpu << "/" << variant << "/" << start);
          uint32_t a[2][32], b[32], d[2][32];
          uint32_t *pa[] = {a[0], a[1]}, *pb = b, *pd[] = {d[0], d[1]};
          for (int lane = 0; lane < 32; ++lane) {
            const auto &test = cases[(start + lane) % cases.size()];
            a[0][lane] = uint32_t(test.first);
            a[1][lane] = uint32_t(test.first >> 32);
            b[lane] = uint32_t(test.second);
          }
          ASSERT_EQ(functions[fp64](cpu, UINT32_MAX, modifiers(variant), pd, pa, &pb), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane) {
            const auto &test = cases[(start + lane) % cases.size()];
            uint64_t want = expected(fp64, test.first, test.second, modifiers(variant));
            uint64_t got = d[0][lane] | (fp64 ? uint64_t(d[1][lane]) << 32 : 0);
            if (nan_bits(fp64, want)) {
              EXPECT_TRUE(nan_bits(fp64, got));
              EXPECT_NE(got & (UINT64_C(1) << (width - 1)), 0u);
            } else {
              EXPECT_EQ(got, want);
            }
          }
        }
  }
}

TEST(Ldexp, LiteralRoundingAndValidation) {
  struct Case {
    uint64_t input;
    int power;
    uint64_t output;
  };

  const Case cases[][8] = {{{1, -1, 0},
                            {3, -1, 2},
                            {5, -1, 2},
                            {0x80000001, -1, 0x80000000},
                            {0x007fffff, 1, 0x00fffffe},
                            {0x00ffffff, -1, 0x00800000},
                            {0x7f7fffff, 1, 0x7f800000},
                            {0x3f800000, -149, 1}},
                           {{1, -1, 0},
                            {3, -1, 2},
                            {5, -1, 2},
                            {0x8000000000000001, -1, 0x8000000000000000},
                            {0x000fffffffffffff, 1, 0x001ffffffffffffe},
                            {0x001fffffffffffff, -1, 0x0010000000000000},
                            {0x7fefffffffffffff, 1, 0x7ff0000000000000},
                            {0x3ff0000000000000, -1074, 1}}};
  for (int fp64 = 0; fp64 < 2; ++fp64)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[2][32], b[32], d[2][32];
      uint32_t *pa[] = {a[0], a[1]}, *pb = b, *pd[] = {d[0], d[1]};
      for (int lane = 0; lane < 32; ++lane) {
        const auto &test = cases[fp64][lane % 8];
        a[0][lane] = uint32_t(test.input);
        a[1][lane] = uint32_t(test.input >> 32);
        b[lane] = uint32_t(test.power);
      }
      ASSERT_EQ(functions[fp64](cpu, UINT32_MAX, 0, pd, pa, &pb), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(d[0][lane], uint32_t(cases[fp64][lane % 8].output));
        if (fp64) {
          EXPECT_EQ(d[1][lane], uint32_t(cases[fp64][lane % 8].output >> 32));
        }
      }
      for (auto &reg : d)
        std::fill(reg, reg + 32, 0xdeadbeef);
      for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
        for (uint32_t invalid :
             {GOC_ALU_NEG_B, GOC_ALU_ABS_B, GOC_ALU_NEG_C, GOC_ALU_HIGH_D, UINT32_C(1) << 31})
          EXPECT_EQ(functions[fp64](cpu, mask, invalid, pd, pa, &pb), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[fp64](cpu | (UINT64_C(1) << 63), mask, 0, pd, pa, &pb),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[fp64](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask,
                                  0, pd, pa, &pb),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (auto &reg : d)
        for (uint32_t value : reg)
          EXPECT_EQ(value, 0xdeadbeef);
      EXPECT_EQ(functions[fp64](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, pd, pa, &pb),
                GOC_SUCCESS);
    }
}

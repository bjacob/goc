// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_omod_reference.h"

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
  long double wide = std::ldexp(static_cast<long double>(value), std::clamp(exponent, -4096, 4096));
  if ((mode & GOC_ALU_OMOD_HALF) &&
      std::abs(wide) < std::ldexp(1.0L, sizeof(T) == sizeof(float) ? -126 : -1022))
    wide = 0;
  value = T(wide);
  if constexpr (sizeof(T) == sizeof(float))
    value = goc_test::omod_f32_reference(value, mode);
  else
    value = goc_test::omod_f64_reference(value, mode);
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
  return fp64 ? (bits & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL
              : (bits & 0x7fffffff) > 0x7f800000;
}

} // namespace

TEST(Ldexp, AllModifiersMasksAndAliases) {
  const int aliases[][2] = {{3, 4}, {0, 1}, {2, 0}, {1, 0}, {0, 2}, {2, 1}, {3, 3}};
  for (int fp64 = 0; fp64 < 2; ++fp64) {
    const int width = fp64 ? 52 : 23, bias = fp64 ? 1023 : 127;
    const uint64_t sign = 1ULL << (fp64 ? 63 : 31);
    const uint64_t normal = 1ULL << width, inf = uint64_t(2 * bias + 1) << width;
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
        for (uint32_t exec_mask : rdna4_exec_masks())
          for (const auto &alias : aliases) {
            SCOPED_TRACE(::testing::Message() << fp64 << "/" << cpu << "/" << variant << "/"
                                              << exec_mask << "/" << alias[0] << "/" << alias[1]);
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
                if (exec_mask >> lane & 1)
                  want[alias[reg]][lane + 1] = uint32_t(results[lane] >> (32 * reg));
            ASSERT_EQ(functions[fp64](cpu, exec_mask, modifiers(variant), d, a, &b, nullptr),
                      GOC_SUCCESS);
            for (int reg = 0; reg < 5; ++reg)
              for (int lane = 0; lane < 34; ++lane)
                EXPECT_EQ(storage[reg][lane], want[reg][lane]);
          }
  }
}

TEST(Ldexp, EveryExponentBoundaryAndRandomValues) {
  for (int fp64 = 0; fp64 < 2; ++fp64) {
    int width = fp64 ? 52 : 23, bias = fp64 ? 1023 : 127, max_field = 2 * bias + 1;
    uint64_t sign = 1ULL << (fp64 ? 63 : 31), fraction = (1ULL << width) - 1;
    std::vector<std::pair<uint64_t, int>> cases;
    for (int field = 0; field <= max_field; ++field)
      for (uint64_t tail : std::initializer_list<uint64_t>{0ULL, 1ULL, fraction})
        for (int target : {-width - 1, -width, -1, 0, 1, max_field - 1, max_field}) {
          uint64_t input = (uint64_t(field) << width) | tail;
          cases.emplace_back(input, target - field);
          cases.emplace_back(input | sign, target - field);
        }
    for (int bit = 0; bit < width; ++bit)
      for (int power : {-1, 0, 1, width, bias, 2 * bias, INT32_MIN, INT32_MAX})
        cases.emplace_back(1ULL << bit, power);
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
          ASSERT_EQ(functions[fp64](cpu, UINT32_MAX, modifiers(variant), pd, pa, &pb, nullptr),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane) {
            const auto &test = cases[(start + lane) % cases.size()];
            uint64_t want = expected(fp64, test.first, test.second, modifiers(variant));
            uint64_t got = d[0][lane] | (fp64 ? uint64_t(d[1][lane]) << 32 : 0);
            if (nan_bits(fp64, want)) {
              EXPECT_TRUE(nan_bits(fp64, got));
              EXPECT_NE(got & (1ULL << (width - 1)), 0u);
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
      ASSERT_EQ(functions[fp64](cpu, UINT32_MAX, 0, pd, pa, &pb, nullptr), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(d[0][lane], uint32_t(cases[fp64][lane % 8].output));
        if (fp64) {
          EXPECT_EQ(d[1][lane], uint32_t(cases[fp64][lane % 8].output >> 32));
        }
      }
      for (auto &reg : d)
        std::fill(reg, reg + 32, 0xdeadbeef);
      for (uint32_t exec_mask : {0U, UINT32_MAX}) {
        for (uint32_t invalid :
             {GOC_ALU_NEG_B, GOC_ALU_ABS_B, GOC_ALU_NEG_C, GOC_ALU_HIGH_D, 1U << 31})
          EXPECT_EQ(functions[fp64](cpu, exec_mask, invalid, pd, pa, &pb, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[fp64](cpu | (1ULL << 63), exec_mask, 0, pd, pa, &pb, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[fp64](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                  exec_mask, 0, pd, pa, &pb, nullptr),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (auto &reg : d)
        for (uint32_t value : reg)
          EXPECT_EQ(value, 0xdeadbeef);
      EXPECT_EQ(
          functions[fp64](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, pd, pa, &pb, nullptr),
          GOC_SUCCESS);
    }
}

TEST(Ldexp, DppHardwareCorpus) {
  const uint32_t values[] = {1,          0x807fffff, 0x00800000, 0x80000000,
                             0x7f800001, 0xff800000, 0x3f800000, 0xff7fffff};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t exec_mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (uint64_t descriptor : goc_test::dpp_modes)
        for (unsigned variant = 0; variant < 32; ++variant) {
          uint32_t a[32], b[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = values[lane % 8];
            b[lane] = lane % 9 - 4;
            d[lane] = 0xdead0000u + lane;
          }
          auto pa = a, pb = b, pd = d;
          ASSERT_EQ(
              functions[0](cpu, exec_mask, descriptor | modifiers(variant), &pd, &pa, &pb, nullptr),
              GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            uint32_t want = 0xdead0000u + lane;
            bool active = goc_test::dpp_source(descriptor, exec_mask, lane, source);
            if (active)
              want = reference<float>(source < 0 ? 0u : a[source], int(lane % 9) - 4,
                                      modifiers(variant));
            if (active && nan_bits(false, want))
              EXPECT_TRUE(nan_bits(false, d[lane]));
            else
              EXPECT_EQ(d[lane], want);
            uint32_t word = d[lane];
            if (nan_bits(false, word))
              word = 0x7fc00000;
            hash = goc_test::capture_hash_word(hash, word);
          }
        }
    EXPECT_EQ(hash, 0x78ec29fe61dad845ULL);
  }
}

TEST(Ldexp, DppModifiersMasksAliasesAndRandomWords) {
  const uint32_t special[] = {0,          0x80000000, 1,          0x807fffff, 0x00800000,
                              0x80800001, 0x3f800000, 0xff7fffff, 0x7f800000, 0xff800001};
  const int exponents[] = {INT32_MIN, -149, -126, -1, 0, 1, 127, INT32_MAX};
  std::mt19937 random(628);
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t descriptor : goc_test::dpp_modes)
      for (unsigned variant = 0; variant < 32; ++variant)
        for (uint32_t exec_mask : rdna4_exec_masks())
          for (unsigned layout = 0; layout < 5; ++layout) {
            uint32_t words[3][34], before[3][34];
            for (auto &reg : words)
              for (auto &word : reg)
                word = random();
            for (unsigned lane = 1; lane <= 32; ++lane) {
              if (lane & 1)
                words[0][lane] = special[(lane / 2) % 10];
              words[1][lane] = uint32_t(exponents[lane % 8]);
            }
            std::memcpy(before, words, sizeof(words));
            unsigned dest = layout % 3 == 0 ? 2 : layout % 3 - 1;
            unsigned breg = layout >= 3 ? 0 : 1;
            auto a = words[0] + 1, b = words[breg] + 1, d = words[dest] + 1;
            ASSERT_EQ(
                functions[0](cpu, exec_mask, descriptor | modifiers(variant), &d, &a, &b, nullptr),
                GOC_SUCCESS);
            for (unsigned reg = 0; reg < 3; ++reg)
              for (unsigned lane = 0; lane < 34; ++lane) {
                int source;
                uint32_t want = before[reg][lane];
                bool active = reg == dest && lane >= 1 && lane <= 32 &&
                              goc_test::dpp_source(descriptor, exec_mask, lane - 1, source);
                if (active) {
                  int32_t exponent;
                  std::memcpy(&exponent, &before[breg][lane], sizeof(exponent));
                  want = reference<float>(source < 0 ? 0u : before[0][source + 1], exponent,
                                          modifiers(variant));
                }
                if (active && nan_bits(false, want))
                  EXPECT_TRUE(nan_bits(false, words[reg][lane]));
                else
                  EXPECT_EQ(words[reg][lane], want);
              }
          }
}

TEST(Ldexp, DppValidation) {
  for (uint64_t descriptor : goc_test::dpp_modes) {
    EXPECT_EQ(functions[0](0, 0, descriptor, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(
        functions[0](0, UINT32_MAX, descriptor | GOC_ALU_NEG_B, nullptr, nullptr, nullptr, nullptr),
        GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        functions[0](0, UINT32_MAX, descriptor | (1ULL << 36), nullptr, nullptr, nullptr, nullptr),
        GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(functions[0](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                           descriptor, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
  }
}

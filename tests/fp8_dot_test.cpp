// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "exec_masks.h"
#include "goc/goc.h"
#include "internal.h"
#include "loose_tolerances.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdint.h>
#include <vector>

namespace {

using Fn = decltype(&goc_v_mad_u32_u24);
const Fn functions[] = {goc_v_dot4_f32_fp8_fp8, goc_v_dot4_f32_fp8_bf8, goc_v_dot4_f32_bf8_fp8,
                        goc_v_dot4_f32_bf8_bf8};

// Independent mathematical decode; no implementation conversion helpers.
double decode(uint8_t value, bool bf8) {
  const int radix = bf8 ? 4 : 8, bias = bf8 ? 15 : 7;
  const int magnitude = value % 128, exponent = magnitude / radix, fraction = magnitude % radix;
  double result;
  if (bf8 && exponent == 31)
    result = fraction ? std::numeric_limits<double>::quiet_NaN()
                      : std::numeric_limits<double>::infinity();
  else if (!bf8 && magnitude == 127)
    result = std::numeric_limits<double>::quiet_NaN();
  else
    result = std::ldexp(exponent ? 1.0 + double(fraction) / radix : double(fraction) / radix,
                        (exponent ? exponent : 1) - bias);
  return value >= 128 ? -result : result;
}

void compare(uint32_t word, double expected) {
  float actual = goc::as_float(word), rounded = float(expected);
  if (std::isnan(rounded)) {
    EXPECT_TRUE(std::isnan(actual));
  } else {
    EXPECT_TRUE(goc_test::fuzzy(actual, rounded, goc_test::fp8_dot_repeated_tolerance));
  }
}

const decltype(&goc_v_dot4_f32_fp8_fp8_wave64) wave64_functions[] = {
    goc_v_dot4_f32_fp8_fp8_wave64, goc_v_dot4_f32_fp8_bf8_wave64, goc_v_dot4_f32_bf8_fp8_wave64,
    goc_v_dot4_f32_bf8_bf8_wave64};

} // namespace

TEST(Fp8Dot, ExhaustiveBytePairsAllFormatsAndModifiers) {
  for (int op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t modifiers : {0U, GOC_DOT_NEG_C, GOC_DOT_ABS_C, GOC_DOT_ABS_C | GOC_DOT_NEG_C})
        for (unsigned base = 0; base < 65536; base += 32) {
          SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << modifiers << "/" << base);
          uint32_t a[32], b[32], c[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = ((base + lane) / 256) * 0x01010101u;
            b[lane] = ((base + lane) % 256) * 0x01010101u;
            c[lane] = goc::as_bits(-1.25f);
          }
          auto pa = a, pb = b, pc = c, pd = d;
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, modifiers, &pd, &pa, &pb, &pc), GOC_SUCCESS);
          double acc = modifiers & GOC_DOT_ABS_C ? 1.25 : -1.25;
          if (modifiers & GOC_DOT_NEG_C)
            acc = -acc;
          for (unsigned lane = 0; lane < 32; ++lane)
            compare(d[lane], 4 * decode(uint8_t((base + lane) / 256), op >= 2) *
                                     decode(uint8_t((base + lane) % 256), op & 1) +
                                 acc);
        }
}

TEST(Fp8Dot, MixedBytesMasksAndAliases) {
  std::mt19937 rng(54321);
  uint32_t inputs[3][32];
  for (auto &reg : inputs)
    for (auto &word : reg)
      word = rng();
  // Every finite integer factor in [-2,2] is exactly encoded in both formats.
  const uint8_t fp8[] = {0xc0, 0xb8, 0, 0x38, 0x40};
  const uint8_t bf8[] = {0xc0, 0xbc, 0, 0x3c, 0x40};
  for (int op = 0; op < 4; ++op) {
    int factors[2][32][4];
    for (int lane = 0; lane < 32; ++lane) {
      for (int operand = 0; operand < 2; ++operand) {
        inputs[operand][lane] = 0;
        const auto *codes = (operand ? op & 1 : op >= 2) ? bf8 : fp8;
        for (int byte = 0; byte < 4; ++byte) {
          int index = rng() % 5;
          factors[operand][lane][byte] = index - 2;
          inputs[operand][lane] |= uint32_t(codes[index]) << (8 * byte);
        }
      }
      inputs[2][lane] = goc::as_bits(float(lane - 16));
    }
    for (uint32_t mode : {0U, GOC_DOT_NEG_C, GOC_DOT_ABS_C, GOC_DOT_NEG_C | GOC_DOT_ABS_C})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t exec_mask : exec_masks())
          for (int alias = 0; alias < 4; ++alias) {
            uint32_t storage[4][34], before[32];
            uint32_t *v[4];
            for (int reg = 0; reg < 4; ++reg) {
              std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
              if (reg < 3)
                std::copy(inputs[reg], inputs[reg] + 32, v[reg]);
            }
            std::copy(v[alias], v[alias] + 32, before);
            ASSERT_EQ(functions[op](cpu, exec_mask, mode, &v[alias], &v[0], &v[1], &v[2]),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              int expected = lane - 16;
              if (mode & GOC_DOT_ABS_C)
                expected = std::abs(expected);
              if (mode & GOC_DOT_NEG_C)
                expected = -expected;
              for (int byte = 0; byte < 4; ++byte)
                expected += factors[0][lane][byte] * factors[1][lane][byte];
              if ((exec_mask >> lane) & 1) {
                // Small integer products and sums are exactly representable.
                EXPECT_EQ(goc::as_float(v[alias][lane]), float(expected));
              } else {
                EXPECT_EQ(v[alias][lane], before[lane]);
              }
            }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
  }
}

TEST(Fp8Dot, ValidationAndLooseFallback) {
  for (auto fn : functions) {
    uint32_t a[32] = {}, d[32];
    std::fill(d, d + 32, 0xdeadbeef);
    auto pa = a, pd = d;
    for (uint32_t exec_mask : {0U, UINT32_MAX}) {
      for (uint32_t invalid : {GOC_DOT_NEG_LO_A, GOC_DOT_CLAMP, GOC_DOT_LO_A_HIGH, 1U << 31})
        EXPECT_EQ(fn(0, exec_mask, invalid, &pd, &pa, &pa, &pa), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, 0, &pd, &pa,
                   &pa, &pa),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : d)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &pd, &pa, &pa, &pa), GOC_SUCCESS);
    for (auto word : d)
      EXPECT_EQ(word, 0u);
  }
}

TEST(Fp8Dot, Wave64ExhaustiveBytePairsAllFormatsAndModifiers) {
  for (int op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t modifiers : {0U, GOC_DOT_NEG_C, GOC_DOT_ABS_C, GOC_DOT_ABS_C | GOC_DOT_NEG_C})
        for (unsigned base = 0; base < 65536; base += 64) {
          SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << modifiers << "/" << base);
          uint32_t a[64], b[64], c[64], d[64];
          for (unsigned lane = 0; lane < 64; ++lane) {
            a[lane] = ((base + lane) / 256) * 0x01010101u;
            b[lane] = ((base + lane) % 256) * 0x01010101u;
            c[lane] = goc::as_bits(-1.25f);
          }
          auto pa = a, pb = b, pc = c, pd = d;
          ASSERT_EQ(wave64_functions[op](cpu, UINT64_MAX, modifiers, &pd, &pa, &pb, &pc),
                    GOC_SUCCESS);
          double acc = modifiers & GOC_DOT_ABS_C ? 1.25 : -1.25;
          if (modifiers & GOC_DOT_NEG_C)
            acc = -acc;
          for (unsigned lane = 0; lane < 64; ++lane)
            compare(d[lane], 4 * decode(uint8_t((base + lane) / 256), op >= 2) *
                                     decode(uint8_t((base + lane) % 256), op & 1) +
                                 acc);
        }
}

TEST(Fp8Dot, Wave64MixedBytesMasksAndAliases) {
  std::mt19937 rng(54321);
  uint32_t inputs[3][64];
  for (auto &reg : inputs)
    for (auto &word : reg)
      word = rng();
  // Every finite integer factor in [-2,2] is exactly encoded in both formats.
  const uint8_t fp8[] = {0xc0, 0xb8, 0, 0x38, 0x40};
  const uint8_t bf8[] = {0xc0, 0xbc, 0, 0x3c, 0x40};
  for (int op = 0; op < 4; ++op) {
    int factors[2][64][4];
    for (int lane = 0; lane < 64; ++lane) {
      for (int operand = 0; operand < 2; ++operand) {
        inputs[operand][lane] = 0;
        const auto *codes = (operand ? op & 1 : op >= 2) ? bf8 : fp8;
        for (int byte = 0; byte < 4; ++byte) {
          int index = rng() % 5;
          factors[operand][lane][byte] = index - 2;
          inputs[operand][lane] |= uint32_t(codes[index]) << (8 * byte);
        }
      }
      inputs[2][lane] = goc::as_bits(float(lane - 16));
    }
    for (uint32_t mode : {0U, GOC_DOT_NEG_C, GOC_DOT_ABS_C, GOC_DOT_NEG_C | GOC_DOT_ABS_C})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t exec_mask : {uint64_t(UINT64_MAX), uint64_t(0xffffffff00000000ULL),
                                   uint64_t(0x87654321fedcba98ULL)})
          for (int alias = 0; alias < 4; ++alias) {
            uint32_t storage[4][66], before[64];
            uint32_t *v[4];
            for (int reg = 0; reg < 4; ++reg) {
              std::fill(storage[reg], storage[reg] + 66, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
              if (reg < 3)
                std::copy(inputs[reg], inputs[reg] + 64, v[reg]);
            }
            std::copy(v[alias], v[alias] + 64, before);
            ASSERT_EQ(wave64_functions[op](cpu, exec_mask, mode, &v[alias], &v[0], &v[1], &v[2]),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 64; ++lane) {
              int expected = lane - 16;
              if (mode & GOC_DOT_ABS_C)
                expected = std::abs(expected);
              if (mode & GOC_DOT_NEG_C)
                expected = -expected;
              for (int byte = 0; byte < 4; ++byte)
                expected += factors[0][lane][byte] * factors[1][lane][byte];
              if ((exec_mask >> lane) & 1) {
                // Small integer products and sums are exactly representable.
                EXPECT_EQ(goc::as_float(v[alias][lane]), float(expected));
              } else {
                EXPECT_EQ(v[alias][lane], before[lane]);
              }
            }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[65], 0xdeadbeef);
            }
          }
  }
}

TEST(Fp8Dot, Wave64ValidationAndLooseFallback) {
  for (auto fn : wave64_functions) {
    uint32_t a[64] = {}, d[64];
    std::fill(d, d + 64, 0xdeadbeef);
    auto pa = a, pd = d;
    for (uint64_t exec_mask : {uint64_t(0), uint64_t(UINT64_MAX)}) {
      for (uint32_t invalid : {GOC_DOT_NEG_LO_A, GOC_DOT_CLAMP, GOC_DOT_LO_A_HIGH, 1U << 31})
        EXPECT_EQ(fn(0, exec_mask, invalid, &pd, &pa, &pa, &pa), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, 0, &pd, &pa,
                   &pa, &pa),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : d)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, 0, &pd, &pa, &pa, &pa), GOC_SUCCESS);
    for (auto word : d)
      EXPECT_EQ(word, 0u);
  }
}

namespace {

template <unsigned Lanes> void dpp_routes_masks_and_aliases() {
  // Integer-valued factors make the arithmetic oracle exact; lane-dependent
  // A/B/C distinguish routing A from accidentally routing the other operands.
  const uint8_t codes[2][5] = {{0xc0, 0xb8, 0, 0x38, 0x40}, {0xc0, 0xbc, 0, 0x3c, 0x40}};
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = GOC_CPU_BASELINE; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto route : goc_test::dpp_modes)
        for (unsigned rows = 0; rows < (route & GOC_DPP16 ? 16U : 1U); ++rows)
          for (uint64_t modifiers : {uint64_t(0), uint64_t(GOC_DOT_NEG_C), uint64_t(GOC_DOT_ABS_C),
                                     uint64_t(GOC_DOT_NEG_C | GOC_DOT_ABS_C)})
            for (uint64_t active : {uint64_t(UINT64_MAX), uint64_t(0x81234567fedcba99ULL)})
              for (unsigned dst = 0; dst < 4; ++dst) {
                uint64_t descriptor = route;
                if (route & GOC_DPP16)
                  descriptor = (route & ~GOC_DPP_ROW_MASK) | (uint64_t(rows) << GOC_DPP_ROW_SHIFT);
                const uint64_t exec_mask = Lanes == 32 ? uint32_t(active) : active;
                SCOPED_TRACE(::testing::Message()
                             << Lanes << "/" << op << "/" << cpu << "/" << descriptor << "/"
                             << modifiers << "/" << exec_mask << "/" << dst);
                uint32_t initial[4][Lanes + 2], words[4][Lanes + 2], expected_words[4][Lanes + 2];
                for (auto &reg : initial)
                  std::fill(reg, reg + Lanes + 2, 0xdeadbeefU);
                for (unsigned lane = 0; lane < Lanes; ++lane) {
                  initial[0][lane + 1] = initial[1][lane + 1] = 0;
                  for (unsigned byte = 0; byte < 4; ++byte) {
                    initial[0][lane + 1] |= uint32_t(codes[op / 2][(lane + byte) % 5])
                                            << (8 * byte);
                    initial[1][lane + 1] |= uint32_t(codes[op % 2][(3 * lane + byte + 1) % 5])
                                            << (8 * byte);
                  }
                  initial[2][lane + 1] = goc::as_bits(float(int(lane) - 37));
                }
                std::memcpy(words, initial, sizeof(words));
                std::memcpy(expected_words, initial, sizeof(expected_words));
                const uint32_t *a = words[0] + 1, *b = words[1] + 1, *c = words[2] + 1;
                uint32_t *d = words[dst] + 1;
                if constexpr (Lanes == 32)
                  ASSERT_EQ(functions[op](cpu, uint32_t(exec_mask), modifiers | descriptor, &d, &a,
                                          &b, &c),
                            GOC_SUCCESS);
                else
                  ASSERT_EQ(
                      wave64_functions[op](cpu, exec_mask, modifiers | descriptor, &d, &a, &b, &c),
                      GOC_SUCCESS);
                for (unsigned lane = 0; lane < Lanes; ++lane) {
                  int source;
                  if (!goc_test::dpp_source(descriptor, exec_mask, lane, source))
                    continue;
                  const uint32_t av = source < 0 ? 0 : initial[0][source + 1];
                  double expected = int(lane) - 37;
                  if (modifiers & GOC_DOT_ABS_C)
                    expected = std::abs(expected);
                  if (modifiers & GOC_DOT_NEG_C)
                    expected = -expected;
                  for (unsigned byte = 0; byte < 4; ++byte)
                    expected += decode(uint8_t(av >> (8 * byte)), op / 2) *
                                decode(uint8_t(initial[1][lane + 1] >> (8 * byte)), op % 2);
                  expected_words[dst][lane + 1] = goc::as_bits(float(expected));
                }
                EXPECT_EQ(std::memcmp(words, expected_words, sizeof(words)), 0);
              }
}

} // namespace

TEST(Fp8Dot, DppRoutesMasksAndAliases) { dpp_routes_masks_and_aliases<32>(); }

TEST(Fp8Dot, Wave64DppRoutesMasksAndAliases) { dpp_routes_masks_and_aliases<64>(); }

TEST(Fp8Dot, DppValidationAndLooseFallback) {
  for (unsigned op = 0; op < 4; ++op)
    for (auto mode : goc_test::dpp_modes) {
      uint32_t a[64] = {}, d[64];
      const uint32_t *pa = a;
      uint32_t *pd = d;
      std::fill(d, d + 64, 0xdeadbeefU);
      for (uint64_t exec_mask : {uint64_t(0), uint64_t(UINT64_MAX)}) {
        EXPECT_EQ(functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                uint32_t(exec_mask), mode, &pd, &pa, &pa, &pa),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(wave64_functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                       exec_mask, mode, &pd, &pa, &pa, &pa),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(
            functions[op](0, uint32_t(exec_mask), mode | GOC_DOT_NEG_LO_A, &pd, &pa, &pa, &pa),
            GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(wave64_functions[op](0, exec_mask, mode | GOC_DOT_NEG_LO_A, &pd, &pa, &pa, &pa),
                  GOC_ERROR_INVALID_FLAGS);
      }
      for (auto word : d)
        EXPECT_EQ(word, 0xdeadbeefU);
      EXPECT_EQ(functions[op](0, 0, mode, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(wave64_functions[op](0, 0, mode, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      ASSERT_EQ(functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, mode, &pd, &pa, &pa, &pa),
                GOC_SUCCESS);
      ASSERT_EQ(
          wave64_functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, mode, &pd, &pa, &pa, &pa),
          GOC_SUCCESS);
      for (unsigned lane = 0; lane < 64; ++lane) {
        int source;
        const bool enabled = goc_test::dpp_source(mode, UINT64_MAX, lane, source);
        EXPECT_EQ(d[lane], enabled ? 0U : 0xdeadbeefU);
      }
    }
}

namespace {

// Mathematical binary64 accumulation, independent of the implementation's
// binary32 FMA chain. Bound factors by 4 to give the fixed absolute tolerance a
// meaningful scale near cancellation; exhaustive byte-pair tests cover the full
// encoding/exponent range separately.
void check_bounded_numerical_corpus(uint32_t seed) {
  std::vector<uint8_t> codes[2];
  for (unsigned format = 0; format < 2; ++format)
    for (unsigned code = 0; code < 256; ++code)
      if (std::isfinite(decode(code, format)) && std::abs(decode(code, format)) <= 4)
        codes[format].push_back(uint8_t(code));
  std::mt19937 random(seed);
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned batch = 0; batch < 64; ++batch) {
      uint32_t words[4][64] = {};
      double products[64] = {};
      for (unsigned lane = 0; lane < 64; ++lane) {
        for (unsigned byte = 0; byte < 4; ++byte) {
          uint8_t a = codes[op / 2][random() % codes[op / 2].size()];
          uint8_t b = codes[op % 2][random() % codes[op % 2].size()];
          words[0][lane] |= uint32_t(a) << (8 * byte);
          words[1][lane] |= uint32_t(b) << (8 * byte);
          products[lane] += decode(a, op / 2) * decode(b, op % 2);
        }
        float acc = float(int(random() % 2049) - 1024) / 256;
        if (batch & 1)
          acc = -float(products[lane]);
        words[2][lane] = goc::as_bits(acc);
      }
      const uint32_t *a = words[0], *b = words[1], *c = words[2];
      uint32_t *d = words[3];
      for (unsigned variant = 0; variant < 4; ++variant) {
        uint32_t mode = (variant & 1 ? GOC_DOT_NEG_C : 0) | (variant & 2 ? GOC_DOT_ABS_C : 0);
        for (bool wide : {false, true})
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            ASSERT_EQ(wide ? wave64_functions[op](cpu, UINT64_MAX, mode, &d, &a, &b, &c)
                           : functions[op](cpu, UINT32_MAX, mode, &d, &a, &b, &c),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
              SCOPED_TRACE(::testing::Message()
                           << seed << '/' << op << '/' << batch << '/' << variant << '/' << wide
                           << '/' << cpu << '/' << lane);
              double expected = goc::as_float(c[lane]);
              if (mode & GOC_DOT_ABS_C)
                expected = std::abs(expected);
              if (mode & GOC_DOT_NEG_C)
                expected = -expected;
              expected += products[lane];
              ASSERT_TRUE(goc_test::fuzzy(goc::as_float(d[lane]), float(expected),
                                          goc_test::fp8_dot_bounded_tolerance));
            }
          }
      }
    }
}

} // namespace

TEST(Fp8Dot, DeterministicNumericalCalibration) { check_bounded_numerical_corpus(39217U); }

TEST(Fp8Dot, DeterministicNumericalHoldout) { check_bounded_numerical_corpus(71993U); }

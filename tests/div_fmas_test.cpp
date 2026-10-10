// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "div_fmas_exceptions_hardware.h"
#include "div_fmas_hardware.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_div_fmas_f32_rdna4);
const Fn functions[] = {goc_v_div_fmas_f32_rdna4, goc_v_div_fmas_f64_rdna4};
const decltype(&goc_v_div_fmas_f32_rdna4_wave64) wave64_functions[] = {
    goc_v_div_fmas_f32_rdna4_wave64, goc_v_div_fmas_f64_rdna4_wave64};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

} // namespace

TEST(DivFmas, HardwareCartesianAndRandomCorpora) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned corpus = 0; corpus < 2; ++corpus)
        for (unsigned col = 0; col < 16; ++col) {
          uint32_t state = 0x9174ab23;
          uint64_t digest = goc_test::capture_hash_seed;
          for (unsigned start = 0; start < 4096; start += 32) {
            uint32_t data[8][32] = {};
            const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                           *c[] = {data[4], data[5]};
            uint32_t *d[] = {data[6], data[7]};
            for (unsigned lane = 0; lane < 32; ++lane) {
              unsigned index = start + lane,
                       indices[] = {index / 256, (index / 16) % 16, index % 16};
              uint64_t random_values[3];
              if (corpus)
                goc_test::fmas_capture_random(op, index, state, random_values);
              for (unsigned operand = 0; operand < 3; ++operand) {
                uint64_t value = corpus ? random_values[operand]
                                        : goc_test::fmas_capture_values[op][indices[operand]];
                data[operand * 2][lane] = uint32_t(value);
                data[operand * 2 + 1][lane] = uint32_t(value >> 32);
              }
            }
            ASSERT_EQ(functions[op](cpu | exact, UINT32_MAX, goc_test::fmas_capture_modes[col % 8],
                                    d, a, b, c, col / 8 ? UINT32_MAX : 0, nullptr),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint64_t value = data[6][lane];
              if (op)
                value |= uint64_t(data[7][lane]) << 32;
              digest = goc_test::capture_hash_bytes(digest, value, (op ? 8u : 4u));
            }
          }
          ASSERT_EQ(digest, (corpus ? goc_test::fmas_random_digests[op][col]
                                    : goc_test::fmas_capture_digests[op][col]))
              << op << "/" << cpu << "/" << col;
        }
}

TEST(DivFmas, AllModifiersRandomBitsAndConditionMasks) {
  std::mt19937 random(819542);
  for (unsigned op = 0; op < 2; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode)
      for (unsigned sample = 0; sample < 8; ++sample) {
        uint32_t data[8][32], expected[2][32], condition = random();
        for (auto &reg : data)
          for (auto &word : reg)
            word = random();
        // Include special values and exact cancellation alongside unrestricted bits.
        for (unsigned lane = 0; lane < 16; ++lane)
          for (unsigned operand = 0; operand < 3; ++operand) {
            uint64_t value =
                goc_test::fmas_capture_values[op][(lane * (operand * 2 + 1) + sample) % 16];
            data[operand * 2][lane] = uint32_t(value);
            data[operand * 2 + 1][lane] = uint32_t(value >> 32);
          }
        const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                       *c[] = {data[4], data[5]};
        uint32_t *d[] = {data[6], data[7]}, *gold[] = {expected[0], expected[1]};
        ASSERT_EQ(functions[op](exact, UINT32_MAX, mode, gold, a, b, c, condition, nullptr),
                  GOC_SUCCESS);
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          ASSERT_EQ(functions[op](cpu | exact | GOC_FP16_OVFL, UINT32_MAX, mode, d, a, b, c,
                                  condition, nullptr),
                    GOC_SUCCESS);
          for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(d[reg][lane], expected[reg][lane])
                  << op << "/" << cpu << "/" << mode << "/" << lane;
        }
      }
}

TEST(DivFmas, MasksAndCrossRegisterAliases) {
  std::mt19937 random(975283);
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant)
        for (uint32_t exec_mask : exec_masks())
          for (unsigned first = 0; first < 8; first += (op ? 1 : 2))
            for (unsigned second = 0; second < (op ? 8u : 1u); ++second) {
              uint32_t data[8][34], expected[8][34], result[2][32], condition = random();
              for (auto &reg : data)
                for (auto &word : reg)
                  word = random();
              std::memcpy(expected, data, sizeof(data));
              uint32_t mode = goc_test::fmas_capture_modes[variant];
              const uint32_t *a[] = {data[0] + 1, data[1] + 1}, *b[] = {data[2] + 1, data[3] + 1},
                             *c[] = {data[4] + 1, data[5] + 1};
              uint32_t *d[] = {data[first] + 1, data[second] + 1}, *gold[] = {result[0], result[1]};
              ASSERT_EQ(functions[op](exact, UINT32_MAX, mode, gold, a, b, c, condition, nullptr),
                        GOC_SUCCESS);
              unsigned dest[] = {first, second};
              for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
                for (unsigned lane = 0; lane < 32; ++lane)
                  if ((exec_mask >> lane) & 1)
                    expected[dest[reg]][lane + 1] = result[reg][lane];
              ASSERT_EQ(functions[op](cpu | (variant & 1 ? exact : 0), exec_mask, mode, d, a, b, c,
                                      condition, nullptr),
                        GOC_SUCCESS);
              ASSERT_EQ(std::memcmp(data, expected, sizeof(data)), 0)
                  << op << "/" << cpu << "/" << variant << "/" << first << "/" << second;
            }
}

TEST(DivFmas, SharedSourcesAndHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 2; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode) {
      uint32_t data[4][32], expected[2][32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        uint64_t value = goc_test::fmas_capture_values[op][lane % 16];
        data[0][lane] = uint32_t(value);
        data[1][lane] = uint32_t(value >> 32);
      }
      const uint32_t *a[] = {data[0], data[1]};
      uint32_t *d[] = {data[2], data[3]}, *gold[] = {expected[0], expected[1]};
      uint32_t expected_flags = 0;
      ASSERT_EQ(functions[op](exact, UINT32_MAX, mode, gold, a, a, a, 0xa5a5a5a5, &expected_flags),
                GOC_SUCCESS);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
          EXPECT_EQ(std::fesetround(rounding), 0);
          EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          if (mode & 1) {
            EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
          }
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          uint32_t actual_flags = 0;
          EXPECT_EQ(
              functions[op](cpu | exact, UINT32_MAX, mode, d, a, a, a, 0xa5a5a5a5, &actual_flags),
              GOC_SUCCESS);
          EXPECT_EQ(actual_flags, expected_flags);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
          for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(d[reg][lane], expected[reg][lane]);
        }
    }
}

TEST(DivFmas, ValidationAndZeroExec) {
  for (auto fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      uint32_t *d[] = {output, output};
      for (unsigned bit = 9; bit < 32; ++bit) {
        EXPECT_EQ(fn(cpu, UINT32_MAX, 1u << bit, d, nullptr, nullptr, nullptr, UINT32_MAX, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(cpu, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr, UINT32_MAX, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
      EXPECT_EQ(fn(cpu | (1ull << 63), UINT32_MAX, 0, d, nullptr, nullptr, nullptr, 0, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, d,
                   nullptr, nullptr, nullptr, 0, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(fn(cpu | exact, 0U, 0, nullptr, nullptr, nullptr, nullptr, UINT32_MAX, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, nullptr, nullptr,
                   nullptr, 0, nullptr),
                GOC_SUCCESS);
    }
}

TEST(DivFmas, FusedRoundingAndOutputModifierBoundaries) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned col = 0; col < 16; ++col) {
        uint32_t data[8][32];
        const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                       *c[] = {data[4], data[5]};
        uint32_t *d[] = {data[6], data[7]};
        for (unsigned lane = 0; lane < 32; ++lane)
          for (unsigned operand = 0; operand < 3; ++operand) {
            uint64_t value = goc_test::fmas_boundary[op][lane % 8][operand];
            data[operand * 2][lane] = uint32_t(value);
            data[operand * 2 + 1][lane] = uint32_t(value >> 32);
          }
        ASSERT_EQ(functions[op](cpu | exact, UINT32_MAX, goc_test::fmas_capture_modes[col % 8], d,
                                a, b, c, col / 8 ? UINT32_MAX : 0, nullptr),
                  GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint64_t result = data[6][lane];
          if (op)
            result |= uint64_t(data[7][lane]) << 32;
          EXPECT_EQ(result, goc_test::fmas_boundary[op][lane % 8][col + 3])
              << op << "/" << cpu << "/" << col << "/" << lane;
        }
      }
}

TEST(DivFmas, ExceptionHardwareCorpus) {
  for (unsigned type = 0; type < 2; ++type) {
    unsigned fraction = type ? 52 : 23, bias = type ? 1023 : 127;
    uint64_t sign = 1ULL << (type ? 63 : 31), infinity = uint64_t(2 * bias + 1) << fraction,
             one = uint64_t(bias) << fraction;
    uint64_t values[] = {0,
                         sign,
                         1,
                         sign | 1,
                         (1ULL << fraction) - 1,
                         1ULL << fraction,
                         one,
                         sign | one,
                         one + (1ULL << fraction),
                         infinity - 1,
                         infinity - 1 - (1ULL << fraction),
                         infinity,
                         sign | infinity,
                         infinity | 1,
                         sign | infinity | (1ULL << (fraction - 1)) | 3,
                         2ULL << fraction};
    for (unsigned condition = 0; condition < 2; ++condition)
      for (unsigned variant = 0; variant < 16; ++variant) {
        uint32_t mode =
            ((variant & 3) << 6) | (variant & 4 ? GOC_ALU_CLAMP : 0) |
            (variant & 8 ? GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_C : 0);
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned i = 0; i < 8192; ++i) {
          uint32_t words[4][2][32] = {};
          uint64_t state = uint64_t(i) * 0x9e3779b97f4a7c15ULL;
          for (unsigned operand = 0; operand < 3; ++operand) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            uint64_t value =
                i < 4096 ? values[(i >> (8 - operand * 4)) & 15] : state * 0x2545f4914f6cdd1dULL;
            words[operand][0][0] = uint32_t(value);
            words[operand][1][0] = uint32_t(value >> 32);
          }
          const uint32_t *a[] = {words[0][0], words[0][1]}, *b[] = {words[1][0], words[1][1]},
                         *c[] = {words[2][0], words[2][1]};
          uint32_t *d[] = {words[3][0], words[3][1]};
          uint32_t exceptions = 0x80000000;
          ASSERT_EQ(functions[type](exact | (i % (goc_init_cpu_flags() + 1)), 1, mode, d, a, b, c,
                                    condition, &exceptions),
                    GOC_SUCCESS);
          ASSERT_TRUE(exceptions & 0x80000000);
          hash = goc_test::capture_hash_word(hash, exceptions & 127);
        }
        EXPECT_EQ(hash, goc_test::div_fmas_exception_hashes[type][condition][variant])
            << type << "/" << condition << "/" << variant;
      }
  }
}

TEST(DivFmas, ExceptionMasksConditionAndAliases) {
  for (unsigned type = 0; type < 2; ++type)
    for (uint32_t exec_mask : exec_masks())
      for (uint32_t condition : {0U, UINT32_MAX, 0xaaaaaaaaU, 0x55555555U})
        for (unsigned dest = 0; dest < 4; ++dest) {
          uint32_t initial[8][32];
          for (unsigned operand = 0; operand < 4; ++operand)
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint64_t value = goc_test::fmas_capture_values[type][(lane + 5 * operand) % 16];
              initial[operand * 2][lane] = uint32_t(value);
              initial[operand * 2 + 1][lane] = uint32_t(value >> 32);
            }
          uint32_t expected_flags = 0x80000000;
          for (unsigned lane = 0; lane < 32; ++lane) {
            if (!((exec_mask >> lane) & 1))
              continue;
            uint32_t words[8][32] = {};
            for (unsigned reg = 0; reg < 8; ++reg)
              words[reg][0] = initial[reg][lane];
            const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]},
                           *c[] = {words[4], words[5]};
            uint32_t *d[] = {words[6], words[7]};
            ASSERT_EQ(
                functions[type](exact, 1, 0, d, a, b, c, (condition >> lane) & 1, &expected_flags),
                GOC_SUCCESS);
          }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t words[8][32], expected[2][32];
            std::memcpy(words, initial, sizeof(words));
            std::memcpy(expected, initial + 2 * dest, sizeof(expected));
            const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]},
                           *c[] = {words[4], words[5]};
            uint32_t *d[] = {words[dest * 2], words[dest * 2 + 1]},
                     *want[] = {expected[0], expected[1]}, exceptions = 0x80000000;
            ASSERT_EQ(functions[type](cpu | exact, exec_mask, 0, want, a, b, c, condition, nullptr),
                      GOC_SUCCESS);
            ASSERT_EQ(
                functions[type](cpu | exact, exec_mask, 0, d, a, b, c, condition, &exceptions),
                GOC_SUCCESS);
            EXPECT_EQ(exceptions, expected_flags);
            for (unsigned reg = 0; reg < 2; ++reg)
              for (unsigned lane = 0; lane < 32; ++lane)
                EXPECT_EQ(d[reg][lane], expected[reg][lane]);
          }
        }
}

TEST(DivFmas, ExceptionOptOutAndErrors) {
  for (auto fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[4][32] = {};
      const uint32_t *a[] = {words[0], words[1]};
      uint32_t *d[] = {words[2], words[3]}, exceptions = 0x80000000;
      ASSERT_EQ(fn(cpu, UINT32_MAX, 0, d, a, a, a, UINT32_MAX, &exceptions), GOC_SUCCESS);
      EXPECT_EQ(exceptions, 0x80000000);
      EXPECT_EQ(fn(cpu | exact, 0, 0, nullptr, nullptr, nullptr, nullptr, 0, &exceptions),
                GOC_SUCCESS);
      EXPECT_EQ(fn(cpu | exact, UINT32_MAX, 1ULL << 31, nullptr, nullptr, nullptr, nullptr, 0,
                   &exceptions),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(exceptions, 0x80000000);
    }
}

TEST(DivFmas, Wave64HardwareCartesianAndRandomCorpora) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned corpus = 0; corpus < 2; ++corpus)
        for (unsigned col = 0; col < 16; ++col) {
          uint32_t state = 0x9174ab23;
          uint64_t digest = goc_test::capture_hash_seed;
          for (unsigned start = 0; start < 4096; start += 64) {
            uint32_t data[8][64] = {};
            const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                           *c[] = {data[4], data[5]};
            uint32_t *d[] = {data[6], data[7]};
            for (unsigned lane = 0; lane < 64; ++lane) {
              unsigned index = start + lane,
                       indices[] = {index / 256, (index / 16) % 16, index % 16};
              uint64_t random_values[3];
              if (corpus)
                goc_test::fmas_capture_random(op, index, state, random_values);
              for (unsigned operand = 0; operand < 3; ++operand) {
                uint64_t value = corpus ? random_values[operand]
                                        : goc_test::fmas_capture_values[op][indices[operand]];
                data[operand * 2][lane] = uint32_t(value);
                data[operand * 2 + 1][lane] = uint32_t(value >> 32);
              }
            }
            ASSERT_EQ(wave64_functions[op](cpu | exact, UINT64_MAX,
                                           goc_test::fmas_capture_modes[col % 8], d, a, b, c,
                                           col / 8 ? UINT64_MAX : 0, nullptr),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 64; ++lane) {
              uint64_t value = data[6][lane];
              if (op)
                value |= uint64_t(data[7][lane]) << 32;
              digest = goc_test::capture_hash_bytes(digest, value, (op ? 8u : 4u));
            }
          }
          ASSERT_EQ(digest, (corpus ? goc_test::fmas_random_digests[op][col]
                                    : goc_test::fmas_capture_digests[op][col]))
              << op << "/" << cpu << "/" << col;
        }
}

TEST(DivFmas, Wave64MasksAndCrossRegisterAliases) {
  std::mt19937 random(975283);
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant)
        for (uint64_t exec_mask : {uint64_t(UINT64_MAX), uint64_t(0xffffffff00000000ULL),
                                   uint64_t(0x87654321fedcba98ULL)})
          for (unsigned first = 0; first < 8; first += (op ? 1 : 2))
            for (unsigned second = 0; second < (op ? 8u : 1u); ++second) {
              uint32_t data[8][66], expected[8][66], result[2][64];
              uint64_t condition = (uint64_t(random()) << 32) | random();
              for (auto &reg : data)
                for (auto &word : reg)
                  word = random();
              std::memcpy(expected, data, sizeof(data));
              uint32_t mode = goc_test::fmas_capture_modes[variant];
              const uint32_t *a[] = {data[0] + 1, data[1] + 1}, *b[] = {data[2] + 1, data[3] + 1},
                             *c[] = {data[4] + 1, data[5] + 1};
              uint32_t *d[] = {data[first] + 1, data[second] + 1}, *gold[] = {result[0], result[1]};
              ASSERT_EQ(
                  wave64_functions[op](exact, UINT64_MAX, mode, gold, a, b, c, condition, nullptr),
                  GOC_SUCCESS);
              unsigned dest[] = {first, second};
              for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
                for (unsigned lane = 0; lane < 64; ++lane)
                  if ((exec_mask >> lane) & 1)
                    expected[dest[reg]][lane + 1] = result[reg][lane];
              ASSERT_EQ(wave64_functions[op](cpu | (variant & 1 ? exact : 0), exec_mask, mode, d, a,
                                             b, c, condition, nullptr),
                        GOC_SUCCESS);
              ASSERT_EQ(std::memcmp(data, expected, sizeof(data)), 0)
                  << op << "/" << cpu << "/" << variant << "/" << first << "/" << second;
            }
}

TEST(DivFmas, Wave64SharedSourcesAndHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 2; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode) {
      uint32_t data[4][64], expected[2][64];
      for (unsigned lane = 0; lane < 64; ++lane) {
        uint64_t value = goc_test::fmas_capture_values[op][(lane + 3 * (lane / 32)) % 16];
        data[0][lane] = uint32_t(value);
        data[1][lane] = uint32_t(value >> 32);
      }
      const uint32_t *a[] = {data[0], data[1]};
      uint32_t *d[] = {data[2], data[3]}, *gold[] = {expected[0], expected[1]};
      uint32_t expected_flags = 0;
      ASSERT_EQ(wave64_functions[op](exact, UINT64_MAX, mode, gold, a, a, a, 0x5a5a5a5aa5a5a5a5ULL,
                                     &expected_flags),
                GOC_SUCCESS);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
          EXPECT_EQ(std::fesetround(rounding), 0);
          EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          if (mode & 1) {
            EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
          }
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          uint32_t actual_flags = 0;
          EXPECT_EQ(wave64_functions[op](cpu | exact, UINT64_MAX, mode, d, a, a, a,
                                         0x5a5a5a5aa5a5a5a5ULL, &actual_flags),
                    GOC_SUCCESS);
          EXPECT_EQ(actual_flags, expected_flags);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
          for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
            for (unsigned lane = 0; lane < 64; ++lane)
              ASSERT_EQ(d[reg][lane], expected[reg][lane]);
        }
    }
}

TEST(DivFmas, Wave64FusedRoundingAndOutputModifierBoundaries) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned col = 0; col < 8; ++col) {
        uint32_t data[8][64];
        const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                       *c[] = {data[4], data[5]};
        uint32_t *d[] = {data[6], data[7]};
        for (unsigned lane = 0; lane < 64; ++lane)
          for (unsigned operand = 0; operand < 3; ++operand) {
            uint64_t value = goc_test::fmas_boundary[op][lane % 8][operand];
            data[operand * 2][lane] = uint32_t(value);
            data[operand * 2 + 1][lane] = uint32_t(value >> 32);
          }
        ASSERT_EQ(wave64_functions[op](cpu | exact, UINT64_MAX,
                                       goc_test::fmas_capture_modes[col % 8], d, a, b, c,
                                       0x87654321fedcba98ULL, nullptr),
                  GOC_SUCCESS);
        for (unsigned lane = 0; lane < 64; ++lane) {
          uint64_t result = data[6][lane];
          if (op)
            result |= uint64_t(data[7][lane]) << 32;
          EXPECT_EQ(
              result,
              goc_test::fmas_boundary[op][lane % 8]
                                     [col + 3 + (((0x87654321fedcba98ULL >> lane) & 1) ? 8 : 0)])
              << op << "/" << cpu << "/" << col << "/" << lane;
        }
      }
}

TEST(DivFmas, Wave64ExceptionHardwareCorpus) {
  for (unsigned type = 0; type < 2; ++type) {
    unsigned fraction = type ? 52 : 23, bias = type ? 1023 : 127;
    uint64_t sign = 1ULL << (type ? 63 : 31), infinity = uint64_t(2 * bias + 1) << fraction,
             one = uint64_t(bias) << fraction;
    uint64_t values[] = {0,
                         sign,
                         1,
                         sign | 1,
                         (1ULL << fraction) - 1,
                         1ULL << fraction,
                         one,
                         sign | one,
                         one + (1ULL << fraction),
                         infinity - 1,
                         infinity - 1 - (1ULL << fraction),
                         infinity,
                         sign | infinity,
                         infinity | 1,
                         sign | infinity | (1ULL << (fraction - 1)) | 3,
                         2ULL << fraction};
    for (unsigned condition = 0; condition < 2; ++condition)
      for (unsigned variant = 0; variant < 16; ++variant) {
        uint32_t mode =
            ((variant & 3) << 6) | (variant & 4 ? GOC_ALU_CLAMP : 0) |
            (variant & 8 ? GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_C : 0);
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned i = 0; i < 8192; ++i) {
          uint32_t words[4][2][64] = {};
          uint64_t state = uint64_t(i) * 0x9e3779b97f4a7c15ULL;
          for (unsigned operand = 0; operand < 3; ++operand) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            uint64_t value =
                i < 4096 ? values[(i >> (8 - operand * 4)) & 15] : state * 0x2545f4914f6cdd1dULL;
            words[operand][0][63] = uint32_t(value);
            words[operand][1][63] = uint32_t(value >> 32);
          }
          const uint32_t *a[] = {words[0][0], words[0][1]}, *b[] = {words[1][0], words[1][1]},
                         *c[] = {words[2][0], words[2][1]};
          uint32_t *d[] = {words[3][0], words[3][1]};
          uint32_t exceptions = 0x80000000;
          ASSERT_EQ(wave64_functions[type](exact | (i % (goc_init_cpu_flags() + 1)), 1ULL << 63,
                                           mode, d, a, b, c, uint64_t(condition) << 63,
                                           &exceptions),
                    GOC_SUCCESS);
          ASSERT_TRUE(exceptions & 0x80000000);
          hash = goc_test::capture_hash_word(hash, exceptions & 127);
        }
        EXPECT_EQ(hash, goc_test::div_fmas_exception_hashes[type][condition][variant])
            << type << "/" << condition << "/" << variant;
      }
  }
}

TEST(DivFmas, Wave64ExceptionOptOutAndErrors) {
  for (auto fn : wave64_functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[4][64] = {};
      const uint32_t *a[] = {words[0], words[1]};
      uint32_t *d[] = {words[2], words[3]}, exceptions = 0x80000000;
      ASSERT_EQ(fn(cpu, UINT64_MAX, 0, d, a, a, a, UINT64_MAX, &exceptions), GOC_SUCCESS);
      EXPECT_EQ(exceptions, 0x80000000);
      EXPECT_EQ(fn(cpu | exact, 0, 0, nullptr, nullptr, nullptr, nullptr, 0, &exceptions),
                GOC_SUCCESS);
      EXPECT_EQ(fn(cpu | exact, UINT64_MAX, 1ULL << 31, nullptr, nullptr, nullptr, nullptr, 0,
                   &exceptions),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(exceptions, 0x80000000);
    }
}

TEST(DivFmas, Wave64ReportingAliasesUpperInput) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t data[6][64] = {}, actual[2][64] = {}, expected[2][64] = {};
      uint64_t one = op ? 0x3ff0000000000000ULL : 0x3f800000ULL;
      // Finite overflow in the lower half produces flags before lane 63 is read.
      uint64_t largest = op ? 0x7fefffffffffffffULL : 0x7f7fffffULL;
      data[0][0] = data[2][0] = uint32_t(largest);
      data[1][0] = data[3][0] = uint32_t(largest >> 32);
      data[0][63] = data[2][63] = uint32_t(one);
      data[1][63] = data[3][63] = uint32_t(one >> 32);
      const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                     *c[] = {data[4], data[5]};
      uint32_t *d[] = {actual[0], actual[1]}, *gold[] = {expected[0], expected[1]}, exceptions = 0;
      ASSERT_EQ(
          wave64_functions[op](cpu | exact, 1ULL | (1ULL << 63), 0, gold, a, b, c, 0, &exceptions),
          GOC_SUCCESS);
      ASSERT_NE(exceptions, 0U);
      uint32_t initial = data[0][63];
      ASSERT_EQ(
          wave64_functions[op](cpu | exact, 1ULL | (1ULL << 63), 0, d, a, b, c, 0, &data[0][63]),
          GOC_SUCCESS);
      EXPECT_EQ(data[0][63], initial | exceptions);
      EXPECT_EQ(std::memcmp(actual, expected, sizeof(actual)), 0);
    }
}

// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "div_scale_hardware.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stddef.h>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_div_scale_f32);
const Fn functions[] = {goc_v_div_scale_f32, goc_v_div_scale_f64_rdna4};
const decltype(&goc_v_div_scale_f32_wave64) wave64_functions[] = {goc_v_div_scale_f32_wave64,
                                                                  goc_v_div_scale_f64_rdna4_wave64};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

uint32_t mode_for(unsigned variant) {
  return (variant & 7) | (((variant >> 3) & 3) << 6) | (variant & 32 ? GOC_ALU_CLAMP : 0);
}

template <size_t Lanes>
void inputs(unsigned op, unsigned role, uint32_t mode, uint64_t b, uint64_t c,
            uint32_t (&data)[8][Lanes], unsigned lane) {
  uint64_t a = role ? c : b;
  if (bool(mode & GOC_ALU_NEG_A) != bool(mode & (role ? GOC_ALU_NEG_C : GOC_ALU_NEG_B)))
    a ^= 1ULL << (op ? 63 : 31);
  uint64_t values[] = {a, b, c};
  for (unsigned i = 0; i < 3; ++i) {
    data[2 * i][lane] = uint32_t(values[i]);
    data[2 * i + 1][lane] = uint32_t(values[i] >> 32);
  }
}

} // namespace

TEST(DivScale, HardwareCartesianCorpus) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned col = 0; col < 128; ++col) {
        uint64_t digest = goc_test::capture_hash_seed;
        for (unsigned start = 0; start < 576; start += 32) {
          uint32_t data[8][32] = {}, condition = 0;
          const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                         *c[] = {data[4], data[5]};
          uint32_t *d[] = {data[6], data[7]};
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned index = start + lane;
            inputs(op, col / 64, mode_for(col % 64), goc_test::scale_capture_values[op][index / 24],
                   goc_test::scale_capture_values[op][index % 24], data, lane);
          }
          ASSERT_EQ(
              functions[op](cpu | exact, UINT32_MAX, mode_for(col % 64), d, &condition, a, b, c),
              GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t value = data[6][lane];
            if (op)
              value |= uint64_t(data[7][lane]) << 32;
            digest = goc_test::capture_hash_bytes(digest, value, (op ? 8u : 4u));
            digest = goc_test::capture_hash_word(digest, (condition >> lane) & 1);
          }
        }
        ASSERT_EQ(digest, goc_test::scale_capture_digests[op][col])
            << op << "/" << cpu << "/" << col;
      }
}

TEST(DivScale, RandomBitsAllModifiersAndHostEnvironment) {
  std::mt19937 random(569827);
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned variant = 0; variant < 64; ++variant)
      for (unsigned sample = 0; sample < 16; ++sample) {
        uint64_t flags = exact | (sample & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                         (sample & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
        uint32_t data[8][32] = {}, expected[2][32], expected_condition;
        const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                       *c[] = {data[4], data[5]};
        uint32_t *d[] = {data[6], data[7]};
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint64_t bv = random(), cv = random();
          if (op) {
            bv |= uint64_t(random()) << 32;
            cv |= uint64_t(random()) << 32;
          }
          inputs(op, lane & 1, mode_for(variant), bv, cv, data, lane);
        }
        ASSERT_EQ(
            functions[op](flags, UINT32_MAX, mode_for(variant), d, &expected_condition, a, b, c),
            GOC_SUCCESS);
        std::memcpy(expected, data[6], sizeof(expected));
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            EXPECT_EQ(std::fesetround(rounding), 0);
            EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
            if (sample & 1) {
              EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
            }
            int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
            uint32_t condition = 0;
            EXPECT_EQ(functions[op](cpu | flags | GOC_FP16_OVFL, UINT32_MAX, mode_for(variant), d,
                                    &condition, a, b, c),
                      GOC_SUCCESS);
            EXPECT_EQ(std::fegetround(), rounding);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
            ASSERT_EQ(condition, expected_condition);
            for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
              for (unsigned lane = 0; lane < 32; ++lane)
                ASSERT_EQ(d[reg][lane], expected[reg][lane])
                    << op << "/" << cpu << "/" << variant << "/" << lane;
          }
      }
}

TEST(DivScale, MasksAndCrossRegisterAliases) {
  std::mt19937 random(871923);
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 64; ++variant)
        for (uint32_t exec_mask : exec_masks())
          for (unsigned first = 0; first < 8; first += (op ? 1 : 2))
            for (unsigned second = 0; second < (op ? 8u : 1u); ++second) {
              uint32_t data[8][32], expected[8][32], result[2][32], condition = 0xa5a5a5a5,
                                                                    expected_condition;
              for (auto &reg : data)
                for (auto &word : reg)
                  word = random();
              for (unsigned lane = 0; lane < 32; ++lane)
                inputs(op, lane & 1, mode_for(variant),
                       goc_test::scale_capture_values[op][(lane + variant) % 24],
                       goc_test::scale_capture_values[op][(lane * 7 + variant) % 24], data, lane);
              std::memcpy(expected, data, sizeof(data));
              const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                             *c[] = {data[4], data[5]};
              uint32_t *gold[] = {result[0], result[1]}, *d[] = {data[first], data[second]};
              ASSERT_EQ(functions[op](exact, UINT32_MAX, mode_for(variant), gold,
                                      &expected_condition, a, b, c),
                        GOC_SUCCESS);
              unsigned dest[] = {first, second};
              for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
                for (unsigned lane = 0; lane < 32; ++lane)
                  if ((exec_mask >> lane) & 1)
                    expected[dest[reg]][lane] = result[reg][lane];
              ASSERT_EQ(functions[op](cpu | (variant & 1 ? exact : 0), exec_mask, mode_for(variant),
                                      d, &condition, a, b, c),
                        GOC_SUCCESS);
              ASSERT_EQ(condition, expected_condition & exec_mask);
              ASSERT_EQ(std::memcmp(data, expected, sizeof(data)), 0)
                  << op << "/" << cpu << "/" << variant << "/" << first << "/" << second;
            }
}

TEST(DivScale, ValidationAndZeroExec) {
  for (auto fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t condition = 0xdeadbeef;
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!((1u << bit) & 0x1c7)) {
          EXPECT_EQ(fn(cpu, 0, 1u << bit, nullptr, &condition, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(condition, 0xdeadbeef);
        }
      EXPECT_EQ(fn(cpu | (1ull << 63), 0, 0, nullptr, &condition, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                   &condition, nullptr, nullptr, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(condition, 0xdeadbeef);
      EXPECT_EQ(fn(cpu | exact, 0U, 0, nullptr, &condition, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(condition, 0u);
      condition = 0xdeadbeef;
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, &condition, nullptr,
                   nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(condition, 0u);
    }
}

TEST(DivScale, SharedSourcesUnalignedStorageAndScalarOutputAlias) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant) {
        uint32_t source[2][34], output[2][34], expected[2][32], condition;
        for (unsigned lane = 0; lane < 34; ++lane) {
          uint64_t value = goc_test::scale_capture_values[op][lane % 24];
          source[0][lane] = uint32_t(value);
          source[1][lane] = uint32_t(value >> 32);
          output[0][lane] = output[1][lane] = 0xdeadbeef;
        }
        const uint32_t *a[] = {source[0] + 1, source[1] + 1};
        uint32_t *d[] = {output[0] + 1, output[1] + 1}, *gold[] = {expected[0], expected[1]};
        uint32_t mode = (variant & 1 ? 7 : 0) | ((variant >> 1) << 6);
        ASSERT_EQ(functions[op](exact, UINT32_MAX, mode, gold, &condition, a, a, a), GOC_SUCCESS);
        // A scalar destination may share storage; it is committed after all VGPR writes.
        ASSERT_EQ(functions[op](cpu | exact, UINT32_MAX, mode, d, d[0] + 13, a, a, a), GOC_SUCCESS);
        expected[0][13] = condition;
        for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg) {
          EXPECT_EQ(output[reg][0], 0xdeadbeef);
          EXPECT_EQ(output[reg][33], 0xdeadbeef);
          for (unsigned lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[reg][lane], expected[reg][lane]);
        }
      }
}

TEST(DivScale, Wave64HardwareCartesianCorpus) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned col = 0; col < 128; ++col) {
        uint64_t digest = goc_test::capture_hash_seed;
        for (unsigned start = 0; start < 576; start += 64) {
          uint32_t data[8][64] = {};
          uint64_t condition = 0;
          const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                         *c[] = {data[4], data[5]};
          uint32_t *d[] = {data[6], data[7]};
          for (unsigned lane = 0; lane < 64; ++lane) {
            unsigned index = start + lane;
            inputs(op, col / 64, mode_for(col % 64), goc_test::scale_capture_values[op][index / 24],
                   goc_test::scale_capture_values[op][index % 24], data, lane);
          }
          ASSERT_EQ(wave64_functions[op](cpu | exact, UINT64_MAX, mode_for(col % 64), d, &condition,
                                         a, b, c),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 64; ++lane) {
            uint64_t value = data[6][lane];
            if (op)
              value |= uint64_t(data[7][lane]) << 32;
            digest = goc_test::capture_hash_bytes(digest, value, (op ? 8u : 4u));
            digest = goc_test::capture_hash_word(digest, (condition >> lane) & 1);
          }
        }
        ASSERT_EQ(digest, goc_test::scale_capture_digests[op][col])
            << op << "/" << cpu << "/" << col;
      }
}

TEST(DivScale, Wave64RandomBitsAllModifiersAndHostEnvironment) {
  std::mt19937 random(569827);
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned variant = 0; variant < 64; ++variant)
      for (unsigned sample = 0; sample < 16; ++sample) {
        uint64_t flags = exact | (sample & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                         (sample & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
        uint32_t data[8][64] = {}, expected[2][64];
        uint64_t expected_condition;
        const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                       *c[] = {data[4], data[5]};
        uint32_t *d[] = {data[6], data[7]};
        for (unsigned lane = 0; lane < 64; ++lane) {
          uint64_t bv = random(), cv = random();
          if (op) {
            bv |= uint64_t(random()) << 32;
            cv |= uint64_t(random()) << 32;
          }
          inputs(op, lane & 1, mode_for(variant), bv, cv, data, lane);
        }
        ASSERT_EQ(wave64_functions[op](flags, UINT64_MAX, mode_for(variant), d, &expected_condition,
                                       a, b, c),
                  GOC_SUCCESS);
        std::memcpy(expected, data[6], sizeof(expected));
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            EXPECT_EQ(std::fesetround(rounding), 0);
            EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
            if (sample & 1) {
              EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
            }
            int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
            uint64_t condition = 0;
            EXPECT_EQ(wave64_functions[op](cpu | flags | GOC_FP16_OVFL, UINT64_MAX,
                                           mode_for(variant), d, &condition, a, b, c),
                      GOC_SUCCESS);
            EXPECT_EQ(std::fegetround(), rounding);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
            ASSERT_EQ(condition, expected_condition);
            for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
              for (unsigned lane = 0; lane < 64; ++lane)
                ASSERT_EQ(d[reg][lane], expected[reg][lane])
                    << op << "/" << cpu << "/" << variant << "/" << lane;
          }
      }
}

TEST(DivScale, Wave64MasksAndCrossRegisterAliases) {
  std::mt19937 random(871923);
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 64; ++variant)
        for (uint64_t exec_mask : {uint64_t(UINT64_MAX), uint64_t(0xffffffff00000000ULL),
                                   uint64_t(0x87654321fedcba98ULL)})
          for (unsigned first = 0; first < 8; first += (op ? 1 : 2))
            for (unsigned second = 0; second < (op ? 8u : 1u); ++second) {
              uint32_t data[8][64], expected[8][64], result[2][64];
              uint64_t condition = 0xa5a5a5a5a5a5a5a5ULL, expected_condition;
              for (auto &reg : data)
                for (auto &word : reg)
                  word = random();
              for (unsigned lane = 0; lane < 64; ++lane)
                inputs(op, lane & 1, mode_for(variant),
                       goc_test::scale_capture_values[op][(lane + variant) % 24],
                       goc_test::scale_capture_values[op][(lane * 7 + variant) % 24], data, lane);
              std::memcpy(expected, data, sizeof(data));
              const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                             *c[] = {data[4], data[5]};
              uint32_t *gold[] = {result[0], result[1]}, *d[] = {data[first], data[second]};
              ASSERT_EQ(wave64_functions[op](exact, UINT64_MAX, mode_for(variant), gold,
                                             &expected_condition, a, b, c),
                        GOC_SUCCESS);
              unsigned dest[] = {first, second};
              for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
                for (unsigned lane = 0; lane < 64; ++lane)
                  if ((exec_mask >> lane) & 1)
                    expected[dest[reg]][lane] = result[reg][lane];
              ASSERT_EQ(wave64_functions[op](cpu | (variant & 1 ? exact : 0), exec_mask,
                                             mode_for(variant), d, &condition, a, b, c),
                        GOC_SUCCESS);
              ASSERT_EQ(condition, expected_condition & exec_mask);
              ASSERT_EQ(std::memcmp(data, expected, sizeof(data)), 0)
                  << op << "/" << cpu << "/" << variant << "/" << first << "/" << second;
            }
}

TEST(DivScale, Wave64ValidationAndZeroExec) {
  for (auto fn : wave64_functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint64_t condition = 0xdeadbeef;
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!((1u << bit) & 0x1c7)) {
          EXPECT_EQ(fn(cpu, 0, 1u << bit, nullptr, &condition, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(condition, 0xdeadbeef);
        }
      EXPECT_EQ(fn(cpu | (1ull << 63), 0, 0, nullptr, &condition, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                   &condition, nullptr, nullptr, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(condition, 0xdeadbeef);
      EXPECT_EQ(fn(cpu | exact, 0U, 0, nullptr, &condition, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(condition, 0u);
      condition = 0xdeadbeef;
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, &condition, nullptr,
                   nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(condition, 0u);
    }
}

TEST(DivScale, Wave64UnalignedConditionAliases) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 64; ++variant)
        for (unsigned target : {0U, 2U, 4U, 6U, 7U})
          for (unsigned byte_offset : {53U, 133U}) {
            uint32_t data[8][66], expected[8][66], source[8][64] = {}, result[2][64];
            for (auto &reg : data)
              std::fill(reg, reg + 66, 0xdeadbeef);
            for (unsigned lane = 0; lane < 64; ++lane)
              inputs(op, lane & 1, mode_for(variant),
                     goc_test::scale_capture_values[op][(lane + variant) % 24],
                     goc_test::scale_capture_values[op][(lane * 7 + variant) % 24], source, lane);
            for (unsigned reg = 0; reg < 6; ++reg)
              std::memcpy(data[reg] + 1, source[reg], sizeof(source[reg]));
            std::memcpy(expected, data, sizeof(data));
            const uint32_t *a[] = {data[0] + 1, data[1] + 1}, *b[] = {data[2] + 1, data[3] + 1},
                           *c[] = {data[4] + 1, data[5] + 1};
            uint32_t *gold[] = {result[0], result[1]}, *d[] = {data[6] + 1, data[7] + 1};
            uint64_t condition = 0;
            const uint64_t exec_mask = 0x87654321fedcba98ULL;
            ASSERT_EQ(wave64_functions[op](exact, exec_mask, mode_for(variant), gold, &condition, a,
                                           b, c),
                      GOC_SUCCESS);
            for (unsigned reg = 0; reg < (op ? 2U : 1U); ++reg)
              for (unsigned lane = 0; lane < 64; ++lane)
                if ((exec_mask >> lane) & 1)
                  expected[reg + 6][lane + 1] = result[reg][lane];
            std::memcpy(reinterpret_cast<unsigned char *>(expected[target]) + byte_offset,
                        &condition, sizeof(condition));
            auto output = reinterpret_cast<uint64_t *>(
                reinterpret_cast<unsigned char *>(data[target]) + byte_offset);
            ASSERT_EQ(
                wave64_functions[op](cpu | exact, exec_mask, mode_for(variant), d, output, a, b, c),
                GOC_SUCCESS);
            EXPECT_EQ(std::memcmp(data, expected, sizeof(data)), 0)
                << op << "/" << target << "/" << byte_offset;
          }
}

TEST(DivScale, ArchitectureNanPayloadsAndAliases) {
  // rocjitsu shared/division.h, called with quiet_nan=false by RDNA3's
  // execute_v_div_scale_f64_vop3 and true by RDNA4. No new hardware model.
  const decltype(&goc_v_div_scale_f64) narrow[] = {goc_v_div_scale_f64, goc_v_div_scale_f64_rdna4};
  const decltype(&goc_v_div_scale_f64_wave64) wide[] = {goc_v_div_scale_f64_wave64,
                                                        goc_v_div_scale_f64_rdna4_wave64};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {uint64_t(0), exact})
      for (unsigned arch = 0; arch < 2; ++arch)
        for (unsigned width : {32U, 64U})
          for (unsigned role = 0; role < 2; ++role)
            for (unsigned alias = 0; alias < 4; ++alias)
              for (bool zero_other : {false, true})
                for (uint64_t exec_mask : {0ULL, 0xa55aa55a5aa55aa5ULL, ~0ULL})
                  for (unsigned variant = 0; variant < 16; ++variant) {
                    SCOPED_TRACE(::testing::Message()
                                 << cpu << '/' << semantics << '/' << arch << '/' << width << '/'
                                 << role << '/' << alias << '/' << zero_other << '/' << exec_mask
                                 << '/' << variant);
                    uint32_t words[8][64], before[8][64];
                    uint32_t mode = ((variant >> 1) & 3) << 6;
                    if (variant & 1)
                      mode |= GOC_ALU_NEG_A | (role ? GOC_ALU_NEG_C : GOC_ALU_NEG_B);
                    if (variant & 8)
                      mode |= GOC_ALU_CLAMP;
                    for (unsigned lane = 0; lane < 64; ++lane) {
                      uint64_t nan = 0x7ff0000000000001ULL | (uint64_t(lane) << 32);
                      if (lane & 1)
                        nan |= 1ULL << 63;
                      if (lane & 2)
                        nan |= 1ULL << 51;
                      uint64_t other = zero_other ? 0 : 0x3ff0000000000000ULL;
                      uint64_t values[] = {nan, role ? other : nan, role ? nan : other,
                                           0xdeadbeef01234567ULL};
                      for (unsigned reg = 0; reg < 4; ++reg) {
                        words[2 * reg][lane] = uint32_t(values[reg]);
                        words[2 * reg + 1][lane] = uint32_t(values[reg] >> 32);
                      }
                    }
                    std::memcpy(before, words, sizeof(words));
                    const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]},
                                   *c[] = {words[4], words[5]};
                    uint32_t *d[] = {words[2 * alias], words[2 * alias + 1]};
                    uint32_t condition32 = 0xdeadbeef;
                    uint64_t condition64 = 0xdeadbeefdeadbeefULL;
                    int status = width == 32 ? narrow[arch](cpu | semantics, uint32_t(exec_mask),
                                                            mode, d, &condition32, a, b, c)
                                             : wide[arch](cpu | semantics, exec_mask, mode, d,
                                                          &condition64, a, b, c);
                    ASSERT_EQ(status, GOC_SUCCESS);
                    if (width == 32) {
                      EXPECT_EQ(condition32, uint32_t(exec_mask));
                    } else {
                      EXPECT_EQ(condition64, exec_mask);
                    }
                    for (unsigned lane = 0; lane < 64; ++lane) {
                      uint64_t want =
                          before[2 * alias][lane] | (uint64_t(before[2 * alias + 1][lane]) << 32);
                      if (lane < width && ((exec_mask >> lane) & 1)) {
                        want = before[0][lane] | (uint64_t(before[1][lane]) << 32);
                        if (variant & 1)
                          want ^= 1ULL << 63;
                        if (arch)
                          want |= 1ULL << 51;
                        if (zero_other)
                          want = 0xfff8000000000000ULL;
                        if (mode & GOC_ALU_CLAMP)
                          want = 0;
                      }
                      EXPECT_EQ(d[0][lane], uint32_t(want));
                      EXPECT_EQ(d[1][lane], uint32_t(want >> 32));
                    }
                  }
}

TEST(DivScale, GuestDenormalPolicies) {
  // The input-flush policy is borrowed from rocjitsu DivisionTest's guest-mode
  // cases: a flushed tiny numerator/denominator produces the zero-operand NaN.
  const Fn narrow[] = {goc_v_div_scale_f32, goc_v_div_scale_f64, goc_v_div_scale_f64_rdna4};
  const decltype(&goc_v_div_scale_f32_wave64) wide[] = {
      goc_v_div_scale_f32_wave64, goc_v_div_scale_f64_wave64, goc_v_div_scale_f64_rdna4_wave64};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {uint64_t(0), exact})
      for (unsigned op = 0; op < 3; ++op)
        for (unsigned width : {32U, 64U})
          for (unsigned settings = 0; settings < 4; ++settings)
            for (unsigned pair = 0; pair < 4; ++pair)
              for (unsigned role = 0; role < 2; ++role)
                for (unsigned alias = 0; alias < 4; ++alias)
                  for (uint64_t exec_mask : {0ULL, 0xa55aa55a5aa55aa5ULL, ~0ULL}) {
                    SCOPED_TRACE(::testing::Message()
                                 << cpu << '/' << semantics << '/' << op << '/' << width << '/'
                                 << settings << '/' << pair << '/' << role << '/' << alias << '/'
                                 << exec_mask);
                    uint64_t one = op ? 0x3ff0000000000000ULL : 0x3f800000ULL;
                    uint64_t minimum = op ? 0x0010000000000000ULL : 0x00800000ULL;
                    uint64_t sign = op ? 1ULL << 63 : 1ULL << 31;
                    uint64_t bv = pair == 3 ? minimum : pair == 2 ? one : 1;
                    uint64_t cv = pair == 3 ? minimum : pair == 1 ? one : 1;
                    uint64_t flags = cpu | semantics |
                                     (settings & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                     (settings & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
                    uint32_t words[8][64], before[8][64];
                    for (unsigned lane = 0; lane < 64; ++lane) {
                      uint64_t a = role ? cv : bv;
                      uint64_t values[] = {a, bv, cv, 0xdeadbeef01234567ULL};
                      for (unsigned reg = 0; reg < 4; ++reg) {
                        uint64_t value = values[reg] ^ (lane & 1 && reg < 3 ? sign : 0);
                        words[2 * reg][lane] = uint32_t(value);
                        words[2 * reg + 1][lane] = uint32_t(value >> 32);
                      }
                    }
                    std::memcpy(before, words, sizeof(words));
                    const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]},
                                   *c[] = {words[4], words[5]};
                    uint32_t *d[] = {words[2 * alias], words[2 * alias + 1]};
                    uint32_t condition32 = 0xdeadbeef;
                    uint64_t condition64 = 0xdeadbeefdeadbeefULL;
                    int status =
                        width == 32
                            ? narrow[op](flags, uint32_t(exec_mask), 0, d, &condition32, a, b, c)
                            : wide[op](flags, exec_mask, 0, d, &condition64, a, b, c);
                    ASSERT_EQ(status, GOC_SUCCESS);
                    uint64_t want_condition = pair == 1 || pair == 2 ? exec_mask : 0;
                    if (width == 32) {
                      EXPECT_EQ(condition32, uint32_t(want_condition));
                    } else {
                      EXPECT_EQ(condition64, want_condition);
                    }
                    for (unsigned lane = 0; lane < 64; ++lane) {
                      uint64_t want =
                          before[2 * alias][lane] | (uint64_t(before[2 * alias + 1][lane]) << 32);
                      if (lane < width && ((exec_mask >> lane) & 1)) {
                        if ((settings & 1) && pair != 3)
                          want = op ? 0xfff8000000000000ULL : 0xffc00000ULL;
                        else {
                          // min-subnormal * 2^64/2^128, or min-normal * that scale.
                          want = pair == 3 ? (op ? 0x0810000000000000ULL : 0x20800000ULL)
                                 : (role ? cv : bv) == one ? one
                                 : op                      ? 0x04d0000000000000ULL
                                                           : 0x15000000ULL;
                          if (lane & 1)
                            want |= sign;
                        }
                      }
                      EXPECT_EQ(d[0][lane], uint32_t(want));
                      EXPECT_EQ(d[1][lane],
                                op ? uint32_t(want >> 32) : before[2 * alias + 1][lane]);
                    }
                  }
}

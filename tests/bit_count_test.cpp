// SPDX-License-Identifier: MIT

#include "bit_count_reference.h"
#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "mbcnt_dpp_hardware.h"
#include "test_instruction.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn = goc_test::WaveInstruction<decltype(&goc_v_bcnt_u32_b32)>;
const Fn functions[] = {goc_test::count_leading,       goc_test::count_trailing,
                        goc_test::count_sign,          goc_v_bcnt_u32_b32,
                        goc_v_mbcnt_lo_u32_b32,        goc_v_mbcnt_hi_u32_b32,
                        goc_v_mbcnt_lo_u32_b32_wave64, goc_v_mbcnt_hi_u32_b32_wave64};

::testing::AssertionResult check(int op, uint64_t flags, uint32_t words[3][64]) {
  unsigned lanes = op < 6 ? 32 : 64;
  uint32_t expected[64];
  for (unsigned lane = 0; lane < 64; ++lane)
    expected[lane] = lane < lanes
                         ? goc_test::bit_count_reference(op, words[0][lane], words[1][lane], lane)
                         : words[2][lane];
  const uint32_t *a[] = {words[0]}, *b[] = {words[1]};
  uint32_t *d[] = {words[2]};
  for (uint32_t mode : {0U, GOC_ALU_CLAMP}) {
    int status = functions[op](flags, UINT64_MAX, mode, d, a, b);
    if (status != GOC_SUCCESS)
      return ::testing::AssertionFailure() << "status " << status;
    for (int lane = 0; lane < 64; ++lane)
      if (words[2][lane] != expected[lane])
        return ::testing::AssertionFailure() << op << "/" << flags << "/" << mode << "/" << lane
                                             << ": " << words[2][lane] << " != " << expected[lane];
  }
  return ::testing::AssertionSuccess();
}

std::vector<uint64_t> masks(int op) {
  const auto wave32 = exec_masks();
  std::vector<uint64_t> result(wave32.begin(), wave32.end());
  if (op >= 6) {
    for (unsigned bit = 0; bit < 64; ++bit) {
      result.push_back(1ULL << bit);
      result.push_back(~(1ULL << bit));
    }
    result.push_back(0xaaaaaaaaaaaaaaaa);
    result.push_back(0x5555555555555555);
    std::mt19937_64 random(754);
    for (int i = 0; i < 16; ++i)
      result.push_back(random());
  }
  return result;
}

} // namespace

TEST(BitCount, EveryBitPairAndComplements) {
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned first = 0; first < 32; ++first)
        for (unsigned second = 0; second < 32; ++second)
          for (bool inverted : {false, true}) {
            uint32_t words[3][64];
            for (unsigned lane = 0; lane < 64; ++lane) {
              words[0][lane] =
                  ((uint32_t(1) << first) | (uint32_t(1) << second)) ^ (inverted ? UINT32_MAX : 0);
              words[1][lane] = UINT32_MAX - lane / 2;
              words[2][lane] = 0x859e43c1;
            }
            ASSERT_TRUE(
                check(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, words));
          }
}

TEST(BitCount, EveryHalfEncodingAndRandomFullWords) {
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      std::mt19937 random(1435);
      for (unsigned start = 0; start < 131072; start += 32) {
        uint32_t words[3][64];
        for (unsigned lane = 0; lane < 64; ++lane) {
          unsigned value = (start + lane) % 65536;
          words[0][lane] = start < 65536 ? value | ((65535 - value) << 16) : random();
          words[1][lane] = random();
          words[2][lane] = random();
        }
        ASSERT_TRUE(check(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, words));
      }
    }
}

TEST(BitCount, LiteralSentinelsAndPhysicalLaneNumbers) {
  const uint32_t values[] = {0, 0xffffffff, 0x80000000, 1, 0x7fffffff, 0xfffffffe};
  const uint32_t answers[3][6] = {{UINT32_MAX, 0, 0, 31, 1, 0},
                                  {UINT32_MAX, 0, 31, 0, 0, 1},
                                  {UINT32_MAX, UINT32_MAX, 1, 31, 1, 31}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int op = 0; op < 8; ++op)
      for (int item = 0; item < 6; ++item) {
        uint32_t words[3][64] = {};
        std::fill_n(words[0], 64, op < 3 ? values[item] : UINT32_MAX);
        std::fill_n(words[1], 64, UINT32_MAX);
        const uint32_t *a[] = {words[0]}, *b[] = {words[1]};
        uint32_t *d[] = {words[2]};
        ASSERT_EQ(functions[op](cpu, UINT64_MAX, 0, d, a, b), GOC_SUCCESS);
        for (unsigned lane = 0; lane < (op < 6 ? 32u : 64u); ++lane) {
          uint32_t expected = op < 3                 ? answers[op][item]
                              : op == 3              ? 31
                              : (op == 4 || op == 6) ? std::min(lane, 32u) - 1
                                                     : (lane < 32 ? 0 : lane - 32) - 1;
          EXPECT_EQ(words[2][lane], expected);
        }
      }
}

TEST(BitCount, EveryPopulationAndAccumulatorWrapping) {
  for (int op = 3; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned ones = 0; ones <= 32; ++ones) {
        uint32_t words[3][64];
        for (unsigned lane = 0; lane < 64; ++lane) {
          words[0][lane] = uint32_t((1ULL << ones) - 1);
          words[1][lane] = lane % 2 ? UINT32_MAX - ones / 2 : 0;
          words[2][lane] = 0x859e43c1;
        }
        ASSERT_TRUE(check(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, words));
      }
}

TEST(BitCount, MaskedCountCompositionReturnsPhysicalLane) {
  for (bool wave64 : {false, true})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t exec_mask : masks(wave64 ? 6 : 4)) {
        uint32_t input[64], zero[64] = {}, result[64];
        std::fill_n(input, 64, UINT32_MAX);
        std::fill_n(result, 64, 0x859e43c1);
        const uint32_t *a[] = {input}, *b[] = {zero}, *previous[] = {result};
        uint32_t *d[] = {result};
        ASSERT_EQ(functions[wave64 ? 6 : 4](cpu, exec_mask, 0, d, a, b), GOC_SUCCESS);
        ASSERT_EQ(functions[wave64 ? 7 : 5](cpu, exec_mask, 0, d, a, previous), GOC_SUCCESS);
        for (unsigned lane = 0; lane < 64; ++lane) {
          bool active = lane < (wave64 ? 64u : 32u) && ((exec_mask >> lane) & 1);
          EXPECT_EQ(result[lane], active ? lane : 0x859e43c1);
        }
      }
}

TEST(BitCount, MasksAliasesAndUnalignedStorage) {
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool same_sources : {false, true}) {
        std::mt19937 random(876);
        uint32_t original[3][67];
        for (auto &reg : original)
          for (auto &word : reg)
            word = random();
        int breg = same_sources ? 0 : 1;
        for (uint64_t exec_mask : masks(op))
          for (int target = 0; target < 3; ++target) {
            uint32_t words[3][67], expected[3][67];
            for (int reg = 0; reg < 3; ++reg) {
              std::copy_n(original[reg], 67, words[reg]);
              std::copy_n(original[reg], 67, expected[reg]);
            }
            for (unsigned lane = 0; lane < (op < 6 ? 32u : 64u); ++lane)
              if ((exec_mask >> lane) & 1)
                expected[target][lane + 1] = goc_test::bit_count_reference(
                    op, original[0][lane + 1], original[breg][lane + 1], lane);
            const uint32_t *a[] = {words[0] + 1}, *b[] = {words[breg] + 1};
            uint32_t *d[] = {words[target] + 1};
            ASSERT_EQ(functions[op](cpu, exec_mask, 0, d, a, b), GOC_SUCCESS);
            for (int reg = 0; reg < 3; ++reg)
              ASSERT_TRUE(std::equal(words[reg], words[reg] + 67, expected[reg]));
          }
      }
}

TEST(BitCount, ValidationAndHostFpState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (int flush = 0; flush < 2; ++flush) {
      std::fesetround(rounding);
      std::feclearexcept(FE_ALL_EXCEPT);
      std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      _mm_setcsr((_mm_getcsr() & ~0x8040u) | (flush ? 0x8040u : 0));
      unsigned before = _mm_getcsr();
#endif
      for (int op = 0; op < 8; ++op)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t words[3][64];
          for (auto &reg : words)
            std::fill_n(reg, 64, 0x7f800001);
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]};
          uint32_t *d[] = {words[2]};
          for (uint64_t exec_mask : std::initializer_list<uint64_t>{0ULL, UINT64_MAX}) {
            for (int bit = 0; bit < 32; ++bit) {
              if ((1U << bit) == GOC_ALU_CLAMP)
                continue;
              EXPECT_EQ(functions[op](cpu, exec_mask, uint32_t(1) << bit, d, a, b),
                        GOC_ERROR_INVALID_FLAGS);
            }
            EXPECT_EQ(functions[op](cpu | (1ULL << 63), exec_mask, 0, d, a, b),
                      GOC_ERROR_INVALID_FLAGS);
          }
          for (const auto &reg : words)
            for (uint32_t value : reg)
              EXPECT_EQ(value, 0x7f800001);
          EXPECT_TRUE(
              check(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT | GOC_FP16_OVFL,
                    words));
        }
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), before);
#endif
    }
}

TEST(BitCount, Wave64DppUsesPhysicalDestinationLane) {
  for (unsigned op : {6U, 7U})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t clamp : {0U, GOC_ALU_CLAMP})
        for (uint64_t mode : goc_test::dpp_modes)
          for (unsigned rows = 0; rows < 16; ++rows)
            for (uint64_t exec_mask : {0ULL, 1ULL << 63, 0xaaaaaaaa55555555ULL, ~0ULL})
              for (unsigned target = 0; target < 3; ++target) {
                uint32_t data[3][66], expected[3][66];
                for (unsigned r = 0; r < 3; ++r)
                  for (unsigned lane = 0; lane < 66; ++lane)
                    data[r][lane] = expected[r][lane] =
                        (0x85930271U * lane) ^ (0xffffffffU - r * 12345);
                uint64_t descriptor = mode | clamp;
                if (descriptor & GOC_DPP16)
                  descriptor =
                      (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(rows) << GOC_DPP_ROW_SHIFT);
                for (unsigned lane = 0; lane < 64; ++lane) {
                  int source;
                  if (goc_test::dpp_source(descriptor, exec_mask, lane, source))
                    expected[target][lane + 1] = goc_test::bit_count_reference(
                        op, source < 0 ? 0 : data[0][source + 1], data[1][lane + 1], lane);
                }
                auto d = data[target] + 1;
                const uint32_t *a = data[0] + 1, *b = data[1] + 1;
                ASSERT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                        exec_mask, descriptor, &d, &a, &b),
                          GOC_SUCCESS);
                for (unsigned r = 0; r < 3; ++r)
                  ASSERT_TRUE(std::equal(data[r], data[r] + 66, expected[r]))
                      << op << '/' << mode << '/' << rows;
              }
  for (unsigned op : {6U, 7U}) {
    EXPECT_EQ(functions[op](0, 0, GOC_DPP8, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(functions[op](0, UINT64_MAX, GOC_DPP8 | GOC_ALU_NEG_A, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(functions[op](0, UINT64_MAX, GOC_DPP8 | GOC_DPP16, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
  }
}

TEST(BitCount, Wave64DppHardwareCorpus) {
  const uint64_t exec_masks[] = {0, 1ULL << 63, 0xaaaaaaaa55555555ULL, UINT64_MAX};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 2; ++op)
      for (unsigned m = 0; m < 7; ++m)
        for (unsigned e = 0; e < 4; ++e) {
          uint32_t a[64], b[64], d[64];
          for (unsigned lane = 0; lane < 64; ++lane) {
            a[lane] = (0x85930271U * lane) ^ 0xffffffffU;
            b[lane] = 0xffffffe0U + lane;
            d[lane] = 0xdead0000U + lane;
          }
          auto pa = a, pb = b, pd = d;
          ASSERT_EQ(functions[6 + op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                      exec_masks[e], goc_test::dpp_modes[m], &pd, &pa, &pb),
                    GOC_SUCCESS);
          uint64_t hash = goc_test::capture_hash_seed;
          for (auto word : d)
            hash = goc_test::capture_hash_word(hash, word);
          EXPECT_EQ(hash, goc_test::mbcnt_dpp_wave64_hashes[(op * 7 + m) * 4 + e])
              << cpu << '/' << op << '/' << m << '/' << e;
        }
}

namespace {

int count_wave64(unsigned op, uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b) {
  switch (op) {
  case 0:
    return goc_v_clz_i32_u32_wave64(flags, exec_mask, mode, d, a);
  case 1:
    return goc_v_ctz_i32_b32_wave64(flags, exec_mask, mode, d, a);
  case 2:
    return goc_v_cls_i32_wave64(flags, exec_mask, mode, d, a);
  default:
    return goc_v_bcnt_u32_b32_wave64(flags, exec_mask, mode, d, a, b);
  }
}

} // namespace

TEST(BitCount, Wave64ScansAndPopulationDppMasksAliases) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (uint32_t clamp : {0U, GOC_ALU_CLAMP})
          for (int routing = -1; routing < 7; ++routing)
            for (uint64_t exec_mask : {0ULL, 1ULL << 63, 0xaaaaaaaa55555555ULL, ~0ULL})
              for (unsigned target = 0; target < 3; ++target) {
                uint32_t data[3][66], expected[3][66];
                for (unsigned r = 0; r < 3; ++r)
                  for (unsigned lane = 0; lane < 66; ++lane)
                    data[r][lane] = expected[r][lane] = (0x73951a83U * lane) ^ (0xfffffffeU * r);
                uint64_t mode = clamp | (routing < 0 ? 0 : goc_test::dpp_modes[routing]);
                for (unsigned lane = 0; lane < 64; ++lane) {
                  int source = int(lane);
                  bool write = routing < 0 ? (exec_mask >> lane) & 1
                                           : goc_test::dpp_source(mode, exec_mask, lane, source);
                  if (write)
                    expected[target][lane + 1] = goc_test::bit_count_reference(
                        op, source < 0 ? 0 : data[0][source + 1], data[1][lane + 1], lane);
                }
                auto d = data[target] + 1;
                const uint32_t *a = data[0] + 1, *b = data[1] + 1;
                ASSERT_EQ(count_wave64(op, cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask, mode,
                                       &d, &a, &b),
                          GOC_SUCCESS);
                for (unsigned r = 0; r < 3; ++r)
                  ASSERT_TRUE(std::equal(data[r], data[r] + 66, expected[r]))
                      << op << '/' << routing;
              }
}

TEST(BitCount, Wave64ScansAndPopulationEveryHalfWord) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned start = 0; start < 65536; start += 64)
        for (unsigned pattern : {0U, 1U, 2U}) {
          uint32_t a[64], b[64], d[64];
          for (unsigned lane = 0; lane < 64; ++lane) {
            uint32_t value = start + lane;
            a[lane] = pattern == 0 ? value : pattern == 1 ? value << 16 : value | (value << 16);
            b[lane] = UINT32_MAX - lane;
          }
          auto pa = a, pb = b, pd = d;
          ASSERT_EQ(count_wave64(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                 UINT64_MAX, 0, &pd, &pa, &pb),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 64; ++lane)
            ASSERT_EQ(d[lane], goc_test::bit_count_reference(op, a[lane], b[lane], lane));
        }
}

TEST(BitCount, Wave64ScansAndPopulationValidationAndHostState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 4; ++op) {
    EXPECT_EQ(count_wave64(op, 0, 0, 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(count_wave64(op, 0, UINT64_MAX, GOC_DPP8 | GOC_ALU_NEG_A, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        uint32_t a[64], b[64], d[64];
        for (unsigned lane = 0; lane < 64; ++lane) {
          a[lane] = 0x7f800001U + lane;
          b[lane] = UINT32_MAX;
        }
        auto pa = a, pb = b, pd = d;
        ASSERT_EQ(std::fesetround(rounding), 0);
        ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
        ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
#if defined(__x86_64__) || defined(_M_X64)
        unsigned before = _mm_getcsr();
#endif
        ASSERT_EQ(count_wave64(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                               UINT64_MAX, goc_test::dpp_modes[3], &pd, &pa, &pb),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
        EXPECT_EQ(_mm_getcsr(), before);
#endif
      }
  }
}

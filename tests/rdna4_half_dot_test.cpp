// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>
#include <vector>

namespace {

using Fn = decltype(&goc_rdna4_v_fma_f32);
const Fn functions[] = {goc_rdna4_v_dot2_f16_f16, goc_rdna4_v_dot2_bf16_bf16};

// Exact encoding for the small integer reference results.
uint16_t encode(int value, bool bf16) {
  if (!value)
    return 0;
  int exponent;
  double mantissa = std::frexp(double(std::abs(value)), &exponent) * 2;
  int fraction = bf16 ? 7 : 10, bias = bf16 ? 127 : 15;
  return uint16_t((value < 0 ? 0x8000 : 0) | ((exponent - 1 + bias) << fraction) |
                  int(std::ldexp(mantissa - 1, fraction)));
}

} // namespace

TEST(HalfDot, AllModifiersHalfSelectionsMasksAndAliases) {
  for (int brain = 0; brain < 2; ++brain)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 256; ++variant) {
        uint32_t mode = (variant & 63) | (variant & 64 ? GOC_ALU_HIGH_C : 0) |
                        (variant & 128 ? GOC_ALU_HIGH_D : 0);
        for (uint32_t mask : rdna4_exec_masks())
          for (int alias = 0; alias < 4; ++alias) {
            SCOPED_TRACE(::testing::Message()
                         << brain << "/" << cpu << "/" << variant << "/" << mask << "/" << alias);
            uint32_t storage[4][34], expected[32], before[32];
            uint32_t *v[4];
            for (int reg = 0; reg < 4; ++reg) {
              std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
            }
            for (int lane = 0; lane < 32; ++lane) {
              int values[3][2] = {{lane % 5 - 2, lane % 3 - 1},
                                  {lane % 7 - 3, lane % 5 - 2},
                                  {lane - 16, 16 - lane}};
              for (int reg = 0; reg < 3; ++reg) {
                v[reg][lane] =
                    encode(values[reg][0], brain) | (uint32_t(encode(values[reg][1], brain)) << 16);
                for (int half = 0; half < 2; ++half) {
                  if (mode & (8u << reg))
                    values[reg][half] = std::abs(values[reg][half]);
                  if (mode & (1u << reg))
                    values[reg][half] = -values[reg][half];
                }
              }
              int result = values[0][0] * values[1][0] + values[0][1] * values[1][1] +
                           values[2][bool(mode & GOC_ALU_HIGH_C)];
              before[lane] = v[alias][lane];
              int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
              expected[lane] =
                  (before[lane] & ~(0xffffu << shift)) | (uint32_t(encode(result, brain)) << shift);
            }
            ASSERT_EQ(functions[brain](cpu, mask, mode, &v[alias], &v[0], &v[1], &v[2]),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              EXPECT_EQ(v[alias][lane], ((mask >> lane) & 1) ? expected[lane] : before[lane]);
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
      }
}

TEST(HalfDot, EveryAccumulatorEncoding) {
  for (int brain = 0; brain < 2; ++brain)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned base = 0; base < 65536; base += 32) {
        uint32_t a[32] = {}, c[32], d[32];
        for (int lane = 0; lane < 32; ++lane) {
          c[lane] = base + lane;
          d[lane] = 0xabcd1234;
        }
        auto pa = a, pc = c, pd = d;
        ASSERT_EQ(functions[brain](cpu, UINT32_MAX, 0, &pd, &pa, &pa, &pc), GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint32_t magnitude = (base + lane) & 0x7fff, infinity = brain ? 0x7f80 : 0x7c00;
          EXPECT_EQ(d[lane] >> 16, 0xabcdu);
          if (magnitude > infinity) {
            EXPECT_GT(d[lane] & 0x7fff, infinity);
          } else {
            uint32_t want = base + lane;
            if (!magnitude || (brain && magnitude < 128))
              want = 0;
            EXPECT_EQ(d[lane] & 0xffff, want);
          }
        }
      }
}

TEST(HalfDot, LiteralRoundingDenormalsAndOverflow) {
  struct Case {
    int brain;
    uint32_t a, b, c;
    uint64_t flags;
    uint16_t expected;
  };

  const Case cases[] = {
      {0, 0x80008000, 0x3c003c00, 0x8000, 0, 0x8000},
      {1, 0x80008000, 0x3f803f80, 0x8000, 0, 0x8000},
      // F16 half-way sums: ties down to even 1, and up to the next even mantissa.
      {0, 0x00003c00, 0x00003c00, 0x1000, 0, 0x3c00},
      {0, 0x00003c01, 0x00003c00, 0x1000, 0, 0x3c02},
      {0, 0x00007bff, 0x00004000, 0, 0, 0x7c00},
      {0, 0x00007bff, 0x00004000, 0, GOC_FP16_OVFL, 0x7bff},
      {0, 0x0000fbff, 0x00004000, 0, GOC_FP16_OVFL, 0xfbff},
      {0, 0x00007c00, 0x00003c00, 0, GOC_FP16_OVFL, 0x7c00},
      {0, 0x00000001, 0x00003c00, 0, 0, 1},
      {0, 0x00000001, 0x00003800, 0, 0, 0},
      {0, 0x00000003, 0x00003800, 0, 0, 2},
      // BF16 input subnormals flush before arithmetic; output subnormals flush too.
      {1, 0x00000001, 0x00007f00, 0, 0, 0},
      {1, 0x00000080, 0x00003f00, 0, 0, 0},
      {1, 0x00003f80, 0x00003f80, 0x3b80, 0, 0x3f80},
      {1, 0x00003f81, 0x00003f80, 0x3b80, 0, 0x3f82}};
  for (auto test : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[32], b[32], c[32], d[32];
      std::fill(a, a + 32, test.a);
      std::fill(b, b + 32, test.b);
      std::fill(c, c + 32, test.c);
      std::fill(d, d + 32, 0x12340000);
      auto pa = a, pb = b, pc = c, pd = d;
      ASSERT_EQ(functions[test.brain](cpu | test.flags, UINT32_MAX, 0, &pd, &pa, &pb, &pc),
                GOC_SUCCESS);
      for (auto word : d)
        EXPECT_EQ(word, 0x12340000u | test.expected);
    }
}

TEST(HalfDot, Validation) {
  for (auto fn : functions) {
    uint32_t a[32] = {}, d[32];
    std::fill(d, d + 32, 0xdeadbeef);
    auto pa = a, pd = d;
    for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
      for (uint32_t invalid : {GOC_ALU_CLAMP, GOC_ALU_OMOD_2, UINT32_C(1) << 31})
        EXPECT_EQ(fn(0, mask, invalid, &pd, &pa, &pa, &pa), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(
          fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, &pd, &pa, &pa, &pa),
          GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : d)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &pd, &pa, &pa, &pa), GOC_SUCCESS);
    for (auto word : d)
      EXPECT_EQ(word, 0xdead0000u);
  }
}

TEST(HalfDot, RandomEncodingsSimdMatchesScalar) {
  std::mt19937 rng(11719);
  for (int brain = 0; brain < 2; ++brain)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int trial = 0; trial < 256; ++trial) {
        uint32_t a[32], b[32], c[32], ref[32], result[32];
        for (int lane = 0; lane < 32; ++lane) {
          a[lane] = rng();
          b[lane] = rng();
          c[lane] = rng();
          ref[lane] = result[lane] = rng();
        }
        uint32_t mode = rng() & (63 | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D);
        uint64_t fp = trial & 1 ? GOC_FP16_OVFL : 0;
        uint32_t mask = rdna4_exec_masks()[trial % rdna4_exec_masks().size()];
        auto pa = a, pb = b, pc = c, pr = ref, pd = result;
        ASSERT_EQ(functions[brain](fp, mask, mode, &pr, &pa, &pb, &pc), GOC_SUCCESS);
        ASSERT_EQ(functions[brain](fp | cpu, mask, mode, &pd, &pa, &pb, &pc), GOC_SUCCESS);
        int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
        for (int lane = 0; lane < 32; ++lane) {
          uint16_t expected = uint16_t(ref[lane] >> shift),
                   actual = uint16_t(result[lane] >> shift);
          if (((mask >> lane) & 1) && (expected & 0x7fff) > (brain ? 0x7f80 : 0x7c00)) {
            EXPECT_GT(actual & 0x7fff, brain ? 0x7f80 : 0x7c00);
            EXPECT_EQ(result[lane] & ~(0xffffu << shift), ref[lane] & ~(0xffffu << shift));
          } else {
            EXPECT_EQ(result[lane], ref[lane]);
          }
        }
      }
}

TEST(HalfDot, DppModifiersMasksAliasesAndGuards) {
  for (int brain = 0; brain < 2; ++brain)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto descriptor : goc_test::dpp_modes)
        for (unsigned variant = 0; variant < 256; ++variant) {
          uint64_t mode = descriptor | (variant & 63) | (variant & 64 ? GOC_ALU_HIGH_C : 0) |
                          (variant & 128 ? GOC_ALU_HIGH_D : 0);
          auto masks =
              (variant == 0 || variant == 63 || variant == 64 || variant == 128 || variant == 255)
                  ? rdna4_exec_masks()
                  : std::vector<uint32_t>{UINT32_MAX};
          for (auto mask : masks)
            for (unsigned alias = 0; alias < 4; ++alias) {
              uint32_t storage[4][34], expected[4][34];
              int values[3][32][2];
              for (unsigned reg = 0; reg < 4; ++reg)
                std::fill_n(storage[reg], 34, 0xdeadbeef);
              for (unsigned lane = 0; lane < 32; ++lane) {
                int raw[3][2] = {{int(lane % 5) - 2, int(lane % 3) - 1},
                                 {int(lane % 7) - 3, int(lane % 5) - 2},
                                 {int(lane) - 16, 16 - int(lane)}};
                for (unsigned reg = 0; reg < 3; ++reg) {
                  storage[reg][lane + 1] =
                      encode(raw[reg][0], brain) | (uint32_t(encode(raw[reg][1], brain)) << 16);
                  for (unsigned half = 0; half < 2; ++half) {
                    int v = raw[reg][half];
                    if (mode & (8u << reg))
                      v = std::abs(v);
                    if (mode & (1u << reg))
                      v = -v;
                    values[reg][lane][half] = v;
                  }
                }
              }
              for (unsigned reg = 0; reg < 4; ++reg)
                std::copy_n(storage[reg], 34, expected[reg]);
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source = 0;
                if (!goc_test::dpp_source(mode, mask, lane, source))
                  continue;
                int a0 = source < 0 ? 0 : values[0][source][0],
                    a1 = source < 0 ? 0 : values[0][source][1];
                int result = a0 * values[1][lane][0] + a1 * values[1][lane][1] +
                             values[2][lane][bool(mode & GOC_ALU_HIGH_C)];
                unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
                expected[alias][lane + 1] = (expected[alias][lane + 1] & ~(0xffffu << shift)) |
                                            (uint32_t(encode(result, brain)) << shift);
              }
              const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[1] + 1},
                             *c[] = {storage[2] + 1};
              uint32_t *d[] = {storage[alias] + 1};
              ASSERT_EQ(functions[brain](cpu, mask, mode, d, a, b, c), GOC_SUCCESS);
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  ASSERT_EQ(storage[reg][word], expected[reg][word])
                      << brain << "/" << cpu << "/" << mode << "/" << mask;
            }
        }
}

TEST(HalfDot, DppValidation) {
  for (auto fn : functions)
    for (auto descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(fn(0, 0, descriptor, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (auto invalid : {UINT64_C(1) << 36, uint64_t(GOC_ALU_CLAMP), uint64_t(GOC_ALU_HIGH_A),
                           uint64_t(GOC_ALU_HIGH_B)})
        EXPECT_EQ(fn(0, 0, descriptor | invalid, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
    }
}

// RX 9070 finite binary-exact inputs, all FI/BOUND pairs and both overflow modes.
TEST(HalfDot, DppHardwareCorpus) {
  const uint32_t values[2][8] = {{0x3c00bc00, 0x4000c000, 0x4200c200, 0x4400c400, 0x00003c00,
                                  0x3800b800, 0x4500c500, 0x4600c600},
                                 {0x3f80bf80, 0x4000c000, 0x4040c040, 0x4080c080, 0x00003f80,
                                  0x3f00bf00, 0x40a0c0a0, 0x40c0c0c0}};
  const uint32_t modes[] = {0, 1, 2, 4, 8, 16, 32, 63, 2048, 4096, 6144, 6207};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (bool sat : {false, true})
      for (auto mask : masks)
        for (unsigned brain = 0; brain < 2; ++brain)
          for (auto descriptor : goc_test::dpp_modes)
            for (auto mode : modes) {
              uint32_t av[32], bv[32], cv[32], output[32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                av[lane] = values[brain][lane % 8];
                bv[lane] = values[brain][(lane + 3) % 8];
                cv[lane] = values[brain][(lane + 5) % 8];
                output[lane] = 0xdead0000u + lane;
              }
              const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
              uint32_t *d[] = {output};
              ASSERT_EQ(functions[brain](cpu | (sat ? GOC_FP16_OVFL : 0), mask, descriptor | mode,
                                         d, a, b, c),
                        GOC_SUCCESS);
              for (auto word : output)
                hash = goc_test::capture_hash_word(hash, word);
            }
    EXPECT_EQ(hash, UINT64_C(0xf79822666bfa7c45)) << cpu;
  }
}

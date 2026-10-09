// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dot_fixtures.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>
#include <vector>

TEST(Dot2, AllModifiersSelectionsMasksAndAliases) {
  const uint16_t f16[] = {0xc000, 0xbc00, 0x3c00, 0x4000};
  const uint16_t bf16[] = {0xc000, 0xbf80, 0x3f80, 0x4000};
  const int values[] = {-2, -1, 1, 2};
  for (bool brain : {false, true}) {
    const auto *codes = brain ? bf16 : f16;
    auto fn = brain ? goc_v_dot2_f32_bf16 : goc_v_dot2_f32_f16;
    for (uint32_t selection = 0; selection < 16; ++selection)
      for (uint32_t negate = 0; negate < 32; ++negate)
        for (uint32_t clamp : {0U, GOC_DOT_CLAMP}) {
          const uint32_t mode = negate | (selection << 7) | clamp;
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
              for (uint32_t exec_mask : {UINT32_MAX, 0xaaaaaaaaU, 0U, 0x80018001U})
                for (int alias = 0; alias < 4; ++alias) {
                  SCOPED_TRACE(::testing::Message()
                               << brain << "/" << mode << "/" << cpu << "/" << semantics << "/"
                               << exec_mask << "/" << alias);
                  uint32_t storage[4][34], before[32], expected[32];
                  uint32_t *v[4];
                  for (int r = 0; r < 4; ++r) {
                    std::fill(storage[r], storage[r] + 34, 0xdeadbeef);
                    v[r] = storage[r] + 1;
                  }
                  for (int lane = 0; lane < 32; ++lane) {
                    int ix[4] = {lane % 4, (lane / 4) % 4, (lane + 1) % 4, (lane / 4 + 2) % 4};
                    v[0][lane] = codes[ix[0]] | (uint32_t(codes[ix[1]]) << 16);
                    v[1][lane] = codes[ix[2]] | (uint32_t(codes[ix[3]]) << 16);
                    int c = lane - 16;
                    v[2][lane] = goc::as_bits(float(c));
                    int x0 = values[ix[selection & 1 ? 1 : 0]];
                    int y0 = values[ix[selection & 2 ? 3 : 2]];
                    int x1 = values[ix[selection & 4 ? 0 : 1]];
                    int y1 = values[ix[selection & 8 ? 2 : 3]];
                    if (negate & 1)
                      x0 = -x0;
                    if (negate & 2)
                      y0 = -y0;
                    if (negate & 4)
                      c = -c;
                    if (negate & 8)
                      x1 = -x1;
                    if (negate & 16)
                      y1 = -y1;
                    expected[lane] = goc::as_bits(float(x0 * y0 + x1 * y1 + c));
                  }
                  std::copy(v[alias], v[alias] + 32, before);
                  uint32_t exceptions = 0x80000055;
                  ASSERT_EQ(fn(cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask, mode, &v[alias],
                               &v[0], &v[1], &v[2], &exceptions),
                            GOC_SUCCESS);
                  EXPECT_EQ(exceptions, 0x80000055);
                  for (int lane = 0; lane < 32; ++lane)
                    EXPECT_EQ(v[alias][lane],
                              (exec_mask >> lane & 1) ? expected[lane] : before[lane]);
                  for (const auto &reg : storage) {
                    EXPECT_EQ(reg[0], 0xdeadbeef);
                    EXPECT_EQ(reg[33], 0xdeadbeef);
                  }
                }
        }
  }
}

TEST(Dot2, ExactModifiersRecoverHardwareGoldens) {
  const auto check = [](const auto &cases, auto fn) {
    for (const auto &f : cases)
      for (uint32_t neg = 0; neg < 32; ++neg)
        for (uint32_t swap = 0; swap < 4; ++swap) {
          uint32_t a[32], b[32], c[32], d[32];
          // Undo signs and invert the half permutation so the modified operands
          // reproduce the hardware-captured original dot exactly.
          uint32_t av = f.a ^ (neg & 1 ? 0x8000 : 0) ^ (neg & 8 ? 0x80000000 : 0);
          uint32_t bv = f.b ^ (neg & 2 ? 0x8000 : 0) ^ (neg & 16 ? 0x80000000 : 0);
          uint32_t mode = neg | GOC_DOT_CLAMP;
          if (swap & 1) {
            av = (av << 16) | (av >> 16);
            mode |= GOC_DOT_LO_A_HIGH | GOC_DOT_HI_A_LOW;
          }
          if (swap & 2) {
            bv = (bv << 16) | (bv >> 16);
            mode |= GOC_DOT_LO_B_HIGH | GOC_DOT_HI_B_LOW;
          }
          std::fill(a, a + 32, av);
          std::fill(b, b + 32, bv);
          std::fill(c, c + 32, f.c ^ (neg & 4 ? 0x80000000 : 0));
          auto pa = a, pb = b, pc = c, pd = d;
          ASSERT_EQ(fn(goc_init_cpu_flags() | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                       UINT32_MAX, mode, &pd, &pa, &pb, &pc, nullptr),
                    GOC_SUCCESS);
          for (auto word : d)
            EXPECT_EQ(word, f.expected);
        }
  };
  check(kGfx12DotF16Cases, goc_v_dot2_f32_f16);
  check(kGfx12DotBF16Cases, goc_v_dot2_f32_bf16);
}

TEST(Dot2, SimdSpecialValuesAndEveryExecMask) {
  const uint32_t a16[] = {0x80008000, 0x00010001, 0x7c003c00, 0x7e003c00, 0x3c013c01, 0x7bff7bff};
  const uint32_t b16[] = {0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3bfe3bfe, 0x04000400};
  const uint32_t abf[] = {0x80008000, 0x00010001, 0x7f803f80, 0x7fc03f80, 0x3f813f81, 0x7f7f7f7f};
  const uint32_t bbf[] = {0x3f803f80, 0x3f803f80, 0x3f803f80, 0x3f803f80, 0x3f7e3f7e, 0x00800080};
  for (bool brain : {false, true})
    for (uint32_t mode :
         {0U, GOC_DOT_NEG_C | GOC_DOT_NEG_LO_A | GOC_DOT_LO_B_HIGH | GOC_DOT_HI_A_LOW})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t exec_mask : exec_masks()) {
          auto fn = brain ? goc_v_dot2_f32_bf16 : goc_v_dot2_f32_f16;
          uint32_t a[32], b[32], c[32], ref[32], d[32];
          for (int i = 0; i < 32; ++i) {
            a[i] = (brain ? abf : a16)[i % 6];
            b[i] = (brain ? bbf : b16)[i % 6];
            c[i] = i % 6 == 0 ? 0x80000000 : 0xbf800000;
            ref[i] = d[i] = 0xdeadbeef;
          }
          auto pa = a, pb = b, pc = c, pr = ref, pd = d;
          ASSERT_EQ(fn(GOC_CPU_BASELINE, exec_mask, mode, &pr, &pa, &pb, &pc, nullptr),
                    GOC_SUCCESS);
          ASSERT_EQ(fn(cpu, exec_mask, mode, &pd, &pa, &pb, &pc, nullptr), GOC_SUCCESS);
          for (int i = 0; i < 32; ++i) {
            if (std::isnan(goc::as_float(ref[i]))) {
              EXPECT_TRUE(std::isnan(goc::as_float(d[i])));
            } else {
              EXPECT_EQ(d[i], ref[i]);
            }
          }
        }
}

TEST(Dot2, DppModifiersMasksAndAliases) {
  const uint16_t codes[2][4] = {{0xc000, 0xbc00, 0x3c00, 0x4000}, {0xc000, 0xbf80, 0x3f80, 0x4000}};
  const int values[] = {-2, -1, 1, 2};
  for (unsigned brain = 0; brain < 2; ++brain)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (auto descriptor : goc_test::dpp_modes)
          for (unsigned variant = 0; variant < 1024; ++variant) {
            uint64_t mode = descriptor | (variant & 31) | ((variant & 992) << 1);
            auto masks = (variant == 0 || variant == 1023) ? exec_masks()
                                                           : std::vector<uint32_t>{UINT32_MAX};
            for (auto exec_mask : masks)
              for (unsigned alias = 0; alias < 4; ++alias) {
                uint32_t storage[4][34], expected[4][34];
                int factors[2][32][2];
                for (unsigned reg = 0; reg < 4; ++reg)
                  std::fill_n(storage[reg], 34, 0xdeadbeef);
                for (unsigned lane = 0; lane < 32; ++lane) {
                  for (unsigned reg = 0; reg < 2; ++reg) {
                    unsigned low = (lane + reg) % 4, high = (lane / 4 + 2 * reg) % 4;
                    storage[reg][lane + 1] =
                        codes[brain][low] | (uint32_t(codes[brain][high]) << 16);
                    factors[reg][lane][0] = values[low];
                    factors[reg][lane][1] = values[high];
                  }
                  storage[2][lane + 1] = goc::as_bits(float(int(lane) - 16));
                }
                for (unsigned reg = 0; reg < 4; ++reg)
                  std::copy_n(storage[reg], 34, expected[reg]);
                for (unsigned lane = 0; lane < 32; ++lane) {
                  int source = 0;
                  if (!goc_test::dpp_source(mode, exec_mask, lane, source))
                    continue;
                  int a0 = source < 0 ? 0 : factors[0][source][bool(mode & GOC_DOT_LO_A_HIGH)];
                  int a1 = source < 0 ? 0 : factors[0][source][!bool(mode & GOC_DOT_HI_A_LOW)];
                  int b0 = factors[1][lane][bool(mode & GOC_DOT_LO_B_HIGH)];
                  int b1 = factors[1][lane][!bool(mode & GOC_DOT_HI_B_LOW)];
                  int c = int(lane) - 16;
                  if (mode & GOC_DOT_NEG_LO_A)
                    a0 = -a0;
                  if (mode & GOC_DOT_NEG_HI_A)
                    a1 = -a1;
                  if (mode & GOC_DOT_NEG_LO_B)
                    b0 = -b0;
                  if (mode & GOC_DOT_NEG_HI_B)
                    b1 = -b1;
                  if (mode & GOC_DOT_NEG_C)
                    c = -c;
                  expected[alias][lane + 1] = goc::as_bits(float(a0 * b0 + a1 * b1 + c));
                }
                const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[1] + 1},
                               *c[] = {storage[2] + 1};
                uint32_t *d[] = {storage[alias] + 1};
                auto fn = brain ? goc_v_dot2_f32_bf16 : goc_v_dot2_f32_f16;
                uint32_t exceptions = 0x80000055;
                ASSERT_EQ(fn(cpu | semantics, exec_mask, mode, d, a, b, c, &exceptions),
                          GOC_SUCCESS);
                EXPECT_EQ(exceptions, 0x80000055);
                for (unsigned reg = 0; reg < 4; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    ASSERT_EQ(storage[reg][word], expected[reg][word])
                        << brain << "/" << cpu << "/" << semantics << "/" << mode;
              }
          }
}

TEST(Dot2, DppValidation) {
  for (auto fn : {goc_v_dot2_f32_f16, goc_v_dot2_f32_bf16})
    for (auto descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(fn(0, 0, descriptor, nullptr, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (auto invalid : {1ULL << 36, 1ULL << 5})
        EXPECT_EQ(fn(0, 0, descriptor | invalid, nullptr, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
    }
}

// RX 9070 finite inputs with exact products and sums. Exact semantics retain
// zero signs; loose semantics may produce either sign of a zero result.
TEST(Dot2, DppHardwareCorpus) {
  const uint32_t values[2][8] = {{0x3c00bc00, 0x4000c000, 0x4200c200, 0x4400c400, 0x00003c00,
                                  0x3800b800, 0x4500c500, 0x4600c600},
                                 {0x3f80bf80, 0x4000c000, 0x4040c040, 0x4080c080, 0x00003f80,
                                  0x3f00bf00, 0x40a0c0a0, 0x40c0c0c0}};
  const uint32_t signs[] = {0, 1, 2, 4, 8, 16, 31, 64, 95};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
      uint64_t hash = goc_test::capture_hash_seed;
      for (auto exec_mask : masks)
        for (unsigned brain = 0; brain < 2; ++brain)
          for (auto descriptor : goc_test::dpp_modes)
            for (unsigned variant = 0; variant < 25; ++variant) {
              uint32_t mode = variant < 9 ? signs[variant] : ((variant - 9) << 7) | 85;
              uint32_t av[32], bv[32], cv[32], output[32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                av[lane] = values[brain][lane % 8];
                bv[lane] = values[brain][(lane + 3) % 8];
                cv[lane] = goc::as_bits(float(int(lane) - 16));
                output[lane] = 0xdead0000u + lane;
              }
              const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
              uint32_t *d[] = {output};
              auto fn = brain ? goc_v_dot2_f32_bf16 : goc_v_dot2_f32_f16;
              ASSERT_EQ(fn(cpu | semantics, exec_mask, descriptor | mode, d, a, b, c, nullptr),
                        GOC_SUCCESS);
              for (auto word : output) {
                if (semantics == GOC_SEMANTICS_LOOSE && !(word & 0x7fffffff))
                  word = 0;
                hash = goc_test::capture_hash_word(hash, word);
              }
            }
      EXPECT_EQ(hash, 0xd46ee8a52b583d85ULL) << cpu << "/" << semantics;
    }
}

TEST(Dot2, NoExceptionHardwareCorpora) {
  // RX 9070 gfx1201, capture_arithmetic_exceptions.py dot2_f32_f16/bf16 16:
  // all 524,288 flag reads are zero. Both FP16_OVFL settings, sixteen modifiers,
  // 4,096 edge triples and 4,096 random triples, with distinct packed halves.
  const uint32_t accumulators[] = {0,          0x80000000, 1,          0x80000001,
                                   0x7fffff,   0x800000,   0x3f800000, 0xbf800000,
                                   0x40000000, 0x7f7fffff, 0x7effffff, 0x7f800000,
                                   0xff800000, 0x7f800001, 0xffc00003, 0x1000000};
  for (bool brain : {false, true}) {
    unsigned fraction = brain ? 7 : 10, bias = brain ? 127 : 15;
    uint32_t infinity = (2 * bias + 1) << fraction, one = bias << fraction;
    const uint32_t values[] = {0,
                               0x8000,
                               1,
                               0x8001,
                               (1U << fraction) - 1,
                               1U << fraction,
                               one,
                               0x8000 | one,
                               one + (1U << fraction),
                               infinity - 1,
                               infinity - 1 - (1U << fraction),
                               infinity,
                               0x8000 | infinity,
                               infinity | 1,
                               0x8000 | infinity | (1U << (fraction - 1)) | 3,
                               2U << fraction};
    auto fn = brain ? goc_v_dot2_f32_bf16 : goc_v_dot2_f32_f16;
    for (unsigned variant = 0; variant < 32; ++variant) {
      uint32_t mode =
          (variant & 1 ? GOC_DOT_NEG_LO_A | GOC_DOT_NEG_C : 0) |
          (variant & 2 ? GOC_DOT_NEG_HI_B : 0) | (variant & 4 ? GOC_DOT_CLAMP : 0) |
          (variant & 8 ? GOC_DOT_LO_A_HIGH | GOC_DOT_LO_B_HIGH | GOC_DOT_HI_A_LOW | GOC_DOT_HI_B_LOW
                       : 0);
      for (unsigned i = 0; i < 8192; ++i) {
        uint32_t words[4][32] = {};
        uint64_t state = uint64_t(i) * 0x9e3779b97f4a7c15ULL;
        for (unsigned operand = 0; operand < 3; ++operand) {
          state ^= state >> 12;
          state ^= state << 25;
          state ^= state >> 27;
          uint32_t random = uint32_t(state * 0x2545f4914f6cdd1dULL);
          unsigned index = (i >> (8 - 4 * operand)) & 15;
          if (operand == 2)
            words[operand][0] = i < 4096 ? accumulators[index] : random;
          else {
            uint32_t lo = i < 4096 ? values[index] : uint16_t(random);
            uint32_t hi = i < 4096 ? values[(index + 5) % 16] : ((lo * 0x9e37 + 0x1234) & 65535);
            words[operand][0] = lo | (hi << 16);
          }
        }
        const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
        uint32_t *d[] = {words[3]}, exceptions = 0x80000055;
        uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                         (variant & 16 ? GOC_FP16_OVFL : 0) | (i % (goc_init_cpu_flags() + 1));
        ASSERT_EQ(fn(flags, 1, mode, d, a, b, c, &exceptions), GOC_SUCCESS);
        EXPECT_EQ(exceptions, 0x80000055);
      }
    }
  }
}

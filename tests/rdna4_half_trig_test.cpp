// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_reference.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn = decltype(&goc_rdna4_v_sin_f16);
const Fn functions[] = {goc_rdna4_v_sin_f16, goc_rdna4_v_cos_f16};
const Fn wide_functions[] = {goc_rdna4_v_sin_f32, goc_rdna4_v_cos_f32};

uint32_t modifiers(int mode) {
  return (mode & 1 ? GOC_ALU_NEG_A : 0) | (mode & 2 ? GOC_ALU_ABS_A : 0) |
         (uint32_t((mode >> 2) & 3) << 6) | (mode & 16 ? GOC_ALU_CLAMP : 0) |
         (mode & 32 ? GOC_ALU_HIGH_A : 0) | (mode & 64 ? GOC_ALU_HIGH_D : 0);
}

// Round the independently tested FP32 exact API's results using the numeric
// half reference, preserving NaN signs/payloads explicitly.
std::vector<uint16_t> reference_table(int op) {
  std::vector<uint16_t> table(65536);
  for (unsigned start = 0; start < 65536; start += 32) {
    uint32_t a[32], d[32];
    const uint32_t *ap[] = {a};
    uint32_t *dp[] = {d};
    for (unsigned lane = 0; lane < 32; ++lane) {
      uint16_t half = uint16_t(start + lane);
      a[lane] = (half & 0x7fff) > 0x7c00
                    ? (uint32_t(half & 0x8000) << 16) | 0x7f800000 | (uint32_t(half & 1023) << 13)
                    : goc::as_bits(float(goc_test::half_value(half)));
    }
    EXPECT_EQ(wide_functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                                 0, dp, ap),
              GOC_SUCCESS);
    for (unsigned lane = 0; lane < 32; ++lane)
      table[start + lane] =
          (d[lane] & 0x7fffffff) > 0x7f800000
              ? uint16_t(((d[lane] >> 16) & 0x8000) | 0x7c00 | ((d[lane] >> 13) & 1023))
              : goc_test::half_bits(double(goc::as_float(d[lane])));
  }
  return table;
}

uint16_t reference(uint16_t value, int mode) {
  double result = goc_test::half_value(value);
  const double scales[] = {1, 2, 4, 0.5};
  if (mode & 12) {
    if (std::abs(result) < 0x1p-14)
      result = 0;
    else
      result *= scales[(mode >> 2) & 3];
    if (!std::isnan(result)) {
      value = goc_test::half_bits(result);
      if ((value & 0x7fff) < 0x400)
        value = 0;
    }
  }
  if (mode & 16) {
    if (std::isnan(result) || (value & 0x8000))
      return 0;
    if (value > 0x3c00)
      return 0x3c00;
  }
  return value;
}

void expect_near(uint16_t actual, uint16_t expected) {
  unsigned magnitude = expected & 0x7fff;
  if (!magnitude || magnitude >= 0x7c00)
    EXPECT_EQ(actual, expected);
  else {
    EXPECT_EQ(actual & 0x8000, expected & 0x8000);
    EXPECT_LE(std::abs(int(actual & 0x7fff) - int(magnitude)), 1);
  }
}

} // namespace

TEST(HalfTrig, EveryEncodingMatchesRocjitsuDigestAndLoosePaths) {
  // FNV-1a over uint16_t results in ascending input-encoding order. Generated
  // directly from rocjitsu util/amdgpu_trig.h with compiler _Float16 conversion,
  // independently of GoC's C++17 adaptation and half conversion implementation.
  const uint64_t digests[] = {0xb6245e65a23f99f9ULL, 0xc6c3112c531e4391ULL};
  for (int op = 0; op < 2; ++op) {
    const auto table = reference_table(op);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true}) {
        uint64_t digest = goc_test::capture_hash_seed;
        for (unsigned start = 0; start < 65536; start += 32) {
          uint32_t a[32], exact[32], loose[32];
          const uint32_t *ap[] = {a};
          uint32_t *ep[] = {exact}, *lp[] = {loose};
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = 0xbadc0000 | (start + lane);
            exact[lane] = loose[lane] = 0xdeadbeef;
          }
          uint64_t flags = cpu | (saturate ? GOC_FP16_OVFL : 0);
          ASSERT_EQ(functions[op](flags | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                  UINT32_MAX, 0, ep, ap),
                    GOC_SUCCESS);
          ASSERT_EQ(functions[op](flags, UINT32_MAX, 0, lp, ap), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            ASSERT_EQ(exact[lane], 0xdead0000u | table[start + lane]);
            EXPECT_EQ(loose[lane] >> 16, 0xdeadu);
            expect_near(uint16_t(loose[lane]), table[start + lane]);
            digest = goc_test::capture_hash_word(digest, uint16_t(exact[lane]));
            if ((a[lane] & 0x7fff) < 0x7c00) {
              double phase = std::remainder(goc_test::half_value(uint16_t(a[lane])), 1.0);
              double angle = phase * 6.283185307179586476925286766559;
              EXPECT_NEAR(goc_test::half_value(uint16_t(loose[lane])),
                          op ? std::cos(angle) : std::sin(angle), 0.0005);
            }
          }
        }
        EXPECT_EQ(digest, digests[op]) << op << "/" << cpu << "/" << saturate;
      }
  }
}

TEST(HalfTrig, AllModifiersRandomInputsAndBoundaries) {
  const uint16_t edges[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x7ff,  0x800,
                            0x1000, 0x1800, 0x3000, 0x3400, 0x3800, 0x3c00, 0x7bff, 0xfbff,
                            0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7eab, 0xfeab};
  for (int op = 0; op < 2; ++op) {
    const auto table = reference_table(op);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < 128; ++mode) {
        std::mt19937 random(815);
        for (int start = 0; start < 4096; start += 32) {
          uint32_t a[32], d[32], exact[32];
          const uint32_t *ap[] = {a};
          uint32_t *dp[] = {d}, *ep[] = {exact};
          uint16_t expected[32];
          for (int lane = 0; lane < 32; ++lane) {
            uint32_t lo = start == 0 ? edges[lane % 22] : uint16_t(random());
            uint32_t hi = start == 0 ? edges[(lane + 7) % 22] : uint16_t(random());
            a[lane] = lo | (hi << 16);
            d[lane] = exact[lane] = 0xfacecafe;
            uint16_t selected = uint16_t(mode & 32 ? hi : lo);
            if (mode & 2)
              selected &= 0x7fff;
            if (mode & 1)
              selected ^= 0x8000;
            expected[lane] = reference(table[selected], mode);
          }
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, modifiers(mode), dp, ap), GOC_SUCCESS);
          ASSERT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                  UINT32_MAX, modifiers(mode), ep, ap),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane) {
            int shift = mode & 64 ? 16 : 0;
            uint32_t keep = mode & 64 ? 0x0000ffff : 0xffff0000;
            ASSERT_EQ(exact[lane], (0xfacecafe & keep) | (uint32_t(expected[lane]) << shift));
            EXPECT_EQ(d[lane] & keep, 0xfacecafe & keep);
            expect_near(uint16_t(d[lane] >> shift), expected[lane]);
          }
        }
      }
  }
}

TEST(HalfTrig, OutputScalingFlushBoundaries) {
  // Columns are input, no OMOD, x2, x4 and /2. Trig rounds to half before
  // scaling, so a tiny result cannot be rescued by multiplying it by four.
  const uint16_t cases[][5] = {{0x00a2, 0x03fa, 0, 0, 0},
                               {0x00a3, 0x0400, 0x0800, 0x0c00, 0},
                               {0x0145, 0x07fa, 0x0bfa, 0x0ffa, 0},
                               {0x0146, 0x0800, 0x0c00, 0x1000, 0x0400}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool exact : {false, true})
      for (int scale = 0; scale < 4; ++scale)
        for (bool negative : {false, true}) {
          uint32_t a[32], d[32] = {};
          const uint32_t *ap[] = {a};
          uint32_t *dp[] = {d};
          for (int lane = 0; lane < 32; ++lane)
            a[lane] = cases[lane % 4][0];
          uint32_t mode = uint32_t(scale) << 6;
          if (negative)
            mode |= GOC_ALU_NEG_A;
          ASSERT_EQ(goc_rdna4_v_sin_f16(cpu | (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL : 0),
                                        UINT32_MAX, mode, dp, ap),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane) {
            uint32_t expected = cases[lane % 4][scale + 1];
            if (negative && expected)
              expected |= 0x8000;
            EXPECT_EQ(d[lane], expected);
          }
        }
}

TEST(HalfTrig, MasksAliasesAndHalfPreservation) {
  for (int op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < 128; ++mode) {
        uint32_t source[32], full[32];
        std::mt19937 random(720);
        for (int lane = 0; lane < 32; ++lane)
          source[lane] = full[lane] = random();
        const uint32_t *ap[] = {source};
        uint32_t *fp[] = {full};
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, modifiers(mode), fp, ap), GOC_SUCCESS);
        for (uint32_t exec_mask : rdna4_exec_masks())
          for (bool alias : {false, true}) {
            uint32_t a[32], d[32];
            std::copy_n(source, 32, a);
            std::copy_n(source, 32, d);
            const uint32_t *input[] = {a};
            uint32_t *output[] = {alias ? a : d};
            ASSERT_EQ(functions[op](cpu, exec_mask, modifiers(mode), output, input), GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              ASSERT_EQ(output[0][lane], (exec_mask >> lane) & 1 ? full[lane] : source[lane]);
          }
      }
}

TEST(HalfTrig, ExactPreservesHostEnvironmentWithEveryModifier) {
  uint32_t source[32], expected[2][128][32];
  for (int lane = 0; lane < 32; ++lane)
    source[lane] = (uint32_t(0x7c00 + lane) << 16) | uint32_t(lane * 457);
  const uint32_t *ap[] = {source};
  for (int op = 0; op < 2; ++op)
    for (int mode = 0; mode < 128; ++mode) {
      std::fill_n(expected[op][mode], 32, 0xdeadbeef);
      uint32_t *dp[] = {expected[op][mode]};
      ASSERT_EQ(functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, modifiers(mode), dp, ap),
                GOC_SUCCESS);
    }
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
      for (int op = 0; op < 2; ++op)
        for (int mode = 0; mode < 128; ++mode) {
          uint32_t d[32];
          std::fill_n(d, 32, 0xdeadbeef);
          uint32_t *dp[] = {d};
          EXPECT_EQ(functions[op](goc_init_cpu_flags() | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                      GOC_SEMANTICS_STRICT,
                                  UINT32_MAX, modifiers(mode), dp, ap),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[lane], expected[op][mode][lane]);
        }
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), before);
#endif
    }
}

TEST(HalfTrig, InvalidFlagsAndReservedSemanticsPreserveDestination) {
  for (Fn fn : functions) {
    uint32_t a[32] = {}, d[32];
    std::fill_n(d, 32, 0xdeadbeef);
    const uint32_t *ap[] = {a};
    uint32_t *dp[] = {d};
    for (int bit = 0; bit < 32; ++bit) {
      uint32_t mode = uint32_t(1) << bit;
      if (mode & (GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP |
                  GOC_ALU_HIGH_A | GOC_ALU_HIGH_D))
        continue;
      for (uint32_t exec_mask : {0U, UINT32_MAX})
        EXPECT_EQ(fn(0, exec_mask, mode, dp, ap), GOC_ERROR_INVALID_FLAGS);
    }
    EXPECT_EQ(fn(1ULL << 63, 0, 0, dp, ap), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT, 0, 0, dp, ap),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_MASK, UINT32_MAX, 0, dp, ap), GOC_SUCCESS);
  }
}

TEST(HalfTrig, DppModifiersMasksAliasesAndGuards) {
  std::mt19937 random(987173);
  uint32_t initial[2][34];
  for (auto &reg : initial)
    for (auto &word : reg)
      word = random();
  for (auto fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (int m = 0; m < 128; ++m)
          for (auto descriptor : goc_test::dpp_modes) {
            auto masks =
                (m == 0 || m == 128 - 1) ? rdna4_exec_masks() : std::vector<uint32_t>{UINT32_MAX};
            for (auto exec_mask : masks)
              for (unsigned target = 0; target < 2; ++target) {
                uint32_t words[2][34], expected[2][34], permuted[32] = {};
                for (unsigned reg = 0; reg < 2; ++reg) {
                  std::copy_n(initial[reg], 34, words[reg]);
                  std::copy_n(initial[reg], 34, expected[reg]);
                }
                uint32_t reference_exec_mask = 0;
                for (unsigned lane = 0; lane < 32; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(descriptor, exec_mask, lane, source)) {
                    reference_exec_mask |= uint32_t(1) << lane;
                    permuted[lane] = source < 0 ? 0 : initial[0][source + 1];
                  }
                }
                const uint32_t *a[] = {words[0] + 1}, *reference_a[] = {permuted};
                uint32_t *d[] = {words[target] + 1}, *reference_d[] = {expected[target] + 1};
                ASSERT_EQ(fn(cpu | semantics, reference_exec_mask, modifiers(m), reference_d,
                             reference_a),
                          GOC_SUCCESS);
                ASSERT_EQ(fn(cpu | semantics, exec_mask, descriptor | modifiers(m), d, a),
                          GOC_SUCCESS);
                for (unsigned reg = 0; reg < 2; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    ASSERT_EQ(words[reg][word], expected[reg][word])
                        << cpu << "/" << semantics << "/" << m << "/" << descriptor << "/"
                        << exec_mask;
              }
          }
}

TEST(HalfTrig, DppValidation) {
  for (auto fn : functions)
    for (auto descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(fn(0, 0, descriptor, nullptr, nullptr), GOC_SUCCESS);
      for (auto invalid : {1ULL << 36, 1ULL << 1})
        EXPECT_EQ(fn(0, 0, descriptor | invalid, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    }
}

// RX 9070 quarter-turn inputs: both operations, all modifiers, seven DPP
// descriptors and eight EXEC masks. Includes zeros and large integral turns.
TEST(HalfTrig, DppHardwareCorpus) {
  const uint32_t values[] = {0x00008000, 0x3400b400, 0x3800b800, 0x3a00ba00,
                             0x3c00bc00, 0x7bfffbff, 0x4000c000, 0x4200c200};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
      uint64_t hash = goc_test::capture_hash_seed;
      for (auto exec_mask : masks)
        for (auto fn : functions)
          for (int m = 0; m < 128; ++m)
            for (auto descriptor : goc_test::dpp_modes) {
              uint32_t av[32], output[32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                av[lane] = values[lane % 8];
                output[lane] = 0xdead0000u + lane;
              }
              const uint32_t *a[] = {av};
              uint32_t *d[] = {output};
              ASSERT_EQ(fn(cpu | semantics, exec_mask, descriptor | modifiers(m), d, a),
                        GOC_SUCCESS);
              for (auto word : output)
                hash = goc_test::capture_hash_word(hash, word);
            }
      EXPECT_EQ(hash, 0x622b2df3f3bc9b25ULL) << cpu << "/" << semantics;
    }
}

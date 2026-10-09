// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_reference.h"
#include "rdna4_mixed_fma_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <ios>
#include <random>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn = decltype(&goc_rdna4_v_fma_mix_f32);
const Fn functions[] = {goc_rdna4_v_fma_mix_f32, goc_rdna4_v_fma_mixlo_f16,
                        goc_rdna4_v_fma_mixhi_f16};
const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000,
                           0x3f000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x3f7fffff, 0x33800000,
                           0x477fe000, 0x477ff000, 0x47800000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
                           0xff800000, 0x7f800012, 0x7fc12345, 0xffcabcde, 0x3c013bff, 0x7bff0400};

bool equal(int op, uint32_t got, uint32_t want, bool exact) {
  if (got == want)
    return true;
  if (!exact && !op)
    return (got & 0x7fffffff) > 0x7f800000 && (want & 0x7fffffff) > 0x7f800000;
  return false;
}

::testing::AssertionResult check(int op, uint64_t flags, uint32_t mode,
                                 const uint32_t input[4][32]) {
  uint32_t out[32];
  std::copy_n(input[3], 32, out);
  const uint32_t *a[] = {input[0]}, *b[] = {input[1]}, *c[] = {input[2]};
  uint32_t *d[] = {out};
  int status = functions[op](flags, UINT32_MAX, mode, d, a, b, c);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t want =
        goc_test::mixed_fma_reference::evaluate(op, input[0][lane], input[1][lane], input[2][lane],
                                                input[3][lane], mode, flags & GOC_FP16_OVFL);
    if (!equal(op, out[lane], want,
               op && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL))
      return ::testing::AssertionFailure()
             << op << '/' << flags << '/' << mode << '/' << lane << ": " << std::hex
             << input[0][lane] << ' ' << input[1][lane] << ' ' << input[2][lane] << " got "
             << out[lane] << " expected " << want;
  }
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(MixedFma, DppAllModifiers) {
  for (int op = 0; op < 3; ++op)
    for (unsigned index = 0; index < 8192; ++index)
      for (uint64_t descriptor : goc_test::dpp_modes) {
        uint32_t input[4][32], want[32];
        for (unsigned reg = 0; reg < 4; ++reg)
          for (unsigned lane = 0; lane < 32; ++lane)
            input[reg][lane] = values[(lane * 7 + reg * 5) % 24];
        uint32_t mode = goc_test::mixed_fma_reference::mode(index);
        bool saturate = index & 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          int source;
          want[lane] = input[3][lane];
          if (goc_test::dpp_source(descriptor, UINT32_MAX, lane, source))
            want[lane] = goc_test::mixed_fma_reference::evaluate(
                op, source < 0 ? 0 : input[0][source], input[1][lane], input[2][lane],
                input[3][lane], mode, saturate);
        }
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t output[32];
          std::copy_n(input[3], 32, output);
          uint32_t *d[] = {output};
          const uint32_t *a[] = {input[0]}, *b[] = {input[1]}, *c[] = {input[2]};
          ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                  descriptor | mode, d, a, b, c),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_TRUE(equal(op, output[lane], want[lane], false))
                << op << "/" << index << "/" << descriptor << "/" << cpu << "/" << lane;
        }
      }
}

TEST(MixedFma, DppHardwareCorpus) {
  // RX 9070/gfx1201, MODE 0xf0: all input format/half combinations with
  // ABS/NEG and CLAMP, eight EXEC masks, and seven DPP descriptors.
  const uint32_t inputs[] = {0x3c003800, 0xbc00b800, 0x40003c00, 0xc000bc00,
                             0x3f800000, 0xbf800000, 0,          0x80000000};
  const uint32_t masks[] = {UINT32_MAX, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t mask : masks)
      for (int op = 0; op < 3; ++op)
        for (unsigned variant = 0; variant < 128; ++variant)
          for (uint64_t descriptor : goc_test::dpp_modes) {
            uint32_t words[4][32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              words[0][lane] = inputs[lane % 8];
              words[1][lane] = inputs[(lane + 3) % 8];
              words[2][lane] = inputs[(lane + 5) % 8];
              words[3][lane] = 0xdead0000u + lane;
            }
            uint32_t mode = ((variant & 7) << 3) | ((variant >> 3) & 7) |
                            ((variant & 64) ? GOC_ALU_CLAMP : 0) | (((variant >> 3) & 7) << 9) |
                            ((variant & 7) << 13);
            const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
            uint32_t *d[] = {words[3]};
            ASSERT_EQ(functions[op](cpu, mask, descriptor | mode, d, a, b, c), GOC_SUCCESS);
            for (uint32_t word : words[3])
              hash = goc_test::capture_hash_word(hash, word);
          }
    EXPECT_EQ(hash, UINT64_C(0x67b9a4c5cebf1238));
  }
}

TEST(MixedFma, DppMasksAliasesAndGuards) {
  for (int op = 0; op < 3; ++op)
    for (unsigned index : {0u, 1u, 85u, 1023u, 4096u, 8191u})
      for (uint64_t descriptor : goc_test::dpp_modes)
        for (uint32_t mask : rdna4_exec_masks())
          for (bool shared : {false, true})
            for (unsigned target = 0; target < 4; ++target) {
              uint32_t initial[4][34], want[32];
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned lane = 0; lane < 34; ++lane)
                  initial[reg][lane] = values[(lane * 7 + reg * 5) % 24];
              uint32_t mode = goc_test::mixed_fma_reference::mode(index);
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source;
                want[lane] = initial[target][lane + 1];
                if (goc_test::dpp_source(descriptor, mask, lane, source))
                  want[lane] = goc_test::mixed_fma_reference::evaluate(
                      op, source < 0 ? 0 : initial[0][source + 1],
                      initial[shared ? 0 : 1][lane + 1], initial[shared ? 0 : 2][lane + 1],
                      want[lane], mode, true);
              }
              for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
                uint32_t words[4][34];
                std::memcpy(words, initial, sizeof(words));
                uint32_t *d[] = {words[target] + 1};
                const uint32_t *a[] = {words[0] + 1}, *b[] = {words[shared ? 0 : 1] + 1},
                               *c[] = {words[shared ? 0 : 2] + 1};
                uint64_t flags = cpu | GOC_FP16_OVFL;
                if (op && shared)
                  flags |= GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
                ASSERT_EQ(functions[op](flags, mask, descriptor | mode, d, a, b, c), GOC_SUCCESS);
                for (unsigned reg = 0; reg < 4; ++reg)
                  for (unsigned lane = 0; lane < 34; ++lane) {
                    uint32_t expected = reg == target && lane > 0 && lane < 33 ? want[lane - 1]
                                                                               : initial[reg][lane];
                    ASSERT_TRUE(equal(op, words[reg][lane], expected, op && shared));
                  }
              }
            }
}

TEST(MixedFma, BoundaryTriplesAndRandomWords) {
  for (int op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      std::mt19937 random(4671);
      for (unsigned base = 0; base < 13824 + 32768; base += 32) {
        uint32_t words[4][32];
        for (int lane = 0; lane < 32; ++lane) {
          unsigned index = base + lane;
          for (int reg = 0; reg < 3; ++reg) {
            words[reg][lane] = base < 13824 ? values[index % 24] : random();
            index /= 24;
          }
          words[3][lane] = random();
        }
        uint32_t mode = (((base / 32) & 7) << 13) | ((base & 64) ? GOC_ALU_CLAMP : 0);
        uint64_t flags = cpu | ((base & 32) ? GOC_FP16_OVFL : 0);
        ASSERT_TRUE(check(op, flags, mode, words));
        if (op && cpu == 0) {
          ASSERT_TRUE(
              check(op, flags | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mode, words));
        }
      }
    }
}

TEST(MixedFma, All8192ModifierCombinations) {
  for (int op = 0; op < 3; ++op)
    for (unsigned index = 0; index < 8192; ++index)
      for (bool saturate : {false, true}) {
        uint32_t words[4][32];
        for (int reg = 0; reg < 4; ++reg)
          for (int lane = 0; lane < 32; ++lane)
            words[reg][lane] = values[(lane * 7 + reg * 5) % 24];
        uint32_t mode = goc_test::mixed_fma_reference::mode(index);
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          ASSERT_TRUE(check(op, cpu | (saturate ? GOC_FP16_OVFL : 0), mode, words));
        if (op) {
          ASSERT_TRUE(check(op,
                            GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                                (saturate ? GOC_FP16_OVFL : 0),
                            mode, words));
        }
      }
}

TEST(MixedFma, UnsupportedNonstrictSemanticsFallBackToLoose) {
  for (int op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[4][32];
      for (int reg = 0; reg < 4; ++reg)
        for (int lane = 0; lane < 32; ++lane)
          words[reg][lane] = values[(lane + reg * 5) % 24];
      for (uint64_t semantics : {UINT64_C(2) << 16, UINT64_C(3) << 16})
        EXPECT_TRUE(check(op, cpu | semantics, 0, words));
      if (!op) {
        EXPECT_TRUE(check(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, 0, words));
      }
    }
}

TEST(MixedFma, EveryHalfEncodingInEachSource) {
  for (int op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int source = 0; source < 3; ++source)
        for (unsigned base = 0; base < 65536; base += 32) {
          uint32_t words[4][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            words[0][lane] = 0x3f800001;
            words[1][lane] = 0x3f7ffffe;
            words[2][lane] = 0x33800000;
            unsigned value = base + lane;
            words[source][lane] = value | ((65535 - value) << 16);
            words[3][lane] = 0xfacecafe;
          }
          uint32_t mode = (GOC_MIX_F16_A << source) | ((base & 32) ? GOC_ALU_HIGH_A << source : 0);
          ASSERT_TRUE(check(op, cpu, mode, words));
        }
}

TEST(MixedFma, CapturedAndSingleRoundingWitnesses) {
  // NaN/invalid-product priorities and the 0x340c product are gfx1100/gfx1201 captures from
  // rocjitsu valu_fp_mode_test.cpp. Remaining finite cases distinguish direct
  // FP16 rounding from intermediate FP32 rounding and overflowing products.
  const uint32_t cases[][4] = {
      {0, 0x7f800000, 0x7fe9a000, 0xfe00},          {0x7f800000, 0, 0x7fe9a000, 0xfe00},
      {0x7fc12000, 0x3f800000, 0x7fe9a000, 0x7e09}, {0x3b333333, 0x42b90000, 0, 0x340c},
      {0x477ff000, 0x3f800000, 1, 0x7c00},          {0x477ff000, 0x3f800000, 0x80000001, 0x7bff},
      {0xc77ff000, 0x3f800000, 1, 0xfbff},          {0xc77ff000, 0x3f800000, 0x80000001, 0xfc00},
      {0x3f801000, 0x3f800001, 0, 0x3c01},          {0x3f803000, 0x3f7fffff, 0, 0x3c01},
      {0x3f800000, 0x3f801000, 1, 0x3c01},          {0x3f800000, 0x3f803000, 0x80000001, 0x3c01},
      {0x7f7fffff, 0x40000000, 0xff7fffff, 0x7c00}, {0x80000000, 0x3f800000, 0x80000000, 0x8000}};
  for (int op : {1, 2})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t sem :
           {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
        for (const auto &item : cases) {
          uint32_t words[4][32];
          for (int reg = 0; reg < 3; ++reg)
            std::fill_n(words[reg], 32, item[reg]);
          std::fill_n(words[3], 32, 0xfacecafe);
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          ASSERT_EQ(functions[op](cpu | sem, UINT32_MAX, 0, d, a, b, c), GOC_SUCCESS);
          uint32_t want = op == 1 ? 0xface0000 | item[3] : 0xcafe | (item[3] << 16);
          for (uint32_t value : words[3])
            EXPECT_EQ(value, want);
        }
}

TEST(MixedFma, EveryPositiveHalfMidpointWithTinyPerturbations) {
  for (int op : {1, 2})
    for (bool negative : {false, true})
      for (bool downward : {false, true})
        for (unsigned base = 0; base < 0x7bff; base += 32) {
          uint32_t words[4][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint16_t low = uint16_t(std::min(base + lane, 0x7bfeu));
            float midpoint =
                float((goc_test::half_value(low) + goc_test::half_value(low + 1)) * 0.5);
            std::memcpy(&words[0][lane], &midpoint, sizeof(midpoint));
            if (negative)
              words[0][lane] ^= 0x80000000;
            words[1][lane] = 0x3f800000;
            words[2][lane] = downward ? 0x80000001 : 1;
            words[3][lane] = 0xfacecafe;
          }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            ASSERT_TRUE(check(op, cpu, 0, words));
          ASSERT_TRUE(check(op, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, words));
        }
}

TEST(MixedFma, CapturedFp32FusedCancellation) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t words[4][32];
    const uint32_t input[] = {0x3f800001, 0x3f7ffffe, 0xbf800000};
    for (int reg = 0; reg < 3; ++reg)
      std::fill_n(words[reg], 32, input[reg]);
    const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
    uint32_t *d[] = {words[3]};
    ASSERT_EQ(functions[0](cpu, UINT32_MAX, 0, d, a, b, c), GOC_SUCCESS);
    for (uint32_t value : words[3])
      EXPECT_EQ(value, 0xa8800000);
  }
}

TEST(MixedFma, MasksUnalignedBuffersAndAllWholeAliases) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  const uint32_t modes[] = {0, goc_test::mixed_fma_reference::mode(8191),
                            GOC_MIX_F16_A | GOC_ALU_HIGH_A | GOC_ALU_NEG_B,
                            GOC_MIX_F16_B | GOC_MIX_F16_C | GOC_ALU_ABS_A | GOC_ALU_HIGH_C};
  for (int op = 0; op < 3; ++op)
    for (unsigned semantics = 0; semantics < (op ? 2u : 1u); ++semantics)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t mode : modes)
          for (uint32_t mask : rdna4_exec_masks())
            for (const auto &layout : layouts)
              for (int target = 0; target < 4; ++target) {
                uint32_t words[4][34], expected[4][34];
                for (int reg = 0; reg < 4; ++reg)
                  for (int lane = 0; lane < 34; ++lane)
                    words[reg][lane] = expected[reg][lane] = values[(lane * 5 + reg * 7) % 24];
                for (int lane = 0; lane < 32; ++lane)
                  if ((mask >> lane) & 1)
                    expected[target][lane + 1] = goc_test::mixed_fma_reference::evaluate(
                        op, words[layout[0]][lane + 1], words[layout[1]][lane + 1],
                        words[layout[2]][lane + 1], words[target][lane + 1], mode, true);
                const uint32_t *a[] = {words[layout[0]] + 1}, *b[] = {words[layout[1]] + 1},
                               *c[] = {words[layout[2]] + 1};
                uint32_t *d[] = {words[target] + 1};
                ASSERT_EQ(
                    functions[op](
                        cpu | GOC_FP16_OVFL |
                            (semantics ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0),
                        mask, mode, d, a, b, c),
                    GOC_SUCCESS);
                for (int reg = 0; reg < 4; ++reg)
                  for (int lane = 0; lane < 34; ++lane) {
                    bool active =
                        reg == target && lane > 0 && lane <= 32 && ((mask >> (lane - 1)) & 1);
                    ASSERT_TRUE(active ? equal(op, words[reg][lane], expected[reg][lane], false)
                                       : words[reg][lane] == expected[reg][lane]);
                  }
              }
}

TEST(MixedFma, ExactPreservesHostStateAndErrorsPrecedeEmptyMask) {
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
      for (int op = 0; op < 3; ++op)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t words[4][32];
          for (int reg = 0; reg < 4; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              words[reg][lane] = values[(lane + reg * 3) % 24];
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          uint32_t original[32];
          std::copy_n(words[3], 32, original);
          uint32_t known = goc_test::mixed_fma_reference::mode(8191);
          for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
            for (int bit = 0; bit < 32; ++bit)
              if (!(known & (uint32_t(1) << bit))) {
                EXPECT_EQ(functions[op](cpu, mask, uint32_t(1) << bit, d, a, b, c),
                          GOC_ERROR_INVALID_FLAGS);
              }
            EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), mask, 0, d, a, b, c),
                      GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(functions[op](cpu | (UINT64_C(2) << 16) | GOC_SEMANTICS_STRICT, mask, 0, d, a,
                                    b, c),
                      GOC_ERROR_UNSUPPORTED_SEMANTICS);
            if (!op) {
              EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                      mask, 0, d, a, b, c),
                        GOC_ERROR_UNSUPPORTED_SEMANTICS);
            }
          }
          EXPECT_TRUE(std::equal(words[3], words[3] + 32, original));
          EXPECT_EQ(functions[op](cpu, UINT32_C(0), known, d, a, b, c), GOC_SUCCESS);
          EXPECT_TRUE(std::equal(words[3], words[3] + 32, original));
          if (op) {
            for (uint32_t mode : {UINT32_C(0), known, GOC_MIX_F16_A | GOC_ALU_HIGH_A})
              EXPECT_TRUE(check(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                mode, words));
            std::fill_n(words[0], 32, 0x3f800000);
            std::fill_n(words[1], 32, 0x3f801000);
            std::fill_n(words[2], 32, 1);
            EXPECT_TRUE(
                check(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, words));
          }
        }
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), before);
#endif
    }
}

TEST(MixedFma, AlternatingDestinationHalvesPreservePriorWrites) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t a[32], b[32], c[32], output[32], expected[32];
    std::fill_n(a, 32, 0x3f800000); // 1
    std::fill_n(b, 32, 0x40000000); // 2
    std::fill_n(c, 32, 0x40400000); // 3
    std::fill_n(output, 32, 0xdeadbeef);
    std::copy_n(output, 32, expected);
    const uint32_t *pa = a, *pb = b, *pc = c;
    uint32_t *pd = output;
    for (unsigned step = 0; step < 8; ++step) {
      bool high = step & 1;
      uint32_t mask = step & 2 ? 0xaaaaaaaa : 0x55555555;
      uint32_t mode = step & 4 ? GOC_ALU_NEG_A : 0;
      uint32_t half = mode ? 0x3c00 : 0x4500; // -1*2+3 = 1; 1*2+3 = 5.
      unsigned shift = high ? 16 : 0;
      for (unsigned lane = 0; lane < 32; ++lane)
        if ((mask >> lane) & 1)
          expected[lane] = (expected[lane] & ~(65535u << shift)) | (half << shift);
      auto instruction = high ? goc_rdna4_v_fma_mixhi_f16 : goc_rdna4_v_fma_mixlo_f16;
      ASSERT_EQ(instruction(cpu, mask, mode, &pd, &pa, &pb, &pc), GOC_SUCCESS);
      for (unsigned lane = 0; lane < 32; ++lane)
        ASSERT_EQ(output[lane], expected[lane]) << cpu << '/' << step << '/' << lane;
    }
  }
}

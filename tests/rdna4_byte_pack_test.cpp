// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_byte_conversion_reference.h"
#include "rdna4_byte_pack_hardware.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_cvt_pk_u8_f32);
const Fn functions[] = {
    [](uint64_t f, uint32_t m, uint64_t i, uint32_t *const *d, const uint32_t *const *a,
       const uint32_t *const *,
       const uint32_t *const *) { return goc_rdna4_v_cvt_off_f32_i4(f, m, i, d, a); },
    goc_rdna4_v_cvt_pk_u8_f32};

uint32_t mode(unsigned op, unsigned variant) {
  return op ? goc_test::byte_pack_mode(variant) : goc_test::byte_conversion_mode(variant);
}

uint32_t reference(unsigned op, uint32_t a, uint32_t b, uint32_t c, uint32_t flags) {
  return op ? goc_test::byte_pack_reference(a, b, c, flags)
            : goc_test::nibble_offset_reference(a, flags);
}

} // namespace

TEST(BytePack, HardwareCapturedRoundingAndModifiers) {
  const uint32_t modes[] = {0, GOC_ALU_CLAMP,  GOC_ALU_ABS_A | GOC_ALU_NEG_A,  GOC_ALU_ABS_A,
                            0, GOC_ALU_OMOD_2, GOC_ALU_CLAMP | GOC_ALU_OMOD_4, GOC_ALU_OMOD_HALF};
  unsigned row = 0;
  for (const auto &capture : goc_test::byte_pack_captures) {
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned column = 0; column < 8; ++column) {
        uint32_t av[32], bv[32], cv[32], output[32];
        std::fill_n(av, 32, capture.source);
        std::fill_n(bv, 32, row & 3);
        std::fill_n(cv, 32, 0x12345678);
        const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
        uint32_t *d[] = {output};
        ASSERT_EQ(functions[column < 4 ? 1 : 0](cpu, UINT32_MAX, modes[column], d, a, b, c),
                  GOC_SUCCESS);
        for (auto word : output)
          ASSERT_EQ(word, capture.expected[column]) << row << "/" << column << "/" << cpu;
      }
    ++row;
  }
}

TEST(BytePack, EveryRoundingBoundaryAndBytePosition) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned variant = 0; variant < 8; ++variant)
      for (unsigned selection = 0; selection < 4; ++selection)
        for (unsigned start = 0; start < 256 * 3; start += 32) {
          uint32_t av[32], bv[32], cv[32], output[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned index = (start + lane) % (256 * 3);
            float value = float(index / 3) + 0.5f;
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            av[lane] = bits + index % 3 - 1;
            bv[lane] = 0xfffffffc | selection;
            cv[lane] = 0x01030709u * lane;
          }
          const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
          uint32_t *d[] = {output};
          uint32_t flags = mode(1, variant);
          ASSERT_EQ(functions[1](cpu, UINT32_MAX, flags, d, a, b, c), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_EQ(output[lane], reference(1, av[lane], bv[lane], cv[lane], flags));
        }
}

TEST(BytePack, EveryNibbleModifierAndUnselectedBit) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned variant = 0; variant < 8; ++variant)
      for (unsigned noise = 4; noise <= 32; ++noise) {
        uint32_t av[32], output[32];
        for (unsigned lane = 0; lane < 32; ++lane)
          av[lane] = (lane & 15) | (noise == 32 ? 0xfffffff0 : 1u << noise);
        const uint32_t *a[] = {av};
        uint32_t *d[] = {output};
        uint32_t flags = mode(0, variant);
        ASSERT_EQ(functions[0](cpu, UINT32_MAX, flags, d, a, nullptr, nullptr), GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane)
          ASSERT_EQ(output[lane], reference(0, av[lane], 0, 0, flags));
      }
}

TEST(BytePack, EveryModifierMaskAndWholeRegisterAlias) {
  std::mt19937 random(449);
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant)
        for (uint32_t mask : rdna4_exec_masks())
          for (unsigned breg = 0; breg < (op ? 2u : 1u); ++breg)
            for (unsigned creg = 0; creg < (op ? 3u : 1u); ++creg)
              for (unsigned dreg = 0; dreg < 4; ++dreg) {
                uint32_t storage[4][34], expected[4][34];
                for (unsigned reg = 0; reg < 4; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    storage[reg][word] = expected[reg][word] = random();
                uint32_t flags = mode(op, variant);
                for (unsigned lane = 0; lane < 32; ++lane)
                  if ((mask >> lane) & 1)
                    expected[dreg][lane + 1] =
                        reference(op, storage[0][lane + 1], storage[breg][lane + 1],
                                  storage[creg][lane + 1], flags);
                const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1},
                               *c[] = {storage[creg] + 1};
                uint32_t *d[] = {storage[dreg] + 1};
                ASSERT_EQ(functions[op](cpu | GOC_FP16_OVFL, mask, flags, d, a, b, c), GOC_SUCCESS);
                for (unsigned reg = 0; reg < 4; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    ASSERT_EQ(storage[reg][word], expected[reg][word])
                        << op << "/" << cpu << "/" << variant;
              }
}

TEST(BytePack, HostRoundingAndOffsetEnvironment) {
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (unsigned op = 0; op < 2; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (const auto &capture : goc_test::byte_pack_captures) {
          uint32_t av[32], bv[32], cv[32], output[32];
          std::fill_n(av, 32, capture.source);
          for (unsigned lane = 0; lane < 32; ++lane) {
            bv[lane] = lane;
            cv[lane] = 0x12345678;
          }
          const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
          uint32_t *d[] = {output};
          EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          EXPECT_EQ(std::feraiseexcept(FE_INEXACT), 0);
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          EXPECT_EQ(functions[op](cpu, UINT32_MAX, 0, d, a, b, c), GOC_SUCCESS);
          if (!op) {
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
          }
          EXPECT_EQ(std::fegetround(), rounding);
          for (unsigned lane = 0; lane < 32; ++lane)
            EXPECT_EQ(output[lane], reference(op, av[lane], bv[lane], cv[lane], 0));
        }
  }
  EXPECT_EQ(std::fesetenv(&saved), 0);
}

TEST(BytePack, ValidationAndSemanticFallback) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      uint32_t known = mode(op, 7);
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!(known & (1u << bit))) {
          EXPECT_EQ(functions[op](cpu, UINT32_MAX, 1u << bit, d, a, a, a), GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(functions[op](cpu, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), UINT32_MAX, 0, d, a, a, a),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                              UINT32_MAX, 0, d, a, a, a),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeefu);
      EXPECT_EQ(functions[op](cpu, UINT32_C(0), 0, nullptr, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      for (unsigned sem = 0; sem < 4; ++sem) {
        EXPECT_EQ(functions[op](cpu | (uint64_t(sem) << 16), UINT32_MAX, 0, d, a, a, a),
                  GOC_SUCCESS);
        for (auto word : output)
          EXPECT_EQ(word, 0u);
      }
    }
}

TEST(BytePack, DppModifiersMasksAliasesAndGuards) {
  std::mt19937 random(449);
  for (auto descriptor : goc_test::dpp_modes)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant)
        for (uint32_t mask : rdna4_exec_masks())
          for (unsigned breg = 0; breg < 2; ++breg)
            for (unsigned creg = 0; creg < 3; ++creg)
              for (unsigned dreg = 0; dreg < 4; ++dreg) {
                uint32_t storage[4][34], expected[4][34];
                for (unsigned reg = 0; reg < 4; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    storage[reg][word] = expected[reg][word] = random();
                uint64_t flags = descriptor | goc_test::byte_pack_mode(variant);
                for (unsigned lane = 0; lane < 32; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(flags, mask, lane, source))
                    expected[dreg][lane + 1] = goc_test::byte_pack_reference(
                        source < 0 ? 0 : storage[0][source + 1], storage[breg][lane + 1],
                        storage[creg][lane + 1], uint32_t(flags));
                }
                const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1},
                               *c[] = {storage[creg] + 1};
                uint32_t *d[] = {storage[dreg] + 1};
                ASSERT_EQ(goc_rdna4_v_cvt_pk_u8_f32(cpu | GOC_FP16_OVFL, mask, flags, d, a, b, c),
                          GOC_SUCCESS);
                for (unsigned reg = 0; reg < 4; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    ASSERT_EQ(storage[reg][word], expected[reg][word])
                        << descriptor << "/" << cpu << "/" << variant;
              }
}

TEST(BytePack, DppValidation) {
  for (auto descriptor : goc_test::dpp_modes) {
    EXPECT_EQ(goc_rdna4_v_cvt_pk_u8_f32(0, 0, descriptor, nullptr, nullptr, nullptr, nullptr),
              GOC_SUCCESS);
    for (uint64_t invalid : {UINT64_C(1) << 36, uint64_t(GOC_ALU_OMOD_2), uint64_t(GOC_ALU_HIGH_A)})
      EXPECT_EQ(
          goc_rdna4_v_cvt_pk_u8_f32(0, 0, descriptor | invalid, nullptr, nullptr, nullptr, nullptr),
          GOC_ERROR_INVALID_FLAGS);
  }
}

// RX 9070 capture: seven DPP descriptors, eight modifiers, eight EXEC masks.
TEST(BytePack, DppHardwareCorpus) {
  const uint32_t values[] = {0x3f000000, 0x3fc00000, 0x40200000, 0xbf000000,
                             0x7f800001, 0xff800000, 0x437e8000, 0x437f8000};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  const uint32_t modifiers[] = {0, 1, 8, 9, 256, 257, 264, 265};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (auto mask : masks)
      for (auto descriptor : goc_test::dpp_modes)
        for (auto modifier : modifiers) {
          uint32_t av[32], bv[32], cv[32], output[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            av[lane] = values[lane % 8];
            bv[lane] = lane;
            cv[lane] = 0x01030709u * lane;
            output[lane] = 0xdead0000u + lane;
          }
          const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
          uint32_t *d[] = {output};
          ASSERT_EQ(goc_rdna4_v_cvt_pk_u8_f32(cpu, mask, descriptor | modifier, d, a, b, c),
                    GOC_SUCCESS);
          for (auto word : output)
            hash = (hash ^ word) * UINT64_C(1099511628211);
        }
    EXPECT_EQ(hash, UINT64_C(0xef165148657501e5)) << cpu;
  }
}

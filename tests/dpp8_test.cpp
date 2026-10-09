// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "exec_masks.h"
#include "goc/goc.h"
#include "omod_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {
uint32_t bits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return u;
}

float number(uint32_t u) {
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}

int call(bool accumulate, uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return accumulate ? goc_v_fmac_f32(flags, exec_mask, mode, d, a, b, nullptr)
                    : goc_v_fma_f32(flags, exec_mask, mode, d, a, b, c, nullptr);
}

float modify(uint32_t raw, uint32_t mode, unsigned operand) {
  if (mode & (8u << operand))
    raw &= 0x7fffffff;
  if (mode & (1u << operand))
    raw ^= 0x80000000;
  return number(raw);
}
} // namespace

TEST(Dpp8, HardwareCorpus) {
  // GFX1201/HIP 7.13: FMA and FMAC, FI 0/1, eight selectors, NEG_A/OMOD
  // combinations, eight EXEC masks. All 16384 raw result words matched.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t exec_mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (bool accumulate : {false, true})
        for (unsigned fi = 0; fi < 2; ++fi)
          for (unsigned pattern = 0; pattern < 8; ++pattern)
            for (unsigned mod = 0; mod < 2; ++mod) {
              uint32_t sel = 0;
              for (unsigned i = 0; i < 8; ++i)
                sel |= ((i * 3 + pattern) & 7) << (3 * i);
              uint64_t mode = GOC_DPP8 | (fi ? GOC_DPP_FI : 0) |
                              (uint64_t(sel) << GOC_DPP8_SELECT_SHIFT) |
                              (mod ? GOC_ALU_NEG_A | GOC_ALU_OMOD_2 : 0);
              uint32_t data[4][32];
              for (unsigned i = 0; i < 32; ++i) {
                data[0][i] = bits(float(i + 1));
                data[1][i] = bits(2);
                data[2][i] = data[3][i] = bits(float(100 + i));
              }
              uint32_t *d = data[3];
              const uint32_t *a = data[0], *b = data[1], *c = data[2];
              ASSERT_EQ(call(accumulate, cpu, exec_mask, mode, &d, &a, &b, &c), GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane) {
                unsigned src = (lane & ~7u) | ((lane * 3 + pattern) & 7);
                int value = fi || ((exec_mask >> src) & 1) ? int(src + 1) : 0;
                int want = 100 + int(lane);
                if ((exec_mask >> lane) & 1)
                  want = mod ? (want - 2 * value) * 2 : want + 2 * value;
                ASSERT_EQ(d[lane], bits(float(want)));
                hash = goc_test::capture_hash_word(hash, d[lane]);
              }
            }
    EXPECT_EQ(hash, 0x3c27b0cc204f2325ULL);
  }
}

TEST(Dpp8, ArithmeticModifiersMasksAndAliases) {
  auto masks = exec_masks();
  std::mt19937 random(87321);
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool accumulate : {false, true})
      for (unsigned low = 0; low < 512; ++low) {
        if (accumulate && (low & (GOC_ALU_NEG_C | GOC_ALU_ABS_C)))
          continue;
        for (unsigned alias = 0; alias < 5; ++alias)
          for (unsigned fi = 0; fi < 2; ++fi) {
            uint32_t data[4][34], saved[4][34], want[32];
            for (auto &reg : data)
              for (auto &word : reg)
                word = bits(float(int(random() % 65) - 32) / 8);
            std::memcpy(saved, data, sizeof(data));
            unsigned di = alias < 4 ? alias : 0, bi = alias == 4 ? 0 : 1,
                     ci = accumulate   ? di
                          : alias == 4 ? 0
                                       : 2;
            uint32_t sel = random() & 0xffffff;
            uint32_t exec_mask = masks[(low + alias * 17) % masks.size()];
            uint64_t mode =
                low | GOC_DPP8 | (fi ? GOC_DPP_FI : 0) | (uint64_t(sel) << GOC_DPP8_SELECT_SHIFT);
            for (unsigned lane = 0; lane < 32; ++lane) {
              want[lane] = saved[di][lane + 1];
              if (!((exec_mask >> lane) & 1))
                continue;
              unsigned src = (lane & ~7u) | ((sel >> (3 * (lane & 7))) & 7);
              uint32_t raw = fi || ((exec_mask >> src) & 1) ? saved[0][src + 1] : 0;
              float value = std::fma(modify(raw, low, 0), modify(saved[bi][lane + 1], low, 1),
                                     modify(saved[ci][lane + 1], low, 2));
              value = goc_test::omod_f32_reference(value, low);
              if (low & GOC_ALU_CLAMP) {
                value = std::min(std::max(value, 0.f), 1.f);
                // Architectural clamp canonicalizes either zero sign to +0.
                if (value == 0)
                  value = 0;
              }
              want[lane] = bits(value);
            }
            uint32_t *d = data[di] + 1;
            const uint32_t *a = data[0] + 1, *b = data[bi] + 1, *c = data[ci] + 1;
            ASSERT_EQ(call(accumulate, cpu, exec_mask, mode, &d, &a, &b, &c), GOC_SUCCESS);
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 34; ++lane)
                ASSERT_EQ(data[reg][lane],
                          reg == di && lane > 0 && lane < 33 ? want[lane - 1] : saved[reg][lane]);
          }
      }
}

TEST(Dpp8, InvalidFlagsAndSemantics) {
  for (bool accumulate : {false, true}) {
    uint32_t d[32];
    std::fill_n(d, 32, 0x12345678);
    uint32_t *pd = d;
    for (uint64_t mode : std::initializer_list<uint64_t>{
             GOC_DPP_FI, GOC_DPP8_SELECT_MASK, GOC_DPP8 | (1ULL << 34), GOC_DPP8 | (1ULL << 31)})
      EXPECT_EQ(call(accumulate, 0, UINT32_MAX, mode, &pd, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(call(accumulate, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                   GOC_DPP8, &pd, nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(call(accumulate, 1ULL << 63, 0, GOC_DPP8, &pd, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    if (accumulate) {
      EXPECT_EQ(call(true, 0, UINT32_MAX, GOC_DPP8 | GOC_ALU_NEG_C, &pd, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    }
    for (auto word : d)
      EXPECT_EQ(word, 0x12345678u);
  }
  EXPECT_EQ(goc_v_fma_dx9_zero_f32(0, 0, GOC_DPP8, nullptr, nullptr, nullptr, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(goc_v_fma_dx9_zero_f32(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0,
                                   GOC_DPP8, nullptr, nullptr, nullptr, nullptr),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
}

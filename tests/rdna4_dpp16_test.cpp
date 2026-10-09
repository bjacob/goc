// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "rdna4_dpp16_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_omod_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
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

float modify(uint32_t raw, unsigned mode, unsigned operand) {
  if (mode & (8u << operand))
    raw &= 0x7fffffff;
  if (mode & (1u << operand))
    raw ^= 0x80000000;
  return number(raw);
}

int call(bool accumulate, uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return accumulate ? goc_rdna4_v_fmac_f32(flags, mask, mode, d, a, b)
                    : goc_rdna4_v_fma_f32(flags, mask, mode, d, a, b, c);
}
} // namespace

TEST(Dpp16, HardwareCorpus) {
  // GFX1201/HIP 7.13: all control families, boundary shifts, FI/BOUND_CTRL,
  // four row/bank combinations, and eight EXEC masks: 65536 raw results.
  const unsigned controls[] = {0xe4,  0x1b,  0,     0xff,  0x101, 0x10f, 0x111, 0x11f,
                               0x121, 0x12f, 0x140, 0x141, 0x150, 0x15f, 0x160, 0x16f};
  const unsigned row_bank[][2] = {{15, 15}, {1, 5}, {2, 10}, {0, 15}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned ctrl : controls)
        for (unsigned fi = 0; fi < 2; ++fi)
          for (unsigned bc = 0; bc < 2; ++bc)
            for (auto &rb : row_bank) {
              uint32_t data[4][32];
              for (unsigned i = 0; i < 32; ++i) {
                data[0][i] = bits(float(i + 1));
                data[1][i] = bits(2);
                data[2][i] = bits(float(100 + i));
                data[3][i] = bits(float(1000 + i));
              }
              uint32_t *d = data[3];
              const uint32_t *a = data[0], *b = data[1], *c = data[2];
              auto mode = goc_test::dpp16_mode(ctrl, fi, bc, rb[0], rb[1]);
              ASSERT_EQ(call(false, cpu, mask, mode, &d, &a, &b, &c), GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane) {
                int src;
                float want = float(1000 + lane);
                if (goc_test::dpp16_reference(ctrl, fi, bc, rb[0], rb[1], mask, lane, src))
                  want = float(100 + lane + 2 * (src < 0 ? 0 : src + 1));
                ASSERT_EQ(d[lane], bits(want));
                hash = goc_test::capture_hash_word(hash, d[lane]);
              }
            }
    EXPECT_EQ(hash, UINT64_C(0x72f55867e357a325));
  }
}

TEST(Dpp16, EveryControlMasksAndAliases) {
  auto masks = rdna4_exec_masks();
  std::mt19937 random(17456);
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned ctrl = 0; ctrl < 512; ++ctrl) {
      if (!goc_test::dpp16_control(ctrl))
        continue;
      for (unsigned fi = 0; fi < 2; ++fi)
        for (unsigned bc = 0; bc < 2; ++bc)
          for (uint32_t mask : masks)
            for (unsigned alias = 0; alias < 4; ++alias) {
              uint32_t data[4][34], saved[4][34];
              for (auto &reg : data)
                for (auto &word : reg)
                  word = bits(float(random() % 128));
              std::memcpy(saved, data, sizeof(data));
              unsigned rows = (ctrl + alias) % 16, banks = (ctrl / 16 + alias) % 16;
              uint32_t *d = data[alias] + 1;
              const uint32_t *a = data[0] + 1, *b = data[1] + 1, *c = data[2] + 1;
              ASSERT_EQ(call(false, cpu, mask, goc_test::dpp16_mode(ctrl, fi, bc, rows, banks), &d,
                             &a, &b, &c),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned lane = 0; lane < 34; ++lane) {
                  uint32_t want = saved[reg][lane];
                  int source;
                  if (reg == alias && lane > 0 && lane < 33 &&
                      goc_test::dpp16_reference(ctrl, fi, bc, rows, banks, mask, lane - 1, source))
                    want = bits((source < 0 ? 0.f : number(saved[0][source + 1])) *
                                    number(saved[1][lane]) +
                                number(saved[2][lane]));
                  ASSERT_EQ(data[reg][lane], want);
                }
            }
    }
}

TEST(Dpp16, AllRowBankAndArithmeticModifiers) {
  std::mt19937 random(14591);
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool accumulate : {false, true})
      for (unsigned low = 0; low < 512; ++low) {
        if (accumulate && (low & (GOC_ALU_NEG_C | GOC_ALU_ABS_C)))
          continue;
        unsigned rows = low % 16, banks = low / 16 % 16;
        for (unsigned flags = 0; flags < 4; ++flags) {
          bool fi = flags & 1, bc = flags & 2;
          uint32_t mask = random();
          unsigned ctrl = 0x101 + (low % 15);
          uint64_t mode = goc_test::dpp16_mode(ctrl, fi, bc, rows, banks) | low;
          uint32_t data[4][32], saved[4][32];
          for (auto &reg : data)
            for (auto &word : reg)
              word = bits(float(int(random() % 65) - 32) / 8);
          std::memcpy(saved, data, sizeof(data));
          unsigned di = low % 4, ci = accumulate ? di : 2;
          uint32_t *d = data[di];
          const uint32_t *a = data[0], *b = data[1], *c = data[ci];
          ASSERT_EQ(call(accumulate, cpu, mask, mode, &d, &a, &b, &c), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int src;
            uint32_t want = saved[di][lane];
            if (goc_test::dpp16_reference(ctrl, fi, bc, rows, banks, mask, lane, src)) {
              float value =
                  std::fma(modify(src < 0 ? 0 : saved[0][src], low, 0),
                           modify(saved[1][lane], low, 1), modify(saved[ci][lane], low, 2));
              value = goc_test::omod_f32_reference(value, low);
              if (low & GOC_ALU_CLAMP) {
                value = std::clamp(value, 0.f, 1.f);
                if (value == 0)
                  value = 0;
              }
              want = bits(value);
            }
            ASSERT_EQ(d[lane], want);
          }
        }
      }
}

TEST(Dpp16, InvalidControlAndDescriptorFields) {
  for (unsigned ctrl = 0; ctrl < 512; ++ctrl) {
    if (goc_test::dpp16_control(ctrl))
      continue;
    EXPECT_EQ(call(false, 0, 0, goc_test::dpp16_mode(ctrl, 0, 0, 15, 15), nullptr, nullptr, nullptr,
                   nullptr),
              GOC_ERROR_INVALID_FLAGS);
  }
  for (uint64_t mode :
       {GOC_DPP16 | GOC_DPP8, GOC_DPP16 | (UINT64_C(1) << 57), GOC_DPP16 | (UINT64_C(1) << 36),
        GOC_DPP_BOUND_CTRL, GOC_DPP_ROW_MASK, GOC_DPP_BANK_MASK})
    EXPECT_EQ(call(false, 0, 0, mode, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
  EXPECT_EQ(call(false, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, GOC_DPP16, nullptr,
                 nullptr, nullptr, nullptr),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
}

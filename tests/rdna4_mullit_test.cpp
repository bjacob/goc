// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_mullit_hardware.h"
#include "rdna4_mullit_reference.h"

#include <cstring>

#include <gtest/gtest.h>
#include <stdint.h>

TEST(Mullit, HardwareSpecialValuesAndModifiers) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned mode = 0; mode < 16; ++mode) {
      uint64_t digest = UINT64_C(14695981039346656037);
      for (unsigned start = 0; start < 4096; start += 32) {
        uint32_t words[4][32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          unsigned index = start + lane;
          words[0][lane] = goc_test::mullit_capture_values[index / 256];
          words[1][lane] = goc_test::mullit_capture_values[(index / 16) % 16];
          words[2][lane] = goc_test::mullit_capture_values[index % 16];
        }
        const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
        uint32_t *d[] = {words[3]};
        ASSERT_EQ(goc_rdna4_v_mullit_f32(cpu, UINT32_MAX, goc_test::mullit_capture_modes[mode], d,
                                         a, b, c),
                  GOC_SUCCESS);
        for (uint32_t word : words[3]) {
          if ((word & 0x7fffffff) > 0x7f800000)
            word = 0x7fc00000;
          for (unsigned byte = 0; byte < 4; ++byte) {
            digest ^= (word >> (8 * byte)) & 255;
            digest *= UINT64_C(1099511628211);
          }
        }
      }
      EXPECT_EQ(digest, goc_test::mullit_capture_digests[mode]) << cpu << "/" << mode;
    }
}

TEST(Mullit, AllModifiersMasksAndAliases) {
  uint32_t initial[4][35];
  for (unsigned reg = 0; reg < 4; ++reg)
    for (unsigned lane = 0; lane < 35; ++lane)
      initial[reg][lane] = goc_test::mullit_capture_values[(lane * (reg + 1) + reg * 3) % 16];
  for (uint32_t mode = 0; mode < 512; ++mode) {
    uint32_t result[32];
    for (unsigned lane = 0; lane < 32; ++lane)
      result[lane] = goc_test::mullit_reference(initial[0][lane + 1], initial[1][lane + 1],
                                                initial[2][lane + 1], mode);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned target = 0; target < 4; ++target)
        for (uint64_t mask : rdna4_exec_masks()) {
          uint32_t words[4][35];
          std::memcpy(words, initial, sizeof(words));
          const uint32_t *a[] = {words[0] + 1}, *b[] = {words[1] + 1}, *c[] = {words[2] + 1};
          uint32_t *d[] = {words[target] + 1};
          ASSERT_EQ(goc_rdna4_v_mullit_f32(cpu, mask, mode, d, a, b, c), GOC_SUCCESS);
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 35; ++lane) {
              uint32_t want = initial[reg][lane], got = words[reg][lane];
              if (reg == target && lane > 0 && lane <= 32 && ((mask >> (lane - 1)) & 1)) {
                want = result[lane - 1];
                if ((got & 0x7fffffff) > 0x7f800000)
                  got = 0x7fc00000;
              }
              ASSERT_EQ(got, want) << mode << "/" << cpu << "/" << target << "/" << lane;
            }
        }
  }
}

TEST(Mullit, InvalidFlagsAndEmptyExec) {
  EXPECT_EQ(goc_rdna4_v_mullit_f32(0, UINT64_C(0xffffffff00000000), 511, nullptr, nullptr, nullptr,
                                   nullptr),
            GOC_SUCCESS);
  for (unsigned bit = 9; bit < 32; ++bit)
    EXPECT_EQ(goc_rdna4_v_mullit_f32(0, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
  EXPECT_EQ(goc_rdna4_v_mullit_f32(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0,
                                   nullptr, nullptr, nullptr, nullptr),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
}

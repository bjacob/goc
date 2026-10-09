// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_mullit_hardware.h"
#include "rdna4_mullit_reference.h"

#include <cstring>

#include <gtest/gtest.h>
#include <stdint.h>

TEST(Mullit, HardwareSpecialValuesAndModifiers) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned mode = 0; mode < 16; ++mode) {
      uint64_t digest = goc_test::capture_hash_seed;
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
          digest = goc_test::capture_hash_bytes(digest, word, 4);
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
        for (uint32_t mask : rdna4_exec_masks()) {
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
  EXPECT_EQ(goc_rdna4_v_mullit_f32(0, 0U, 511, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
  for (unsigned bit = 9; bit < 32; ++bit)
    EXPECT_EQ(goc_rdna4_v_mullit_f32(0, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
  EXPECT_EQ(goc_rdna4_v_mullit_f32(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0,
                                   nullptr, nullptr, nullptr, nullptr),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
}

TEST(Mullit, DppModifiersMasksAliasesAndGuards) {
  for (uint32_t mode = 0; mode < 512; ++mode)
    for (uint64_t descriptor : goc_test::dpp_modes)
      for (bool shared : {false, true}) {
        uint32_t initial[4][34];
        for (unsigned reg = 0; reg < 4; ++reg)
          for (unsigned lane = 0; lane < 34; ++lane)
            initial[reg][lane] = goc_test::mullit_capture_values[(lane * (reg + 1) + reg * 3) % 16];
        for (uint32_t mask : rdna4_exec_masks()) {
          if (mode != 0 && mode != 511 && mask != UINT32_MAX)
            continue;
          uint32_t result[32], writes = 0;
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            if (goc_test::dpp_source(descriptor, mask, lane, source)) {
              writes |= 1u << lane;
              result[lane] = goc_test::mullit_reference(source < 0 ? 0 : initial[0][source + 1],
                                                        initial[shared ? 0 : 1][lane + 1],
                                                        initial[shared ? 0 : 2][lane + 1], mode);
            }
          }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            for (unsigned target = 0; target < 4; ++target) {
              uint32_t words[4][34];
              std::memcpy(words, initial, sizeof(words));
              const uint32_t *a[] = {words[0] + 1}, *b[] = {words[shared ? 0 : 1] + 1},
                             *c[] = {words[shared ? 0 : 2] + 1};
              uint32_t *d[] = {words[target] + 1};
              ASSERT_EQ(goc_rdna4_v_mullit_f32(cpu, mask, descriptor | mode, d, a, b, c),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned lane = 0; lane < 34; ++lane) {
                  uint32_t want = initial[reg][lane], got = words[reg][lane];
                  if (reg == target && lane > 0 && lane < 33 && ((writes >> (lane - 1)) & 1)) {
                    want = result[lane - 1];
                    if ((got & 0x7fffffff) > 0x7f800000)
                      got = 0x7fc00000;
                  }
                  ASSERT_EQ(got, want) << mode << "/" << descriptor << "/" << cpu << "/" << target;
                }
            }
        }
      }
}

TEST(Mullit, DppHardwareCorpus) {
  // RX 9070/gfx1201, MODE 0xf0: all 512 modifiers, seven DPP descriptors,
  // and eight EXEC masks. FNV hashes words with NaN payloads canonicalized.
  const uint32_t values[] = {1,          0x807fffff, 0x00800000, 0x80000000,
                             0x7f800001, 0xff800000, 0x3f800000, 0xff7fffff};
  const uint32_t masks[] = {UINT32_MAX, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t mask : masks)
      for (unsigned mode = 0; mode < 512; ++mode)
        for (uint64_t descriptor : goc_test::dpp_modes) {
          uint32_t words[4][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            words[0][lane] = values[lane % 8];
            words[1][lane] = values[(lane + 3) % 8];
            words[2][lane] = values[(lane + 5) % 8];
            words[3][lane] = 0xdead0000u + lane;
          }
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          ASSERT_EQ(goc_rdna4_v_mullit_f32(cpu, mask, descriptor | mode, d, a, b, c), GOC_SUCCESS);
          for (uint32_t word : words[3]) {
            if ((word & 0x7fffffff) > 0x7f800000)
              word = 0x7fc00000;
            hash = goc_test::capture_hash_word(hash, word);
          }
        }
    EXPECT_EQ(hash, 0x7d17538d3e94b7a5ULL);
  }
}

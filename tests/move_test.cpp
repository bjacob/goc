// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "goc/goc.h"
#include "half_move_hardware.h"

#include <gtest/gtest.h>
#include <stdint.h>

TEST(Move, SharedRdna3Rdna4BitPatternsMasksAndAliases) {
  for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
    for (uint32_t exec_mask : exec_masks())
      for (bool alias : {false, true}) {
        uint32_t input[32], output[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          input[lane] = 0x7f800001U + lane * 0x1234567U;
          output[lane] = 0xdeadbeef;
        }
        const uint32_t *a[] = {input};
        uint32_t *d[] = {alias ? input : output};
        ASSERT_EQ(goc_v_mov_b32(semantics | GOC_SEMANTICS_STRICT, exec_mask, 0, d, a), GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane)
          EXPECT_EQ(d[0][lane], alias || ((exec_mask >> lane) & 1) ? 0x7f800001U + lane * 0x1234567U
                                                                   : 0xdeadbeefU);
      }
}

TEST(Move, DppReadsBeforeAliasedStores) {
  uint32_t data[32];
  for (unsigned lane = 0; lane < 32; ++lane)
    data[lane] = lane;
  uint32_t *d[] = {data};
  const uint32_t *a[] = {data};
  uint64_t mode = GOC_DPP8;
  for (unsigned lane = 0; lane < 8; ++lane)
    mode |= uint64_t(7 - lane) << (40 + 3 * lane);
  ASSERT_EQ(goc_v_mov_b32(0, UINT32_MAX, mode, d, a), GOC_SUCCESS);
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_EQ(data[lane], lane ^ 7);
  EXPECT_EQ(goc_v_mov_b32(0, 0, 0, nullptr, nullptr), GOC_SUCCESS);
  EXPECT_EQ(goc_v_mov_b32(0, 0, 1, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
}

TEST(Move, HalfHardwareModifierCorpus) {
  uint64_t hash = goc_test::capture_hash_seed;
  for (unsigned mode = 0; mode < 32; ++mode)
    for (unsigned halves = 0; halves < 4; ++halves) {
      uint32_t input[32], output[32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        input[lane] = goc_test::half_move_inputs[lane] |
                      (uint32_t(goc_test::half_move_inputs[(lane + 7) % 32]) << 16);
        output[lane] = 0xabcd1234U;
      }
      uint64_t flags = (mode & 1 ? GOC_ALU_ABS_A : 0) | (mode & 2 ? GOC_ALU_NEG_A : 0) |
                       (((mode >> 2) & 3) * GOC_ALU_OMOD_2) | (mode & 16 ? GOC_ALU_CLAMP : 0) |
                       (halves & 1 ? GOC_ALU_HIGH_A : 0) | (halves & 2 ? GOC_ALU_HIGH_D : 0);
      const uint32_t *a = input;
      uint32_t *d = output;
      ASSERT_EQ(goc_v_mov_b16(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                              flags, &d, &a),
                GOC_SUCCESS);
      for (uint32_t value : output)
        hash = goc_test::capture_hash_word(hash, value);
    }
  EXPECT_EQ(hash, goc_test::half_move_hash);
}

TEST(Move, EveryHalfEncodingAndSignModifier) {
  for (unsigned start = 0; start < 65536; start += 32)
    for (unsigned mode = 0; mode < 16; ++mode) {
      unsigned sa = mode & 1 ? 16 : 0, sd = mode & 2 ? 16 : 0;
      uint32_t data[32], before[32];
      for (unsigned lane = 0; lane < 32; ++lane)
        before[lane] = data[lane] = (start + lane) | ((65535U - start - lane) << 16);
      uint64_t flags = (sa ? GOC_ALU_HIGH_A : 0) | (sd ? GOC_ALU_HIGH_D : 0) |
                       (mode & 4 ? GOC_ALU_ABS_A : 0) | (mode & 8 ? GOC_ALU_NEG_A : 0);
      uint32_t *d = data;
      ASSERT_EQ(goc_v_mov_b16(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                              flags, &d, &d),
                GOC_SUCCESS);
      for (unsigned lane = 0; lane < 32; ++lane) {
        uint32_t value = (before[lane] >> sa) & 65535;
        if (mode & 4)
          value &= 32767;
        if (mode & 8)
          value ^= 32768;
        EXPECT_EQ(data[lane], (before[lane] & ~(65535U << sd)) | (value << sd));
      }
    }
}

TEST(Move, HalfDppMasksAliasesAndErrors) {
  for (uint64_t dpp : goc_test::dpp_modes)
    for (unsigned halves = 0; halves < 4; ++halves)
      for (uint32_t exec_mask : exec_masks()) {
        uint32_t words[34], before[34];
        for (unsigned lane = 0; lane < 34; ++lane)
          before[lane] = words[lane] = 0xabcd0000U + lane;
        unsigned sa = halves & 1 ? 16 : 0, sd = halves & 2 ? 16 : 0;
        uint64_t mode = dpp | GOC_ALU_NEG_A | (sa ? GOC_ALU_HIGH_A : 0) | (sd ? GOC_ALU_HIGH_D : 0);
        uint32_t *d = words + 1;
        ASSERT_EQ(goc_v_mov_b16(GOC_SEMANTICS_EXACT_EMPIRICAL, exec_mask, mode, &d, &d),
                  GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane) {
          int source;
          uint32_t expected = before[lane + 1];
          if (goc_test::dpp_source(dpp, exec_mask, lane, source)) {
            uint32_t value = source < 0 ? 0 : (before[source + 1] >> sa) & 65535;
            expected = (expected & ~(65535U << sd)) | ((value ^ 32768) << sd);
          }
          EXPECT_EQ(words[lane + 1], expected);
        }
        EXPECT_EQ(words[0], before[0]);
        EXPECT_EQ(words[33], before[33]);
      }
  EXPECT_EQ(goc_v_mov_b16(0, 0, 0, nullptr, nullptr), GOC_SUCCESS);
  EXPECT_EQ(goc_v_mov_b16(0, UINT32_MAX, GOC_ALU_NEG_B, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
}

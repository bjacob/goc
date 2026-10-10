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
  EXPECT_EQ(goc_v_mov_b32(0, 0, GOC_ALU_NEG_B, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
}

TEST(Move, HalfHardwareModifierCorpus) {
  for (unsigned lanes : {32U, 64U}) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (unsigned mode = 0; mode < 32; ++mode)
      for (unsigned halves = 0; halves < 4; ++halves) {
        uint32_t input[64], output[64];
        for (unsigned lane = 0; lane < lanes; ++lane) {
          input[lane] = goc_test::half_move_inputs[lane % 32] |
                        (uint32_t(goc_test::half_move_inputs[(lane + 7) % 32]) << 16);
          output[lane] = 0xabcd1234U;
        }
        uint64_t flags = (mode & 1 ? GOC_ALU_ABS_A : 0) | (mode & 2 ? GOC_ALU_NEG_A : 0) |
                         (((mode >> 2) & 3) * GOC_ALU_OMOD_2) | (mode & 16 ? GOC_ALU_CLAMP : 0) |
                         (halves & 1 ? GOC_ALU_HIGH_A : 0) | (halves & 2 ? GOC_ALU_HIGH_D : 0);
        const uint32_t *a = input;
        uint32_t *d = output;
        ASSERT_EQ(lanes == 32
                      ? goc_v_mov_b16(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                      UINT32_MAX, flags, &d, &a)
                      : goc_v_mov_b16_wave64(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                             UINT64_MAX, flags, &d, &a),
                  GOC_SUCCESS);
        for (unsigned lane = 0; lane < lanes; ++lane)
          hash = goc_test::capture_hash_word(hash, output[lane]);
      }
    EXPECT_EQ(hash, lanes == 32 ? goc_test::half_move_hash : goc_test::half_move_wave64_hash);
  }
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

TEST(Move, Wave64DppRowsMasksAliasesAndHalfModifiers) {
  const unsigned controls[] = {0xe4,  0x1b,  0x101, 0x10f, 0x111, 0x11f, 0x121,
                               0x12f, 0x140, 0x141, 0x150, 0x15f, 0x160, 0x16f};
  const uint64_t masks[] = {0, 0x8000000100000001ULL, 0xaaaaaaaaaaaaaaaaULL, UINT64_MAX};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool half : {false, true})
      for (unsigned ctrl : controls)
        for (unsigned rows = 0; rows < 16; ++rows)
          for (unsigned fi_bound = 0; fi_bound < 4; ++fi_bound)
            for (uint64_t exec_mask : masks) {
              uint64_t mode = goc_test::dpp16_mode(ctrl, fi_bound & 1, fi_bound & 2, rows, 5);
              if (half)
                mode |= GOC_ALU_HIGH_A | GOC_ALU_NEG_A | GOC_ALU_HIGH_D;
              uint32_t data[66], before[64];
              data[0] = data[65] = 0xdeadbeef;
              for (unsigned lane = 0; lane < 64; ++lane)
                data[lane + 1] = before[lane] = (0x7c01U + lane) << 16 | (0x1234U + lane);
              auto d = data + 1;
              const uint32_t *a = d;
              ASSERT_EQ(half ? goc_v_mov_b16_wave64(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, exec_mask,
                                                    mode, &d, &a)
                             : goc_v_mov_b32_wave64(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, exec_mask,
                                                    mode, &d, &a),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source;
                uint32_t expected = before[lane];
                if (goc_test::dpp_source(mode, exec_mask, lane, source)) {
                  uint32_t value = source < 0 ? 0 : before[source];
                  expected =
                      half ? ((value ^ 0x80000000U) & 0xffff0000U) | (before[lane] & 65535) : value;
                }
                ASSERT_EQ(d[lane], expected) << half << '/' << ctrl << '/' << rows << '/' << lane;
              }
              EXPECT_EQ(data[0], 0xdeadbeefU);
              EXPECT_EQ(data[65], 0xdeadbeefU);
            }
}

TEST(Move, Wave64Dpp8RawMovesAndValidation) {
  for (bool half : {false, true})
    for (uint64_t exec_mask : {0ULL, 1ULL << 63, 0xaaaaaaaaaaaaaaaaULL, ~0ULL})
      for (uint64_t mode : {0ULL, 0x539770100000000ULL, 0x539770300000000ULL}) {
        uint32_t input[64], output[64];
        for (unsigned lane = 0; lane < 64; ++lane) {
          input[lane] = 0x81230000U + lane;
          output[lane] = 0xdeadbeef;
        }
        auto d = output;
        const uint32_t *a = input;
        ASSERT_EQ(half ? goc_v_mov_b16_wave64(0, exec_mask, mode, &d, &a)
                       : goc_v_mov_b32_wave64(0, exec_mask, mode, &d, &a),
                  GOC_SUCCESS);
        for (unsigned lane = 0; lane < 64; ++lane) {
          int source = int(lane);
          bool write =
              mode ? goc_test::dpp_source(mode, exec_mask, lane, source) : (exec_mask >> lane) & 1;
          uint32_t value = source < 0 ? 0 : input[source];
          uint32_t expected = write ? (half ? 0xdead0000U | (value & 65535) : value) : 0xdeadbeef;
          EXPECT_EQ(output[lane], expected);
        }
      }
  for (auto fn : {goc_v_mov_b32_wave64, goc_v_mov_b16_wave64}) {
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, 1ULL << 63, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, UINT64_MAX, GOC_DPP8 | GOC_ALU_NEG_C, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
  }
}

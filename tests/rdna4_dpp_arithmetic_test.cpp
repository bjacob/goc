// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp_arithmetic_reference.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

TEST(DppArithmetic, HardwareCorpus) {
  // GFX1201/HIP 7.13, MODE 0xf0 (RNE, preserve denormals): 18 operations, seven
  // DPP8/DPP16/FI/BOUND/row/bank descriptors, NEG_A/OMOD combinations and eight EXEC masks: 64512
  // raw words.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned op = 0; op < 18; ++op)
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (unsigned modified = 0; modified < 2; ++modified) {
            uint32_t data[4][32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              data[0][lane] = goc::as_bits(float(int(lane % 11) - 5));
              data[1][lane] = goc::as_bits(float(int(lane % 7) - 3));
              data[2][lane] = goc::as_bits(float(int(lane % 9) - 4));
              data[3][lane] = goc::as_bits(float(100 + lane));
            }
            uint32_t *d = data[3];
            const uint32_t *a = data[0], *b = data[1], *c = data[2];
            uint32_t low = modified ? GOC_ALU_NEG_A | GOC_ALU_OMOD_2 : 0;
            ASSERT_EQ(
                goc_test::dpp_arithmetic_call(op, cpu, mask, descriptor | low, &d, &a, &b, &c),
                GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              int source;
              uint32_t want = goc::as_bits(float(100 + lane));
              if (goc_test::dpp_source(descriptor, mask, lane, source))
                want = goc_test::dpp_arithmetic_reference(op, source < 0 ? 0 : a[source], b[lane],
                                                          c[lane], low);
              ASSERT_EQ(d[lane], want) << op << "/" << cpu << "/" << descriptor << "/" << lane;
              hash = goc_test::capture_hash_word(hash, d[lane]);
            }
          }
    EXPECT_EQ(hash, UINT64_C(0xc028318d5f87d325));
  }
}

TEST(DppArithmetic, OmodBoundaryHardware) {
  const uint32_t values[] = {0x0,        0x1,        0x1fffff,   0x3fffff,   0x400000,   0x400001,
                             0x7ffffe,   0x7fffff,   0x800000,   0x800001,   0xbfffff,   0xffffff,
                             0x1000000,  0x1000001,  0x1800000,  0x3f800000, 0x80000000, 0x80000001,
                             0x801fffff, 0x803fffff, 0x80400000, 0x80400001, 0x807ffffe, 0x807fffff,
                             0x80800000, 0x80800001, 0x80bfffff, 0x80ffffff, 0x81000000, 0x81000001,
                             0x81800000, 0xbf800000};
  // GFX1201/HIP 7.13, MODE 0xf0: tiny FP32 inputs and all OMOD values, seven
  // DPP8/DPP16/FI/BOUND/row/bank descriptors, NEG_A/OMOD combinations and eight EXEC masks: 64512
  // raw words.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned op = 0; op < 18; ++op)
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (unsigned modified = 0; modified < 4; ++modified) {
            uint32_t data[4][32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              data[0][lane] = values[lane];
              data[1][lane] = goc::as_bits(float(int(lane % 7) - 3));
              data[2][lane] = goc::as_bits(float(int(lane % 9) - 4));
              data[3][lane] = goc::as_bits(float(100 + lane));
            }
            uint32_t *d = data[3];
            const uint32_t *a = data[0], *b = data[1], *c = data[2];
            uint32_t low = modified ? GOC_ALU_NEG_A | (modified << 6) : 0;
            ASSERT_EQ(
                goc_test::dpp_arithmetic_call(op, cpu, mask, descriptor | low, &d, &a, &b, &c),
                GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              int source;
              uint32_t want = goc::as_bits(float(100 + lane));
              if (goc_test::dpp_source(descriptor, mask, lane, source))
                want = goc_test::dpp_arithmetic_reference(op, source < 0 ? 0 : a[source], b[lane],
                                                          c[lane], low);
              ASSERT_EQ(d[lane], want) << op << "/" << cpu << "/" << descriptor << "/" << lane;
              hash = goc_test::capture_hash_word(hash, d[lane]);
            }
          }
    EXPECT_EQ(hash, UINT64_C(0x3fb5391fac1f8c9e));
  }
}

TEST(DppArithmetic, ModifiersMasksAliasesAndSpecialValues) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000,
                             0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x3f7fffff,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
                             0x3e800000, 0x40000000, 0xc0400000, 0x7f812345, 0xff812346};
  auto masks = rdna4_exec_masks();
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 18; ++op)
      for (unsigned low = 0; low < 512; ++low) {
        if (op < 9 && (low & (GOC_ALU_NEG_C | GOC_ALU_ABS_C)))
          continue;
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (unsigned alias = 0; alias < 5; ++alias) {
            uint32_t data[4][34], before[4][34];
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 34; ++lane)
                data[reg][lane] = values[(lane * (2 * reg + 1) + reg * 3 + low) % 23];
            std::memcpy(before, data, sizeof(data));
            unsigned di = alias < 4 ? alias : 0, bi = alias == 4 ? 0 : 1, ci = alias == 4 ? 0 : 2;
            uint32_t mask = masks[(low + alias * 17) % masks.size()];
            uint32_t *d = data[di] + 1;
            const uint32_t *a = data[0] + 1, *b = data[bi] + 1, *c = data[ci] + 1;
            ASSERT_EQ(
                goc_test::dpp_arithmetic_call(op, cpu, mask, descriptor | low, &d, &a, &b, &c),
                GOC_SUCCESS);
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 34; ++lane) {
                int source;
                uint32_t want = before[reg][lane];
                bool written = reg == di && lane > 0 && lane < 33 &&
                               goc_test::dpp_source(descriptor, mask, lane - 1, source);
                if (written)
                  want =
                      goc_test::dpp_arithmetic_reference(op, source < 0 ? 0 : before[0][source + 1],
                                                         before[bi][lane], before[ci][lane], low);
                if (written && std::isnan(goc::as_float(want))) {
                  ASSERT_TRUE(std::isnan(goc::as_float(data[reg][lane])));
                } else {
                  ASSERT_EQ(data[reg][lane], want) << op << "/" << cpu << "/" << low << "/"
                                                   << descriptor << "/" << mask << "/" << alias;
                }
              }
          }
      }
}

TEST(DppArithmetic, ValidationBeforeOperandAccess) {
  for (unsigned op = 0; op < 18; ++op)
    for (uint64_t descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(goc_test::dpp_arithmetic_call(op, 0, UINT32_MAX, descriptor | (UINT64_C(1) << 31),
                                              nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(goc_test::dpp_arithmetic_call(op, UINT64_C(1) << 63, UINT32_MAX, descriptor,
                                              nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(
          goc_test::dpp_arithmetic_call(op, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                        UINT32_MAX, descriptor, nullptr, nullptr, nullptr, nullptr),
          GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(
          goc_test::dpp_arithmetic_call(op, 0, 0, descriptor, nullptr, nullptr, nullptr, nullptr),
          GOC_SUCCESS);
      if (op < 9) {
        EXPECT_EQ(goc_test::dpp_arithmetic_call(op, 0, UINT32_MAX, descriptor | GOC_ALU_NEG_C,
                                                nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
    }
}

TEST(DppArithmetic, OmodWithoutPermutation) {
  const uint32_t inputs[] = {0x80000000, 0x007fffff, 0x807fffff, 0x00800000,
                             0x80800000, 0x00ffffff, 0x80ffffff, 0x01000000};
  const uint32_t doubled[] = {0, 0, 0, 0x01000000, 0x81000000, 0x017fffff, 0x817fffff, 0x01800000};
  const uint32_t halved[] = {0, 0, 0, 0, 0x80000000, 0, 0x80000000, 0x00800000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned scale : {1u, 3u}) {
      uint32_t a[32], b[32] = {}, d[32];
      for (unsigned lane = 0; lane < 32; ++lane)
        a[lane] = inputs[lane % 8];
      const uint32_t *pa = a, *pb = b;
      uint32_t *pd = d;
      ASSERT_EQ(goc_rdna4_v_add_f32(cpu, UINT32_MAX, uint64_t(scale) << 6, &pd, &pa, &pb),
                GOC_SUCCESS);
      for (unsigned lane = 0; lane < 32; ++lane)
        EXPECT_EQ(d[lane], scale == 1 ? doubled[lane % 8] : halved[lane % 8]);
    }
}

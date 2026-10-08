// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dot_fixtures.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(Dot2, AllModifiersSelectionsMasksAndAliases) {
  const uint16_t f16[] = {0xc000, 0xbc00, 0x3c00, 0x4000};
  const uint16_t bf16[] = {0xc000, 0xbf80, 0x3f80, 0x4000};
  const int values[] = {-2, -1, 1, 2};
  for (bool brain : {false, true}) {
    const auto *codes = brain ? bf16 : f16;
    auto fn = brain ? goc_rdna4_v_dot2_f32_bf16 : goc_rdna4_v_dot2_f32_f16;
    for (uint32_t selection = 0; selection < 16; ++selection)
      for (uint32_t negate = 0; negate < 32; ++negate)
        for (uint32_t clamp : {UINT32_C(0), GOC_DOT_CLAMP}) {
          const uint32_t mode = negate | (selection << 7) | clamp;
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
              for (uint64_t mask : {UINT64_MAX, UINT64_C(0xaaaaaaaa), UINT64_C(0xffffffff00000000),
                                    UINT64_C(0x80018001)})
                for (int alias = 0; alias < 4; ++alias) {
                  SCOPED_TRACE(::testing::Message() << brain << "/" << mode << "/" << cpu << "/"
                                                    << semantics << "/" << mask << "/" << alias);
                  uint32_t storage[4][34], before[32], expected[32];
                  uint32_t *v[4];
                  for (int r = 0; r < 4; ++r) {
                    std::fill(storage[r], storage[r] + 34, 0xdeadbeef);
                    v[r] = storage[r] + 1;
                  }
                  for (int lane = 0; lane < 32; ++lane) {
                    int ix[4] = {lane % 4, (lane / 4) % 4, (lane + 1) % 4, (lane / 4 + 2) % 4};
                    v[0][lane] = codes[ix[0]] | (uint32_t(codes[ix[1]]) << 16);
                    v[1][lane] = codes[ix[2]] | (uint32_t(codes[ix[3]]) << 16);
                    int c = lane - 16;
                    v[2][lane] = goc::as_bits(float(c));
                    int x0 = values[ix[selection & 1 ? 1 : 0]];
                    int y0 = values[ix[selection & 2 ? 3 : 2]];
                    int x1 = values[ix[selection & 4 ? 0 : 1]];
                    int y1 = values[ix[selection & 8 ? 2 : 3]];
                    if (negate & 1)
                      x0 = -x0;
                    if (negate & 2)
                      y0 = -y0;
                    if (negate & 4)
                      c = -c;
                    if (negate & 8)
                      x1 = -x1;
                    if (negate & 16)
                      y1 = -y1;
                    expected[lane] = goc::as_bits(float(x0 * y0 + x1 * y1 + c));
                  }
                  std::copy(v[alias], v[alias] + 32, before);
                  ASSERT_EQ(fn(cpu | semantics | GOC_SEMANTICS_STRICT, mask, mode, &v[alias], &v[0],
                               &v[1], &v[2]),
                            GOC_SUCCESS);
                  for (int lane = 0; lane < 32; ++lane)
                    EXPECT_EQ(v[alias][lane], (mask >> lane & 1) ? expected[lane] : before[lane]);
                  for (const auto &reg : storage) {
                    EXPECT_EQ(reg[0], 0xdeadbeef);
                    EXPECT_EQ(reg[33], 0xdeadbeef);
                  }
                }
        }
  }
}

TEST(Dot2, ExactModifiersRecoverHardwareGoldens) {
  const auto check = [](const auto &cases, auto fn) {
    for (const auto &f : cases)
      for (uint32_t neg = 0; neg < 32; ++neg)
        for (uint32_t swap = 0; swap < 4; ++swap) {
          uint32_t a[32], b[32], c[32], d[32];
          // Undo signs and invert the half permutation so the modified operands
          // reproduce the hardware-captured original dot exactly.
          uint32_t av = f.a ^ (neg & 1 ? 0x8000 : 0) ^ (neg & 8 ? 0x80000000 : 0);
          uint32_t bv = f.b ^ (neg & 2 ? 0x8000 : 0) ^ (neg & 16 ? 0x80000000 : 0);
          uint32_t mode = neg | GOC_DOT_CLAMP;
          if (swap & 1) {
            av = (av << 16) | (av >> 16);
            mode |= GOC_DOT_LO_A_HIGH | GOC_DOT_HI_A_LOW;
          }
          if (swap & 2) {
            bv = (bv << 16) | (bv >> 16);
            mode |= GOC_DOT_LO_B_HIGH | GOC_DOT_HI_B_LOW;
          }
          std::fill(a, a + 32, av);
          std::fill(b, b + 32, bv);
          std::fill(c, c + 32, f.c ^ (neg & 4 ? 0x80000000 : 0));
          auto pa = a, pb = b, pc = c, pd = d;
          ASSERT_EQ(fn(goc_init_cpu_flags() | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                       UINT32_MAX, mode, &pd, &pa, &pb, &pc),
                    GOC_SUCCESS);
          for (auto word : d)
            EXPECT_EQ(word, f.expected);
        }
  };
  check(kGfx12DotF16Cases, goc_rdna4_v_dot2_f32_f16);
  check(kGfx12DotBF16Cases, goc_rdna4_v_dot2_f32_bf16);
}

TEST(Dot2, SimdSpecialValuesAndEveryExecMask) {
  const uint32_t a16[] = {0x80008000, 0x00010001, 0x7c003c00, 0x7e003c00, 0x3c013c01, 0x7bff7bff};
  const uint32_t b16[] = {0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3bfe3bfe, 0x04000400};
  const uint32_t abf[] = {0x80008000, 0x00010001, 0x7f803f80, 0x7fc03f80, 0x3f813f81, 0x7f7f7f7f};
  const uint32_t bbf[] = {0x3f803f80, 0x3f803f80, 0x3f803f80, 0x3f803f80, 0x3f7e3f7e, 0x00800080};
  for (bool brain : {false, true})
    for (uint32_t mode :
         {UINT32_C(0), GOC_DOT_NEG_C | GOC_DOT_NEG_LO_A | GOC_DOT_LO_B_HIGH | GOC_DOT_HI_A_LOW})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t mask : rdna4_exec_masks()) {
          auto fn = brain ? goc_rdna4_v_dot2_f32_bf16 : goc_rdna4_v_dot2_f32_f16;
          uint32_t a[32], b[32], c[32], ref[32], d[32];
          for (int i = 0; i < 32; ++i) {
            a[i] = (brain ? abf : a16)[i % 6];
            b[i] = (brain ? bbf : b16)[i % 6];
            c[i] = i % 6 == 0 ? 0x80000000 : 0xbf800000;
            ref[i] = d[i] = 0xdeadbeef;
          }
          auto pa = a, pb = b, pc = c, pr = ref, pd = d;
          ASSERT_EQ(fn(GOC_CPU_BASELINE, mask, mode, &pr, &pa, &pb, &pc), GOC_SUCCESS);
          ASSERT_EQ(fn(cpu, mask, mode, &pd, &pa, &pb, &pc), GOC_SUCCESS);
          for (int i = 0; i < 32; ++i) {
            if (std::isnan(goc::as_float(ref[i]))) {
              EXPECT_TRUE(std::isnan(goc::as_float(d[i])));
            } else {
              EXPECT_EQ(d[i], ref[i]);
            }
          }
        }
}

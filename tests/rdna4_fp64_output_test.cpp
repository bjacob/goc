// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_fp64.h"
#include "rdna4_fp64_output_hardware.h"

#include <cmath>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

int call(unsigned op, uint64_t cpu, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *b, const uint32_t *const *c) {
  switch (op) {
  case 0:
    return goc_rdna4_v_add_f64(cpu, UINT32_MAX, mode, d, a, b);
  case 1:
    return goc_rdna4_v_mul_f64(cpu, UINT32_MAX, mode, d, a, b);
  case 2:
    return goc_rdna4_v_fma_f64(cpu, UINT32_MAX, mode, d, a, b, c);
  case 3:
    return goc_rdna4_v_trunc_f64(cpu, UINT32_MAX, mode, d, a);
  case 4:
    return goc_rdna4_v_ceil_f64(cpu, UINT32_MAX, mode, d, a);
  case 5:
    return goc_rdna4_v_rndne_f64(cpu, UINT32_MAX, mode, d, a);
  case 6:
    return goc_rdna4_v_floor_f64(cpu, UINT32_MAX, mode, d, a);
  case 7:
    return goc_rdna4_v_fract_f64(cpu, UINT32_MAX, mode, d, a);
  case 8:
    return goc_rdna4_v_sqrt_f64(cpu, UINT32_MAX, mode, d, a);
  case 9:
    return goc_rdna4_v_rcp_f64(cpu, UINT32_MAX, mode, d, a);
  case 10:
    return goc_rdna4_v_rsq_f64(cpu, UINT32_MAX, mode, d, a);
  case 11:
    return goc_rdna4_v_min_num_f64(cpu, UINT32_MAX, mode, d, a, b);
  case 12:
    return goc_rdna4_v_max_num_f64(cpu, UINT32_MAX, mode, d, a, b);
  case 13:
    return goc_rdna4_v_minimum_f64(cpu, UINT32_MAX, mode, d, a, b);
  case 14:
    return goc_rdna4_v_maximum_f64(cpu, UINT32_MAX, mode, d, a, b);
  case 15:
    return goc_rdna4_v_frexp_mant_f64(cpu, UINT32_MAX, mode, d, a);
  }
  return GOC_ERROR_INVALID_FLAGS;
}

} // namespace

TEST(Fp64Output, HardwareModifiersAndCrossHalfAliases) {
  const unsigned aliases[][2] = {{6, 7}, {0, 1}, {1, 0}, {2, 3}, {4, 5}, {1, 2}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 16; ++op)
      for (const auto &alias : aliases) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << '/' << op << '/' << alias[0] << '/' << alias[1]);
        uint64_t hash = UINT64_C(14695981039346656037);
        for (unsigned v = 0; v < 16; ++v) {
          uint32_t words[8][32] = {};
          uint32_t *p[8];
          for (unsigned reg = 0; reg < 8; ++reg)
            p[reg] = words[reg];
          uint32_t *d[] = {p[alias[0]], p[alias[1]]};
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t raw = goc_test::fp64_omod_inputs[lane];
            words[0][lane] = uint32_t(raw);
            words[1][lane] = uint32_t(raw >> 32);
            words[3][lane] = op == 0 ? 0x80000000 : 0x3ff00000;
            words[5][lane] = 0x80000000;
          }
          uint64_t mode = ((v >> 2) << 6) | (v & 2 ? GOC_ALU_CLAMP : 0) | (v & 1);
          ASSERT_EQ(call(op, cpu, mode, d, p, p + 2, p + 4), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t raw = d[0][lane] | (uint64_t(d[1][lane]) << 32);
            if (op >= 8 && op <= 10) {
              uint64_t want = goc_test::fp64_transcendental_hardware[op - 8][v][lane];
              uint64_t magnitude = want & UINT64_C(0x7fffffffffffffff);
              if (magnitude > UINT64_C(0x7ff0000000000000)) {
                EXPECT_TRUE(std::isnan(goc::as_double(raw)));
              } else if (!magnitude || magnitude == UINT64_C(0x7ff0000000000000)) {
                EXPECT_EQ(raw, want);
              } else {
                // RDNA4 FP64 transcendental approximations have substantially
                // less than binary64 precision; preserve sign and classification.
                EXPECT_EQ(raw >> 63, want >> 63);
                EXPECT_EQ(std::fpclassify(goc::as_double(raw)),
                          std::fpclassify(goc::as_double(want)));
                EXPECT_NEAR(goc::as_double(raw) / goc::as_double(want), 1.0, 0x1p-23);
              }
            }
            if ((raw & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000))
              raw = UINT64_C(0x7ff8000000000000);
            for (unsigned shift = 0; shift < 64; shift += 8)
              hash = (hash ^ ((raw >> shift) & 255)) * UINT64_C(1099511628211);
          }
        }
        if (op < 8 || op > 10) {
          EXPECT_EQ(hash, goc_test::fp64_omod_hashes[op]);
        }
      }
}

TEST(Fp64Output, LdexpHardwareBoundaries) {
  // Same gfx1201 capture and inputs as the arithmetic corpus, exponent 0/-1.
  const uint64_t hashes[] = {UINT64_C(0x91b37a3decfcf2a5), UINT64_C(0xad1830abf629d830)};
  const unsigned aliases[][2] = {{3, 4}, {0, 1}, {1, 0}, {2, 0}, {1, 2}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned set = 0; set < 2; ++set)
      for (const auto &alias : aliases) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << '/' << set << '/' << alias[0] << '/' << alias[1]);
        uint64_t hash = UINT64_C(14695981039346656037);
        for (unsigned v = 0; v < 16; ++v) {
          uint32_t words[5][32] = {};
          uint32_t *p[] = {words[0], words[1], words[2], words[3], words[4]};
          uint32_t *d[] = {p[alias[0]], p[alias[1]]};
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t raw = goc_test::fp64_omod_inputs[lane];
            words[0][lane] = uint32_t(raw);
            words[1][lane] = uint32_t(raw >> 32);
            words[2][lane] = set ? UINT32_MAX : 0;
          }
          uint64_t mode = ((v >> 2) << 6) | (v & 2 ? GOC_ALU_CLAMP : 0) | (v & 1);
          ASSERT_EQ(goc_rdna4_v_ldexp_f64(cpu, UINT32_MAX, mode, d, p, p + 2), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t raw = d[0][lane] | (uint64_t(d[1][lane]) << 32);
            if ((raw & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000))
              raw = UINT64_C(0x7ff8000000000000);
            for (unsigned shift = 0; shift < 64; shift += 8)
              hash = (hash ^ ((raw >> shift) & 255)) * UINT64_C(1099511628211);
          }
        }
        EXPECT_EQ(hash, hashes[set]);
      }
}

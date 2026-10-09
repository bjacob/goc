// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_swmmac8_hardware.h"

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_swmmac_f32_16x16x32_fp8_fp8);
const Fn functions[] = {
    goc_rdna4_v_swmmac_f32_16x16x32_fp8_fp8, goc_rdna4_v_swmmac_f32_16x16x32_fp8_bf8,
    goc_rdna4_v_swmmac_f32_16x16x32_bf8_fp8, goc_rdna4_v_swmmac_f32_16x16x32_bf8_bf8};

uint32_t mode(unsigned variant) { return variant ? GOC_SWMMAC_INDEX_KEY_1 : 0; }

struct Registers {
  uint32_t data[15][32];
  const uint32_t *a[2], *b[4], *index[1];
  uint32_t *d[8];

  explicit Registers(unsigned op, unsigned sample = 0) {
    goc_test::swmmac8_capture_inputs(op, sample, data);
    for (unsigned reg = 0; reg < 2; ++reg)
      a[reg] = data[reg];
    for (unsigned reg = 0; reg < 4; ++reg)
      b[reg] = data[2 + reg];
    for (unsigned reg = 0; reg < 8; ++reg)
      d[reg] = data[6 + reg];
    index[0] = data[14];
  }
};

} // namespace

TEST(Swmmac8, HardwareAllModifiers) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 2; ++variant)
        for (unsigned sample = 0; sample < 16; ++sample) {
          Registers r(op, sample);
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, mode(variant), r.d, r.a, r.b, r.index),
                    GOC_SUCCESS);
          uint64_t digest = UINT64_C(14695981039346656037);
          for (unsigned reg = 0; reg < 8u; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              for (unsigned byte = 0; byte < 4; ++byte) {
                digest ^= (r.d[reg][lane] >> (8 * byte)) & 255;
                digest *= UINT64_C(1099511628211);
              }
          ASSERT_EQ(digest, goc_test::swmmac8_capture_digests[op][sample][variant])
              << op << "/" << cpu << "/" << variant;
        }
}

TEST(Swmmac8, EveryExecLaneAndModifier) {
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant = 0; variant < 2; ++variant) {
      Registers full(op), initial(op);
      ASSERT_EQ(functions[op](0, UINT32_MAX, mode(variant), full.d, full.a, full.b, full.index),
                GOC_SUCCESS);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t mask : rdna4_exec_masks()) {
          Registers r(op);
          ASSERT_EQ(functions[op](cpu, mask, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
          for (unsigned reg = 0; reg < 8u; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(r.d[reg][lane],
                        ((mask >> lane) & 1) ? full.d[reg][lane] : initial.d[reg][lane])
                  << op << "/" << cpu << "/" << variant << "/" << mask;
        }
    }
}

TEST(Swmmac8, OverlappingSourcesAndDuplicateDestinations) {
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant = 0; variant < 2; ++variant)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned alias = 0; alias < 4; ++alias) {
          Registers r(op), reference(op);
          unsigned regs = 8;
          unsigned offsets[8];
          for (unsigned reg = 0; reg < regs; ++reg) {
            offsets[reg] = alias == 0 ? reg % 2 : alias == 1 ? 2 + reg % 4 : alias == 2 ? 14 : 6;
            r.d[reg] = r.data[offsets[reg]];
            std::memcpy(reference.d[reg], r.d[reg], 32 * sizeof(uint32_t));
          }
          const uint32_t mask = 0xa35ac69d;
          ASSERT_EQ(functions[op](0, mask, mode(variant), reference.d, reference.a, reference.b,
                                  reference.index),
                    GOC_SUCCESS);
          uint32_t expected[15][32];
          std::memcpy(expected, r.data, sizeof(expected));
          for (unsigned reg = 0; reg < regs; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              if ((mask >> lane) & 1)
                expected[offsets[reg]][lane] = reference.d[reg][lane];
          ASSERT_EQ(functions[op](cpu, mask, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
          ASSERT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0)
              << op << "/" << cpu << "/" << variant << "/" << alias;
        }
}

TEST(Swmmac8, ValidationAndEmptyExec) {
  for (Fn fn : functions) {
    EXPECT_EQ(fn(0, UINT32_C(0), 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, nullptr, nullptr, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr,
                 nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (unsigned bit = 0; bit < 32; ++bit)
      if (!((1u << bit) & mode(1))) {
        EXPECT_EQ(fn(0, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
      }
  }
}

TEST(Swmmac8, EveryInputEncoding) {
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned source = 0; source < 2; ++source)
      for (unsigned code = 0; code < 256; ++code) {
        bool bf8 = source ? op % 2 : op / 2;
        unsigned mantissa_bits = bf8 ? 2 : 3;
        unsigned exponent = (code & 127) >> mantissa_bits;
        unsigned fraction = code & ((1u << mantissa_bits) - 1);
        float value;
        if (bf8 && exponent == 31)
          value = fraction ? std::numeric_limits<float>::quiet_NaN()
                           : std::numeric_limits<float>::infinity();
        else if (!bf8 && exponent == 15 && fraction == 7)
          value = std::numeric_limits<float>::quiet_NaN();
        else
          value = std::ldexp(float(fraction + (exponent ? 1u << mantissa_bits : 0)),
                             int(exponent ? exponent : 1) - (bf8 ? 15 : 7) - int(mantissa_bits));
        if (code & 128)
          value = -value;
        float want = std::fma(16.0f, value, 0.0f);
        uint32_t want_bits;
        std::memcpy(&want_bits, &want, sizeof(want));
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          Registers r(op);
          unsigned av = source == 0 ? code : op / 2 ? 0x3c : 0x38;
          unsigned bv = source == 1 ? code : op % 2 ? 0x3c : 0x38;
          for (unsigned reg = 0; reg < 6; ++reg)
            for (auto &word : r.data[reg])
              word = (reg < 2 ? av : bv) * 0x01010101u;
          for (unsigned reg = 6; reg < 14; ++reg)
            for (auto &word : r.data[reg])
              word = 0;
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, r.d, r.a, r.b, r.index), GOC_SUCCESS);
          for (unsigned reg = 0; reg < 8; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane) {
              if (std::isnan(want)) {
                EXPECT_GT(r.d[reg][lane] & 0x7fffffffu, 0x7f800000u);
              } else {
                EXPECT_EQ(r.d[reg][lane], want_bits) << op << "/" << source << "/" << code;
              }
            }
        }
      }
}

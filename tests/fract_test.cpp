// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "fp_environment.h"
#include "fract_hardware.h"
#include "goc/goc.h"
#include "unary_capture_input.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

uint32_t capture_input(unsigned corpus, unsigned i) {
  if (!corpus)
    return goc_test::unary_capture_input(0, i);
  const int offsets[] = {0, 1, -1, 0x400000};
  uint32_t word = (((i / 32) % 254 + 1) << 23) + offsets[i % 4];
  return word | (((i % 32) / 4 % 2) << 31);
}

int invoke(bool rdna3, bool wide, uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *d,
           const uint32_t *a, uint32_t *state) {
  if (wide)
    return (rdna3 ? goc_v_fract_f32_wave64 : goc_v_fract_f32_rdna4_wave64)(flags, exec_mask, mode,
                                                                           &d, &a, state);
  return (rdna3 ? goc_v_fract_f32 : goc_v_fract_f32_rdna4)(flags, uint32_t(exec_mask), mode, &d, &a,
                                                           state);
}

uint32_t modifiers(unsigned m) {
  return (m & 1 ? GOC_ALU_NEG_A : 0) | (m & 2 ? GOC_ALU_ABS_A : 0) | ((m >> 2 & 3) << 6) |
         (m & 16 ? GOC_ALU_CLAMP : 0);
}

} // namespace

TEST(Fract, HardwareNumericsAndArchitecturePolicy) {
  for (unsigned corpus = 0; corpus < 2; ++corpus)
    for (bool exact : {false})
      for (bool rdna3 : {false, true})
        for (bool wide : {false, true})
          for (unsigned policy = 0; policy < 8; ++policy)
            for (unsigned m = 0; m < 32; ++m) {
              unsigned lanes = wide ? 64 : 32;
              uint64_t hash = goc_test::capture_hash_seed;
              for (unsigned base = 0; base < 8192; base += lanes) {
                uint32_t input[64], output[64];
                for (unsigned lane = 0; lane < lanes; ++lane)
                  input[lane] = capture_input(corpus, base + lane);
                uint64_t flags =
                    ((base / lanes) % (goc_init_cpu_flags() + 1)) |
                    (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0) |
                    (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                    (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                    (policy & 4 ? GOC_FP_IEEE_MODE : 0);
                ASSERT_EQ(
                    invoke(rdna3, wide, flags, UINT64_MAX, modifiers(m), output, input, nullptr),
                    GOC_SUCCESS);
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  uint32_t word = output[lane];
                  if ((word & 0x7fffffff) > 0x7f800000)
                    word = 0x7fc00000;
                  hash = goc_test::capture_hash_word(hash, word);
                }
              }
              unsigned effective = rdna3 && ((policy & 4) || !(policy & 2)) ? m & ~12U : m;
              EXPECT_EQ(hash, goc_test::fract_hardware[corpus][policy & 3][effective][0])
                  << exact << '/' << rdna3 << '/' << wide << '/' << policy << '/' << m;
            }
}

TEST(Fract, RemovedExactPathsRejectAndLooseStillWorks) {
  const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  const decltype(&goc_v_fract_f32) narrow[] = {goc_v_fract_f32, goc_v_fract_f32_rdna4,
                                               goc_v_fract_f16, goc_v_fract_f16_rdna4};
  const decltype(&goc_v_fract_f32_wave64) wide[] = {
      goc_v_fract_f32_wave64, goc_v_fract_f32_rdna4_wave64, goc_v_fract_f16_wave64,
      goc_v_fract_f16_rdna4_wave64};
  for (unsigned op = 0; op < 4; ++op)
    for (bool wave64 : {false, true})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        uint32_t input[64], output[64], state = 0x80000040;
        std::fill(input, input + 64, op < 2 ? 0x3fc00000U : 0x3e00U);
        std::fill(output, output + 64, 0xdeadbeef);
        const uint32_t *a = input;
        uint32_t *d = output;
        auto call = [&](uint64_t flags, uint64_t exec_mask, uint32_t *report) {
          return wave64 ? wide[op](flags | cpu, exec_mask, 0, &d, &a, report)
                        : narrow[op](flags | cpu, uint32_t(exec_mask), 0, &d, &a, report);
        };
        for (uint64_t exec_mask : {0ULL, 1ULL, ~0ULL}) {
          EXPECT_EQ(call(exact, exec_mask, nullptr), GOC_ERROR_UNSUPPORTED_SEMANTICS);
          EXPECT_EQ(call(exact, exec_mask, &state), GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
          EXPECT_EQ(state, 0x80000040U);
          EXPECT_TRUE(
              std::all_of(output, output + 64, [](uint32_t word) { return word == 0xdeadbeef; }));
        }
        ASSERT_EQ(call(0, 1, &state), GOC_SUCCESS);
        EXPECT_EQ(output[0], op < 2 ? 0x3f000000U : 0xdead3800U);
        EXPECT_EQ(output[1], 0xdeadbeefU);
        EXPECT_EQ(state, 0x80000040U);
        output[0] = 0xdeadbeef;
        ASSERT_EQ(call(GOC_SEMANTICS_EXACT_EMPIRICAL, 1, nullptr), GOC_SUCCESS);
        EXPECT_EQ(output[0], op < 2 ? 0x3f000000U : 0xdead3800U);
      }
}

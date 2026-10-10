// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "fract_hardware.h"
#include "goc/goc.h"
#include "unary_capture_input.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
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

TEST(Fract, HardwareNumericsAndLooseFallback) {
  for (unsigned corpus = 0; corpus < 2; ++corpus)
    for (bool fallback : {false, true})
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
                uint64_t flags = ((base / lanes) % (goc_init_cpu_flags() + 1)) |
                                 (fallback ? GOC_SEMANTICS_EXACT_EMPIRICAL : 0) |
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
                  << fallback << '/' << rdna3 << '/' << wide << '/' << policy << '/' << m;
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
          EXPECT_EQ(call(GOC_SEMANTICS_EXACT_EMPIRICAL, exec_mask, &state),
                    GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
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

TEST(Fract, FractionBoundsAndGuestDenormals) {
  // Paired FP32/FP16 encodings with literal, nearest-even fractional results.
  // A negative tiny input must stay below one rather than round up to one.
  const uint32_t inputs[2][12] = {
      {0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x3fa00000,
       0xbfa00000, 0x3f000000, 0xbf000000},
      {0, 0x8000, 1, 0x8001, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x3d00, 0xbd00, 0x3800, 0xb800}};
  const uint32_t fractions[2][12] = {
      {0, 0, 1, 0x3f7fffff, 0x00800000, 0x3f7fffff, 0, 0, 0x3e800000, 0x3f400000, 0x3f000000,
       0x3f000000},
      {0, 0, 1, 0x3bff, 0x0400, 0x3bff, 0, 0, 0x3400, 0x3a00, 0x3800, 0x3800}};
  const decltype(&goc_v_fract_f32) narrow[2][2] = {{goc_v_fract_f32, goc_v_fract_f32_rdna4},
                                                   {goc_v_fract_f16, goc_v_fract_f16_rdna4}};
  const decltype(&goc_v_fract_f32_wave64) wide[2][2] = {
      {goc_v_fract_f32_wave64, goc_v_fract_f32_rdna4_wave64},
      {goc_v_fract_f16_wave64, goc_v_fract_f16_rdna4_wave64}};
  for (unsigned half = 0; half < 2; ++half)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (bool wave64 : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (bool fallback : {false, true})
            for (unsigned policy = 0; policy < 8; ++policy)
              for (unsigned halves = 0; halves < (half ? 4U : 1U); ++halves)
                for (uint64_t exec_mask : {UINT64_MAX, uint64_t(0xa5a571abcdef9192ULL)}) {
                  const unsigned lanes = wave64 ? 64 : 32;
                  const unsigned a_shift = halves & 1 ? 16 : 0, d_shift = halves & 2 ? 16 : 0;
                  uint32_t values[64], expected[64];
                  for (unsigned lane = 0; lane < 64; ++lane) {
                    const unsigned i = lane % 12;
                    values[lane] = half ? (inputs[half][i] << a_shift) | (0x1234U << (16 - a_shift))
                                        : inputs[half][i];
                    expected[lane] = values[lane];
                    if (lane >= lanes || !((exec_mask >> lane) & 1))
                      continue;
                    uint32_t result = fractions[half][i];
                    if (((policy & 1) && (i == 2 || i == 3)) || ((policy & 2) && i == 2))
                      result = 0;
                    expected[lane] =
                        half ? (values[lane] & ~(65535U << d_shift)) | (result << d_shift) : result;
                  }
                  const uint64_t flags = cpu | (fallback ? GOC_SEMANTICS_EXACT_EMPIRICAL : 0) |
                                         (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                         (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                                         (policy & 4 ? GOC_FP_IEEE_MODE : 0);
                  const uint64_t mode =
                      (a_shift ? GOC_ALU_HIGH_A : 0) | (d_shift ? GOC_ALU_HIGH_D : 0);
                  uint32_t *d = values, state = 0x80000055;
                  const uint32_t *a = values;
                  ASSERT_EQ(wave64 ? wide[half][arch](flags, exec_mask, mode, &d, &a,
                                                      fallback ? nullptr : &state)
                                   : narrow[half][arch](flags, uint32_t(exec_mask), mode, &d, &a,
                                                        fallback ? nullptr : &state),
                            GOC_SUCCESS);
                  EXPECT_EQ(state, 0x80000055U);
                  EXPECT_EQ(std::memcmp(values, expected, sizeof values), 0)
                      << half << '/' << arch << '/' << wave64 << '/' << cpu << '/' << policy;
                }
}

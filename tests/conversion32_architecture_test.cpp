// SPDX-License-Identifier: MIT

#include "conversion32_reference.h"
#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "test_instruction.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn = goc_test::WaveInstruction<decltype(&goc_v_cvt_f32_i32)>;
const Fn functions[2][2][6] = {
    {{goc_v_cvt_f32_i32, goc_v_cvt_f32_u32, goc_v_cvt_i32_f32, goc_v_cvt_u32_f32,
      goc_v_cvt_nearest_i32_f32, goc_v_cvt_floor_i32_f32},
     {goc_v_cvt_f32_i32_wave64, goc_v_cvt_f32_u32_wave64, goc_v_cvt_i32_f32_wave64,
      goc_v_cvt_u32_f32_wave64, goc_v_cvt_nearest_i32_f32_wave64, goc_v_cvt_floor_i32_f32_wave64}},
    {{goc_v_cvt_f32_i32_rdna4, goc_v_cvt_f32_u32_rdna4, goc_v_cvt_i32_f32_rdna4,
      goc_v_cvt_u32_f32_rdna4, goc_v_cvt_nearest_i32_f32, goc_v_cvt_floor_i32_f32},
     {goc_v_cvt_f32_i32_rdna4_wave64, goc_v_cvt_f32_u32_rdna4_wave64,
      goc_v_cvt_i32_f32_rdna4_wave64, goc_v_cvt_u32_f32_rdna4_wave64,
      goc_v_cvt_nearest_i32_f32_wave64, goc_v_cvt_floor_i32_f32_wave64}}};

uint64_t policy(unsigned bits) {
  return (bits & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (bits & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (bits & 4 ? GOC_FP_IEEE_MODE : 0);
}

} // namespace

TEST(Conversion32Architecture, OutputScalingLiteralWitness) {
  // rocjitsu fp_mode::effective_omod: ordinary RDNA3 output scaling requires
  // output flush and IEEE=0. Integer 3 converts exactly on every CPU path.
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t input[64], output[64];
            std::fill(input, input + 64, 3);
            std::fill(output, output + 64, 0xdeadbeef);
            const uint32_t *a = input;
            uint32_t *d = output;
            ASSERT_EQ(
                functions[arch][wide][op](cpu | policy(guest), UINT64_MAX, GOC_ALU_OMOD_2, &d, &a),
                GOC_SUCCESS);
            bool scale = arch || ((guest & 2) && !(guest & 4));
            for (unsigned lane = 0; lane < 64; ++lane)
              EXPECT_EQ(output[lane], lane < (wide ? 64U : 32U) ? scale ? 0x40c00000U : 0x40400000U
                                                                : 0xdeadbeefU);
          }
}

TEST(Conversion32Architecture, GuestPoliciesModifiersDppMasksAndAliases) {
  const uint32_t edge[] = {0,          0x80000000, 1,          0x80000001, 0x7fffff,   0x800000,
                           0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000, 0x4f000000, 0xcf000000,
                           0x4f800000, 0x7f800000, 0xff800000, 0x7f800123, 0xffc00055, 16777217,
                           16777219,   0xffffffff, 0x7fffffff};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 6; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned m = 0; m < goc_test::conversion32_modes(op); ++m)
            for (unsigned routing = 0; routing < 8; ++routing)
              for (bool alias : {false, true}) {
                uint64_t descriptor = routing ? goc_test::dpp_modes[routing - 1] : 0;
                uint32_t mode = goc_test::conversion32_mode(op, m);
                uint32_t effective = !arch && op < 2 && (!(guest & 2) || (guest & 4))
                                         ? mode & ~GOC_ALU_OMOD_HALF
                                         : mode;
                const uint64_t exec_mask = wide ? 0xc5912d6ea3157bf1ULL : 0xa3157bf1U;
                uint32_t original[66], input[66], separate[66], expected[66];
                for (unsigned lane = 0; lane < 66; ++lane) {
                  original[lane] = lane < 32 ? edge[lane % 21] : lane * 0x7395a831U;
                  input[lane] = original[lane];
                  separate[lane] = 0xdeadbeef;
                  expected[lane] = alias ? original[lane] : separate[lane];
                }
                for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
                  int source = int(lane);
                  bool write = descriptor
                                   ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                   : ((exec_mask >> lane) & 1);
                  if (write)
                    expected[lane + 1] = goc_test::conversion32_reference(
                        op, source < 0 ? 0 : original[source + 1], effective);
                }
                for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
                  std::copy(original, original + 66, input);
                  std::fill(separate, separate + 66, 0xdeadbeef);
                  uint32_t *d = (alias ? input : separate) + 1;
                  const uint32_t *a = input + 1;
                  uint32_t state = 0x80000055;
                  ASSERT_EQ(functions[arch][wide][op](cpu | policy(guest), exec_mask,
                                                      mode | descriptor, &d, &a,
                                                      op == 2 || op == 3 ? &state : nullptr),
                            GOC_SUCCESS);
                  ASSERT_TRUE(std::equal(d - 1, d + 65, expected))
                      << arch << '/' << wide << '/' << op << '/' << guest << '/' << m << '/'
                      << routing << '/' << alias << '/' << cpu;
                  EXPECT_EQ(state, 0x80000055U);
                }
              }
}

TEST(Conversion32Architecture, ValidationStateAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
#if defined(__x86_64__) || defined(_M_X64)
  _mm_setcsr(_mm_getcsr() & ~0x8040U);
#endif
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 6; ++op)
        for (unsigned guest = 0; guest < 8; ++guest) {
          uint32_t input[64], output[64], state = 0x80000055;
          std::fill(input, input + 64, 3);
          std::fill(output, output + 64, 0xdeadbeef);
          const uint32_t *a = input;
          uint32_t *d = output;
          auto fn = functions[arch][wide][op];
          for (uint64_t exec_mask : {0ULL, ~0ULL}) {
            EXPECT_EQ(fn(policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                         exec_mask, 0, &d, &a),
                      GOC_ERROR_UNSUPPORTED_SEMANTICS);
            EXPECT_EQ(fn(policy(guest), exec_mask, GOC_ALU_NEG_B, &d, &a), GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(fn(policy(guest) | (1ULL << 63), exec_mask, 0, &d, &a),
                      GOC_ERROR_INVALID_FLAGS);
            if (op == 2 || op == 3) {
              for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
                EXPECT_EQ(fn(policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | strict, exec_mask, 0,
                             &d, &a, &state),
                          GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
            }
            EXPECT_EQ(state, 0x80000055U);
            EXPECT_TRUE(
                std::all_of(output, output + 64, [](uint32_t v) { return v == 0xdeadbeef; }));
          }
          unsigned mode = op < 4 ? GOC_ALU_OMOD_4 : 0;
#if defined(__x86_64__) || defined(_M_X64)
          unsigned controls = _mm_getcsr();
#endif
          ASSERT_EQ(
              fn(policy(guest), UINT64_MAX, mode, &d, &a, op == 2 || op == 3 ? &state : nullptr),
              GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), FE_TONEAREST);
#if defined(__x86_64__) || defined(_M_X64)
          EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls & ~0x3fU);
#endif
          EXPECT_EQ(state, 0x80000055U);
        }
}

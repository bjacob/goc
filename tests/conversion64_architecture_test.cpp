// SPDX-License-Identifier: MIT

#include "conversion64_reference.h"
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

using Fn = goc_test::WaveInstruction<decltype(&goc_v_cvt_f64_i32)>;
const Fn functions[2][2][6] = {
    {{goc_v_cvt_f64_i32, goc_v_cvt_f64_u32, goc_v_cvt_i32_f64, goc_v_cvt_u32_f64, goc_v_cvt_f64_f32,
      goc_v_cvt_f32_f64},
     {goc_v_cvt_f64_i32_wave64, goc_v_cvt_f64_u32_wave64, goc_v_cvt_i32_f64_wave64,
      goc_v_cvt_u32_f64_wave64, goc_v_cvt_f64_f32_wave64, goc_v_cvt_f32_f64_wave64}},
    {{goc_v_cvt_f64_i32_rdna4, goc_v_cvt_f64_u32_rdna4, goc_v_cvt_i32_f64_rdna4,
      goc_v_cvt_u32_f64_rdna4, goc_v_cvt_f64_f32_rdna4, goc_v_cvt_f32_f64_rdna4},
     {goc_v_cvt_f64_i32_rdna4_wave64, goc_v_cvt_f64_u32_rdna4_wave64,
      goc_v_cvt_i32_f64_rdna4_wave64, goc_v_cvt_u32_f64_rdna4_wave64,
      goc_v_cvt_f64_f32_rdna4_wave64, goc_v_cvt_f32_f64_rdna4_wave64}}};

uint64_t policy(unsigned bits) {
  return (bits & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (bits & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (bits & 4 ? GOC_FP_IEEE_MODE : 0);
}

} // namespace

TEST(Conversion64Architecture, OutputScalingLiteralWitness) {
  // RDNA3 ISA 7.2.3.1 and rocjitsu fp_mode::effective_omod.
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned guest = 0; guest < 8; ++guest)
        for (unsigned op : {0U, 1U, 4U, 5U})
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t input[2][64], output[2][64];
            for (unsigned lane = 0; lane < 64; ++lane) {
              input[0][lane] = op < 2 ? 3 : op == 4 ? 0x40400000 : 0;
              input[1][lane] = 0x40080000;
              output[0][lane] = output[1][lane] = 0xdeadbeef;
            }
            const uint32_t *a[] = {input[0], op == 5 ? input[1] : nullptr};
            uint32_t *d[] = {output[0], op == 5 ? nullptr : output[1]};
            ASSERT_EQ(
                functions[arch][wide][op](cpu | policy(guest), UINT64_MAX, GOC_ALU_OMOD_2, d, a),
                GOC_SUCCESS);
            bool scale = arch || ((guest & 2) && !(guest & 4));
            for (unsigned lane = 0; lane < 64; ++lane) {
              bool active = lane < (wide ? 64U : 32U);
              EXPECT_EQ(output[0][lane], !active   ? 0xdeadbeefU
                                         : op == 5 ? (scale ? 0x40c00000U : 0x40400000U)
                                                   : 0U);
              EXPECT_EQ(output[1][lane], !active || op == 5 ? 0xdeadbeefU
                                         : scale            ? 0x40180000U
                                                            : 0x40080000U);
            }
          }
}

TEST(Conversion64Architecture, GuestPoliciesModifiersMasksAndAliases) {
  const uint64_t edge[] = {0,
                           0x8000000000000000ULL,
                           1,
                           0x8000000000000001ULL,
                           0xfffffffffffffULL,
                           0x10000000000000ULL,
                           0x380fffffffffffffULL,
                           0x3810000000000000ULL,
                           0x3ff0000000000000ULL,
                           0xbff8000000000000ULL,
                           0x41e0000000000000ULL,
                           0x41f0000000000000ULL,
                           0x47effffff0000000ULL,
                           0x7ff0000000000000ULL,
                           0xfff0000000000000ULL,
                           0xffffffff,
                           0x3f800000,
                           0x80000001,
                           0x800000,
                           0x7f800000,
                           0xff800000};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 6; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned m = 0; m < goc_test::conversion64_modes(op); ++m)
            for (unsigned alias = 0; alias < 3; ++alias) {
              uint32_t mode = goc_test::conversion64_mode(op, m);
              uint32_t effective = !arch && op != 2 && op != 3 && (!(guest & 2) || (guest & 4))
                                       ? mode & ~GOC_ALU_OMOD_HALF
                                       : mode;
              const uint64_t exec_mask = wide ? 0xc5912d6ea3157bf1ULL : 0xa3157bf1U;
              unsigned outputs = goc_test::conversion64_wide_output(op) ? 2 : 1;
              unsigned destinations[] = {alias == 1 ? 1U : 2U, alias == 1   ? 0U
                                                               : alias == 2 ? 2U
                                                                            : 3U};
              uint32_t original[4][66], expected[4][66], actual[4][66];
              for (unsigned lane = 0; lane < 66; ++lane) {
                uint64_t raw = lane < 32 ? edge[lane % 21] : lane * 0x7395a8317b421eb7ULL;
                // Payload handling has separate coverage. This test compares all
                // words exactly, including when the two destinations overlap.
                if (op == 4 && (uint32_t(raw) & 0x7fffffff) > 0x7f800000)
                  raw &= ~0x7fffffULL;
                if (op != 4 && (raw & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL)
                  raw &= ~0xfffffffffffffULL;
                original[0][lane] = uint32_t(raw);
                original[1][lane] = uint32_t(raw >> 32);
                original[2][lane] = original[3][lane] = 0xdeadbeef;
              }
              std::memcpy(expected, original, sizeof(expected));
              for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
                if ((exec_mask >> lane) & 1) {
                  uint64_t raw = original[0][lane + 1] | (uint64_t(original[1][lane + 1]) << 32);
                  uint64_t result = goc_test::conversion64_reference(
                      op, op == 4 ? uint32_t(raw) : raw, effective);
                  for (unsigned reg = 0; reg < outputs; ++reg)
                    expected[destinations[reg]][lane + 1] = uint32_t(result >> (32 * reg));
                }
              for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
                std::memcpy(actual, original, sizeof(actual));
                uint32_t *d[] = {actual[destinations[0]] + 1,
                                 outputs == 2 ? actual[destinations[1]] + 1 : nullptr};
                const uint32_t *a[] = {actual[0] + 1, outputs == 1 ? actual[1] + 1 : nullptr};
                uint32_t state = 0x80000055;
                ASSERT_EQ(functions[arch][wide][op](cpu | policy(guest), exec_mask, mode, d, a,
                                                    op >= 2 ? &state : nullptr),
                          GOC_SUCCESS);
                ASSERT_EQ(std::memcmp(actual, expected, sizeof(actual)), 0)
                    << arch << '/' << wide << '/' << op << '/' << guest << '/' << m << '/' << alias
                    << '/' << cpu;
                EXPECT_EQ(state, 0x80000055U);
              }
            }
}

TEST(Conversion64Architecture, ValidationAndHostControls) {
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
          uint32_t input[2][64] = {}, output[2][64] = {}, state = 0x80000055;
          const uint32_t *a[] = {input[0], input[1]};
          uint32_t *d[] = {output[0], output[1]};
          auto fn = functions[arch][wide][op];
          for (uint64_t exec_mask : {0ULL, ~0ULL}) {
            EXPECT_EQ(fn(policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                         exec_mask, 0, d, a),
                      GOC_ERROR_UNSUPPORTED_SEMANTICS);
            for (uint64_t invalid : {uint64_t(GOC_ALU_NEG_B), uint64_t(1) << 32})
              EXPECT_EQ(fn(policy(guest), exec_mask, invalid, d, a), GOC_ERROR_INVALID_FLAGS);
            if (op >= 2) {
              for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
                EXPECT_EQ(fn(policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | strict, exec_mask, 0,
                             d, a, &state),
                          GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
            }
            EXPECT_EQ(state, 0x80000055U);
            for (auto &reg : output)
              EXPECT_TRUE(std::all_of(reg, reg + 64, [](uint32_t v) { return v == 0; }));
          }
#if defined(__x86_64__) || defined(_M_X64)
          unsigned controls = _mm_getcsr();
#endif
          ASSERT_EQ(fn(policy(guest), UINT64_MAX, GOC_ALU_OMOD_2, d, a, op >= 2 ? &state : nullptr),
                    GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), FE_TONEAREST);
#if defined(__x86_64__) || defined(_M_X64)
          EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls & ~0x3fU);
#endif
          EXPECT_EQ(state, 0x80000055U);
        }
}

TEST(Conversion64Architecture, NanResultsAcrossGuestPolicies) {
  // Loose conversion preserves the NaN class and quiets results; it does not
  // promise payload bits. Integer conversions map either NaN kind to zero.
  const uint64_t nans[] = {0x7ff0000000000001ULL, 0xfff0000000000001ULL, 0x7ff8123456789abcULL,
                           0xfff8123456789abcULL};
  const uint32_t float_nans[] = {0x7f800001, 0xff800001, 0x7fc12345, 0xffc12345};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 2; op < 6; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned m = 0; m < goc_test::conversion64_modes(op); ++m)
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              uint32_t source[2][64], output[2][64];
              for (unsigned lane = 0; lane < 64; ++lane) {
                uint64_t raw = op == 4 ? float_nans[lane % 4] : nans[lane % 4];
                source[0][lane] = uint32_t(raw);
                source[1][lane] = uint32_t(raw >> 32);
              }
              const uint32_t *a[] = {source[0], source[1]};
              uint32_t *d[] = {output[0], op == 4 ? output[1] : nullptr};
              uint32_t mode = goc_test::conversion64_mode(op, m);
              uint32_t effective =
                  !arch && (!(guest & 2) || (guest & 4)) ? mode & ~GOC_ALU_OMOD_HALF : mode;
              uint32_t state = 0x80000055;
              ASSERT_EQ(
                  functions[arch][wide][op](cpu | policy(guest), UINT64_MAX, mode, d, a, &state),
                  GOC_SUCCESS);
              for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
                uint64_t raw = op == 4 ? float_nans[lane % 4] : nans[lane % 4];
                uint64_t expected = goc_test::conversion64_reference(op, raw, effective);
                uint64_t actual = output[0][lane];
                if (op == 4)
                  actual |= uint64_t(output[1][lane]) << 32;
                ASSERT_TRUE(goc_test::conversion64_equal(op, actual, expected))
                    << arch << '/' << wide << '/' << op << '/' << guest << '/' << m << '/' << cpu;
              }
              EXPECT_EQ(state, 0x80000055U);
            }
}

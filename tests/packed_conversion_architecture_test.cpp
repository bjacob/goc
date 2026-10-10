// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "normalized_reference.h"
#include "packed_conversion_reference.h"
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

using Fn = goc_test::WaveInstruction<decltype(&goc_v_cvt_pk_norm_i16_f32)>;
const Fn functions[2][2][9] = {
    {
        {
            goc_v_cvt_pk_rtz_f16_f32,
            goc_v_cvt_pk_i16_f32,
            goc_v_cvt_pk_u16_f32,
            goc_v_cvt_pk_norm_i16_f32,
            goc_v_cvt_pk_norm_u16_f32,
            goc_v_cvt_pk_norm_i16_f16,
            goc_v_cvt_pk_norm_u16_f16,
            +[](uint64_t f, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *) {
              return goc_v_cvt_norm_i16_f16(f, exec_mask, mode, d, a);
            },
            +[](uint64_t f, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *) {
              return goc_v_cvt_norm_u16_f16(f, exec_mask, mode, d, a);
            },
        },
        {
            goc_v_cvt_pk_rtz_f16_f32_wave64,
            goc_v_cvt_pk_i16_f32_wave64,
            goc_v_cvt_pk_u16_f32_wave64,
            goc_v_cvt_pk_norm_i16_f32_wave64,
            goc_v_cvt_pk_norm_u16_f32_wave64,
            goc_v_cvt_pk_norm_i16_f16_wave64,
            goc_v_cvt_pk_norm_u16_f16_wave64,
            +[](uint64_t f, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *) {
              return goc_v_cvt_norm_i16_f16_wave64(f, exec_mask, mode, d, a);
            },
            +[](uint64_t f, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *) {
              return goc_v_cvt_norm_u16_f16_wave64(f, exec_mask, mode, d, a);
            },
        },
    },
    {
        {
            goc_v_cvt_pk_rtz_f16_f32_rdna4,
            goc_v_cvt_pk_i16_f32_rdna4,
            goc_v_cvt_pk_u16_f32_rdna4,
            goc_v_cvt_pk_norm_i16_f32,
            goc_v_cvt_pk_norm_u16_f32,
            goc_v_cvt_pk_norm_i16_f16_rdna4,
            goc_v_cvt_pk_norm_u16_f16_rdna4,
            +[](uint64_t f, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *) {
              return goc_v_cvt_norm_i16_f16(f, exec_mask, mode, d, a);
            },
            +[](uint64_t f, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *) {
              return goc_v_cvt_norm_u16_f16(f, exec_mask, mode, d, a);
            },
        },
        {
            goc_v_cvt_pk_rtz_f16_f32_rdna4_wave64,
            goc_v_cvt_pk_i16_f32_rdna4_wave64,
            goc_v_cvt_pk_u16_f32_rdna4_wave64,
            goc_v_cvt_pk_norm_i16_f32_wave64,
            goc_v_cvt_pk_norm_u16_f32_wave64,
            goc_v_cvt_pk_norm_i16_f16_rdna4_wave64,
            goc_v_cvt_pk_norm_u16_f16_rdna4_wave64,
            +[](uint64_t f, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *) {
              return goc_v_cvt_norm_i16_f16_wave64(f, exec_mask, mode, d, a);
            },
            +[](uint64_t f, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *) {
              return goc_v_cvt_norm_u16_f16_wave64(f, exec_mask, mode, d, a);
            },
        },
    },
};

uint64_t policy(unsigned bits) {
  return (bits & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (bits & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (bits & 4 ? GOC_FP_IEEE_MODE : 0);
}

bool reports(unsigned op) { return op < 3 || op == 5 || op == 6; }

unsigned modes(unsigned op) {
  return op < 3 ? goc_test::packed_conversion_modes(op) : goc_test::normalized_modes(op - 3);
}

uint32_t mode_at(unsigned op, unsigned m) {
  return op < 3 ? goc_test::packed_conversion_mode(op, m) : goc_test::normalized_mode(op - 3, m);
}

uint32_t reference(unsigned op, uint32_t a, uint32_t b, uint32_t old, uint32_t mode) {
  return op < 3 ? goc_test::packed_conversion_reference(op, a, b, mode)
                : goc_test::normalized_reference(op - 3, a, b, old, mode);
}

} // namespace

TEST(PackedConversionArchitecture, GuestPoliciesModifiersRoutingAndAliases) {
  const uint32_t edge[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000,
                           0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x46ffffff, 0x477fffff,
                           0x7f800000, 0xff800000, 0x7f800001, 0xffc12345, 0x80010001, 0x84000400,
                           0xbc003c00, 0xfc007c00, 0xfc017c01, 0xffff7fff};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 9; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned m = 0; m < modes(op); ++m)
            for (unsigned route = 0; route < 3; ++route)
              for (unsigned alias = 0; alias < 3; ++alias) {
                uint64_t descriptor = route ? goc_test::dpp_modes[route == 1 ? 0 : 6] : 0;
                uint32_t mode = mode_at(op, m);
                uint64_t exec_mask = wide ? 0xc5912d6ea3157bf1ULL : 0xa3157bf1U;
                uint32_t original[3][66], expected[3][66], actual[3][66];
                for (unsigned lane = 0; lane < 66; ++lane) {
                  original[0][lane] = lane < 44 ? edge[lane % 22] : lane * 0x7395a831U;
                  original[1][lane] = lane < 44 ? edge[(lane + 9) % 22] : lane * 0x59137af1U;
                  original[2][lane] = 0xdeadbeef;
                }
                std::memcpy(expected, original, sizeof(expected));
                for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
                  int source = int(lane);
                  bool write = descriptor
                                   ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                   : (exec_mask >> lane) & 1;
                  if (write)
                    expected[alias][lane + 1] =
                        reference(op, source < 0 ? 0 : original[0][source + 1],
                                  original[1][lane + 1], original[alias][lane + 1], mode);
                }
                for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
                  std::memcpy(actual, original, sizeof(actual));
                  uint32_t *d = actual[alias] + 1;
                  const uint32_t *a = actual[0] + 1, *b = actual[1] + 1;
                  uint32_t state = 0x80000055;
                  ASSERT_EQ(functions[arch][wide][op](cpu | policy(guest), exec_mask,
                                                      mode | descriptor, &d, &a, &b,
                                                      reports(op) ? &state : nullptr),
                            GOC_SUCCESS);
                  ASSERT_EQ(std::memcmp(actual, expected, sizeof(actual)), 0)
                      << arch << '/' << wide << '/' << op << '/' << guest << '/' << m << '/'
                      << route << '/' << alias << '/' << cpu;
                  EXPECT_EQ(state, 0x80000055U);
                }
              }
}

TEST(PackedConversionArchitecture, ValidationAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
#if defined(__x86_64__) || defined(_M_X64)
  _mm_setcsr(_mm_getcsr() & ~0x8040U);
#endif
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 9; ++op)
        for (unsigned guest = 0; guest < 8; ++guest) {
          uint32_t input[64], output[64];
          std::fill_n(input, 64, 0x3e803800);
          std::fill_n(output, 64, 0xdeadbeef);
          const uint32_t *a = input;
          uint32_t *d = output;
          auto fn = functions[arch][wide][op];
          uint32_t state = 0x80000055;
          for (uint64_t exec_mask : {0ULL, ~0ULL}) {
            EXPECT_EQ(fn(policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                         exec_mask, 0, &d, &a, &a),
                      GOC_ERROR_UNSUPPORTED_SEMANTICS);
            EXPECT_EQ(fn(policy(guest), exec_mask, GOC_ALU_NEG_C, &d, &a, &a),
                      GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(fn(policy(guest) | (1ULL << 63), exec_mask, 0, &d, &a, &a),
                      GOC_ERROR_INVALID_FLAGS);
            if (reports(op)) {
              for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
                EXPECT_EQ(fn(policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | strict, exec_mask, 0,
                             &d, &a, &a, &state),
                          GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
            }
            EXPECT_TRUE(
                std::all_of(output, output + 64, [](uint32_t v) { return v == 0xdeadbeef; }));
            EXPECT_EQ(state, 0x80000055U);
          }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
#if defined(__x86_64__) || defined(_M_X64)
            unsigned controls = _mm_getcsr();
#endif
            ASSERT_EQ(fn(cpu | policy(guest), UINT64_MAX, mode_at(op, modes(op) - 1), &d, &a, &a,
                         reports(op) ? &state : nullptr),
                      GOC_SUCCESS);
            EXPECT_EQ(std::fegetround(), FE_TONEAREST);
#if defined(__x86_64__) || defined(_M_X64)
            EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls & ~0x3fU);
#endif
            EXPECT_EQ(state, 0x80000055U);
          }
        }
}

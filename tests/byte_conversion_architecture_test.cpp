// SPDX-License-Identifier: MIT

#include "byte_conversion_reference.h"
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

using Fn = goc_test::WaveInstruction<decltype(&goc_v_cvt_f32_ubyte0)>;
const Fn functions[2][2][5] = {
    {{goc_v_cvt_f32_ubyte0, goc_v_cvt_f32_ubyte1, goc_v_cvt_f32_ubyte2, goc_v_cvt_f32_ubyte3,
      goc_v_cvt_off_f32_i4},
     {goc_v_cvt_f32_ubyte0_wave64, goc_v_cvt_f32_ubyte1_wave64, goc_v_cvt_f32_ubyte2_wave64,
      goc_v_cvt_f32_ubyte3_wave64, goc_v_cvt_off_f32_i4_wave64}},
    {{goc_v_cvt_f32_ubyte0_rdna4, goc_v_cvt_f32_ubyte1_rdna4, goc_v_cvt_f32_ubyte2_rdna4,
      goc_v_cvt_f32_ubyte3_rdna4, goc_v_cvt_off_f32_i4_rdna4},
     {goc_v_cvt_f32_ubyte0_rdna4_wave64, goc_v_cvt_f32_ubyte1_rdna4_wave64,
      goc_v_cvt_f32_ubyte2_rdna4_wave64, goc_v_cvt_f32_ubyte3_rdna4_wave64,
      goc_v_cvt_off_f32_i4_rdna4_wave64}}};

uint64_t policy(unsigned bits) {
  return (bits & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (bits & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (bits & 4 ? GOC_FP_IEEE_MODE : 0);
}

uint32_t reference(unsigned arch, unsigned guest, unsigned op, uint32_t input, uint32_t mode) {
  // RDNA3 ISA 7.2.3.1: output flush and IEEE=0 are required. Input flush
  // is irrelevant to these integer inputs. RDNA4 enables OMOD in all modes.
  const bool scaling[2][8] = {{false, false, true, true, false, false, false, false},
                              {true, true, true, true, true, true, true, true}};
  if (!scaling[arch][guest])
    mode &= ~GOC_ALU_OMOD_HALF;
  return op == 4 ? goc_test::nibble_offset_reference(input, mode)
                 : goc_test::byte_conversion_reference(op, input, mode);
}

} // namespace

TEST(ByteConversionArchitecture, EveryInputPolicyModifierAndSemantics) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 5; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned m = 0; m < 8; ++m)
            for (uint64_t semantics :
                 {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
              for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
                for (unsigned base = 0; base < 256; base += wide ? 64 : 32) {
                  uint32_t input[64], output[64];
                  std::fill_n(output, 64, 0xdeadbeef);
                  for (unsigned lane = 0; lane < 64; ++lane) {
                    unsigned shift = op == 4 ? 0 : op * 8;
                    uint32_t select = op == 4 ? 15 : 255;
                    uint32_t noise = (lane + 1) * 0x7395a831U;
                    input[lane] =
                        (noise & ~(select << shift)) | (((base + lane) & select) << shift);
                    output[lane] = 0xdeadbeef;
                  }
                  const uint32_t *a = input;
                  uint32_t *d = output;
                  uint32_t mode = goc_test::byte_conversion_mode(m);
                  ASSERT_EQ(functions[arch][wide][op](cpu | policy(guest) | semantics, UINT64_MAX,
                                                      mode, &d, &a),
                            GOC_SUCCESS);
                  for (unsigned lane = 0; lane < 64; ++lane)
                    ASSERT_EQ(output[lane], lane < (wide ? 64U : 32U)
                                                ? reference(arch, guest, op, input[lane], mode)
                                                : 0xdeadbeefU)
                        << arch << '/' << wide << '/' << op << '/' << guest << '/' << m << '/'
                        << semantics << '/' << cpu << '/' << base << '/' << lane;
                }
}

TEST(ByteConversionArchitecture, DppMasksAliasesAndGuards) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 5; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned m = 0; m < 8; ++m)
            for (uint64_t descriptor : goc_test::dpp_modes)
              for (bool alias : {false, true}) {
                const uint64_t exec_mask = wide ? 0xc5912d6ea3157bf1ULL : 0xa3157bf1U;
                uint32_t input[66], separate[66], original[66], expected[66];
                uint32_t mode = goc_test::byte_conversion_mode(m);
                for (unsigned lane = 0; lane < 66; ++lane) {
                  original[lane] = lane * 0x7395a831U;
                  expected[lane] = alias ? original[lane] : 0xdeadbeef;
                }
                for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
                  int source = int(lane);
                  if (goc_test::dpp_source(descriptor, exec_mask, lane, source))
                    expected[lane + 1] =
                        reference(arch, guest, op, source < 0 ? 0 : original[source + 1], mode);
                }
                for (uint64_t semantics :
                     {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
                  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
                    std::copy(original, original + 66, input);
                    std::fill(separate, separate + 66, 0xdeadbeef);
                    const uint32_t *a = input + 1;
                    uint32_t *d = (alias ? input : separate) + 1;
                    ASSERT_EQ(functions[arch][wide][op](cpu | policy(guest) | semantics, exec_mask,
                                                        mode | descriptor, &d, &a),
                              GOC_SUCCESS);
                    ASSERT_TRUE(std::equal(d - 1, d + 65, expected))
                        << arch << '/' << wide << '/' << op << '/' << guest << '/' << m << '/'
                        << alias << '/' << semantics << '/' << cpu;
                  }
              }
}

TEST(ByteConversionArchitecture, HostModesAndValidation) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    for (bool flush : {false, true}) {
#if defined(__x86_64__) || defined(_M_X64)
      _mm_setcsr((_mm_getcsr() & ~0x8040U) | (flush ? 0x8040U : 0));
#else
      (void)flush;
#endif
      ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
      const int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
#if defined(__x86_64__) || defined(_M_X64)
      const unsigned controls = _mm_getcsr();
#endif
      for (unsigned arch = 0; arch < 2; ++arch)
        for (unsigned wide = 0; wide < 2; ++wide)
          for (unsigned op = 0; op < 5; ++op)
            for (unsigned guest = 0; guest < 8; ++guest)
              for (unsigned m = 0; m < 8; ++m)
                for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
                  uint32_t input[64], output[64];
                  std::fill_n(output, 64, 0xdeadbeef);
                  for (unsigned lane = 0; lane < 64; ++lane)
                    input[lane] = lane * 0x0103070bU;
                  const uint32_t *a = input;
                  uint32_t *d = output;
                  uint64_t flags =
                      cpu | policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
                  uint32_t mode = goc_test::byte_conversion_mode(m);
                  ASSERT_EQ(functions[arch][wide][op](flags, UINT64_MAX, mode, &d, &a),
                            GOC_SUCCESS);
                  for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
                    ASSERT_EQ(output[lane], reference(arch, guest, op, input[lane], mode));
                  for (uint64_t exec_mask : {0ULL, ~0ULL}) {
                    uint32_t before[64];
                    std::memcpy(before, output, sizeof(before));
                    EXPECT_EQ(
                        functions[arch][wide][op](flags, exec_mask, mode | GOC_ALU_ABS_A, &d, &a),
                        GOC_ERROR_INVALID_FLAGS);
                    EXPECT_EQ(
                        functions[arch][wide][op](flags | (1ULL << 63), exec_mask, mode, &d, &a),
                        GOC_ERROR_INVALID_FLAGS);
                    EXPECT_EQ(std::memcmp(before, output, sizeof(before)), 0);
                  }
                }
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), controls);
#endif
    }
  }
}

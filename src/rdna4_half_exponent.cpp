// SPDX-License-Identifier: MIT

#include "rdna4_half_exponent.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_dpp.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace {

template <bool Ldexp>
int run(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b) {
  if (mode >> 32)
    return goc::execute_dpp(flags, mask, mode, a,
                            [&](uint32_t effective, const uint32_t *const *source) {
                              return run<Ldexp>(flags, effective, uint32_t(mode), d, source, b);
                            });
  const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP |
                         GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | (Ldexp ? GOC_ALU_HIGH_B : 0);
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::half_exponent_x86_64_v3<Ldexp>(bool(flags & GOC_FP16_OVFL), uint32_t(mask), mode, d[0],
                                        a[0], Ldexp ? b[0] : nullptr);
    return GOC_SUCCESS;
  }
#endif
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & GOC_ALU_HIGH_B ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t bits = uint16_t(a[0][lane] >> a_shift);
    if constexpr (Ldexp) {
      uint16_t raw_exponent = uint16_t(b[0][lane] >> b_shift);
      int exponent = int(raw_exponent) - (raw_exponent & 0x8000 ? 65536 : 0);
      // Following rocjitsu's F16 LDEXP model, bound the exponent while retaining
      // every FP16 rounding outcome, including output scaling and saturation.
      float value = std::ldexp(goc::alu_input(goc::as_bits(goc::f16_to_float(bits)), mode),
                               std::clamp(exponent, -64, 64));
      if (mode & GOC_ALU_OMOD_HALF) {
        // Test tininess before rounding; finite overflow is rounded before OMOD.
        value = std::abs(value) < 0x1p-14f
                    ? 0.0f
                    : goc::f16_to_float(goc::float_to_f16(value, flags & GOC_FP16_OVFL));
      }
      result[lane] = goc::float_to_f16(goc::alu_output_f16(value, mode), flags & GOC_FP16_OVFL);
    } else {
      // Sign modifiers do not affect the exponent; OMOD is ignored for integer
      // results and all FP16 exponents fit int16_t, so CLAMP has no effect.
      int field = (bits >> 10) & 31, fraction = bits & 1023;
      int exponent = field - 14;
      if (field == 31 || (bits & 0x7fff) == 0)
        exponent = 0;
      else if (field == 0) {
        exponent = -24;
        while (fraction) {
          ++exponent;
          fraction >>= 1;
        }
      }
      result[lane] = uint16_t(exponent);
    }
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] =
          (d[0][lane] & ~(UINT32_C(0xffff) << d_shift)) | (uint32_t(result[lane]) << d_shift);
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_ldexp_f16(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b) {
  return run<true>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_frexp_exp_i16_f16(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                                  const uint32_t *const *a) {
  return run<false>(flags, mask, mode, d, a, nullptr);
}

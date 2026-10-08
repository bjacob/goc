// SPDX-License-Identifier: MIT

#include "rdna4_half_fma.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_fp64.h"

#include <cfenv>
#include <cmath>
#include <limits>
#include <stdint.h>

namespace {

// Restore rounding, exception flags and trap enables after exact arithmetic.
class Environment {
public:
  explicit Environment(bool exact) : active(exact) {
    if (active) {
      std::feholdexcept(&saved);
      std::fesetround(FE_TONEAREST);
    }
  }

  ~Environment() {
    if (active)
      std::fesetenv(&saved);
  }

private:
  bool active;
  std::fenv_t saved;
};

// Round finite double values to FP16 without double rounding through FP32.
uint16_t narrow(double value, bool saturate) {
  float rounded = float(value);
  uint32_t bits = goc::as_bits(rounded);
  if (double(rounded) != value && !(bits & 1)) {
    bool increase = (value > double(rounded)) == !std::signbit(rounded);
    rounded = goc::as_float(bits + (increase ? 1u : UINT32_MAX));
  }
  return goc::float_to_f16(rounded, saturate);
}

uint16_t clamp(uint16_t value) {
  return (value & 0x8000) || (value & 0x7fff) > 0x7c00 ? 0 : value > 0x3c00 ? 0x3c00 : value;
}

// Adapted from rocjitsu shared/fp_mode.h: fma_f16 and finish_fma_f16.
// RNE and preserved input/output denormals are the currently exposed FP policy.
uint16_t fma(uint16_t a, uint16_t b, uint16_t c, uint32_t mode, bool saturate) {
  uint16_t inputs[] = {a, b, c};
  for (int i = 0; i < 3; ++i) {
    if (mode & (GOC_ALU_ABS_A << i))
      inputs[i] &= 0x7fff;
    if (mode & (GOC_ALU_NEG_A << i))
      inputs[i] ^= 0x8000;
  }
  a = inputs[0];
  b = inputs[1];
  c = inputs[2];
  uint16_t ma = a & 0x7fff, mb = b & 0x7fff, mc = c & 0x7fff;
  uint16_t exceptional = 0;
  if ((ma == 0 && mb == 0x7c00) || (mb == 0 && ma == 0x7c00))
    exceptional = 0xfe00;
  else if (ma > 0x7c00 || mb > 0x7c00 || mc > 0x7c00)
    exceptional = (ma > 0x7c00 ? a : mb > 0x7c00 ? b : c) | 0x0200;
  else if (ma == 0x7c00 || mb == 0x7c00) {
    exceptional = ((a ^ b) & 0x8000) | 0x7c00;
    if (mc == 0x7c00 && ((exceptional ^ c) & 0x8000))
      exceptional = 0xfe00;
  } else if (mc == 0x7c00)
    exceptional = c;
  if (exceptional)
    return mode & GOC_ALU_CLAMP ? clamp(exceptional) : exceptional;

  // Half products are exact in double. TwoSum plus round-to-odd retains tiny
  // addends that would otherwise disappear at a later half rounding boundary.
  double product = double(goc::f16_to_float(a)) * double(goc::f16_to_float(b));
  double addend = goc::f16_to_float(c);
  double value = product + addend;
  double virtual_c = value - product;
  double error = (product - (value - virtual_c)) + (addend - virtual_c);
  if (error != 0 && !(goc::double_bits(value) & 1))
    value = std::nextafter(value, std::copysign(std::numeric_limits<double>::infinity(), error));
  // Under nearest-even, only matching negative zero terms produce -0.
  if (value == 0)
    value = ((a ^ b) & c & 0x8000) && mc == 0 ? -0.0 : 0.0;

  uint16_t result = narrow(value, saturate);
  uint32_t omod = (mode >> 6) & 3;
  if (omod) {
    uint16_t magnitude = result & 0x7fff;
    if (magnitude < 0x0400 || (magnitude == 0x0400 && std::abs(value) < 0x1p-14 - 0x1p-26))
      result = 0;
    else if (omod == 3 && magnitude < 0x0800)
      result &= 0x8000;
    else {
      const float scales[] = {1, 2, 4, 0.5f};
      result = goc::float_to_f16(goc::f16_to_float(result) * scales[omod], saturate);
    }
  }
  return mode & GOC_ALU_CLAMP ? clamp(result) : result;
}

int run(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = goc::validate(flags, mode & ~UINT32_C(0x1fff), true))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
  bool exact = (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
#if defined(GOC_HAVE_X86_64_V3)
  if (!exact && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::half_fma_x86_64_v3(bool(flags & GOC_FP16_OVFL), uint32_t(mask), mode, d[0], a[0], b[0],
                            c[0]);
    return GOC_SUCCESS;
  }
#endif
  Environment environment(exact);
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & GOC_ALU_HIGH_B ? 16 : 0;
  int c_shift = mode & GOC_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane)
    result[lane] = fma(uint16_t(a[0][lane] >> a_shift), uint16_t(b[0][lane] >> b_shift),
                       uint16_t(c[0][lane] >> c_shift), mode, flags & GOC_FP16_OVFL);
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] =
          (d[0][lane] & ~(UINT32_C(0xffff) << d_shift)) | (uint32_t(result[lane]) << d_shift);
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_fma_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return run(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_fmac_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B |
                         GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  // FMAC reads the same destination half that it overwrites.
  if (mode & GOC_ALU_HIGH_D)
    mode |= GOC_ALU_HIGH_C;
  return run(flags, mask, mode, d, a, b, d);
}

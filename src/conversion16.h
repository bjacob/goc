// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class Conversion16 {
  SignedToHalf,
  UnsignedToHalf,
  HalfToSigned,
  HalfToUnsigned,
  FloatToHalf,
  HalfToFloat
};

template <Conversion16 Op>
void conversion16_x86_64_v3(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a);

template <Conversion16 Op>
void conversion16_x86_64_v4(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a);

// Widen with guest input flushing and NaN policy. IEEE selects quiet NaNs;
// the caller sets it unconditionally for RDNA4. Ordinary RDNA4 widening uses
// the template above so these policy checks do not burden its fast path.
void conversion16_widen_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                  const uint32_t *a);
void conversion16_widen_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                  const uint32_t *a);

} // namespace goc

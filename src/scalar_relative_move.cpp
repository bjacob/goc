// SPDX-License-Identifier: MIT

// Both ISA manuals, sections 3.3.1 and S_MOVREL: indices count DWORDs,
// and indexing cannot cross between SGPR/VCC and trap-temporary regions.

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

int validate(uint64_t flags, uint64_t mode) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return goc::validate(flags, uint32_t(mode), true);
}

bool in_range(uint32_t base, uint32_t offset, uint32_t count, unsigned words) {
  uint64_t end = uint64_t(base) + offset + words;
  unsigned region_end = base < 108 ? 108 : base < 124 ? 124 : 0;
  return end <= region_end && end <= count;
}

uint64_t read(const uint32_t *a, uint32_t count, uint32_t base, uint32_t offset, unsigned words) {
  uint32_t index = in_range(base, offset, count, words) ? base + offset : 0;
  uint64_t value = index < count ? a[index] : 0;
  if (words == 2 && index + 1 < count)
    value |= uint64_t(a[index + 1]) << 32;
  return value;
}

} // namespace

int goc_s_movrels_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, const uint32_t *a,
                      uint32_t a_count, uint32_t a_base, uint32_t m0) {
  if (int error = validate(flags, instruction_flags))
    return error;
  *d = uint32_t(read(a, a_count, a_base, m0, 1));
  return GOC_SUCCESS;
}

int goc_s_movrels_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, const uint32_t *a,
                      uint32_t a_count, uint32_t a_base, uint32_t m0) {
  if (int error = validate(flags, instruction_flags))
    return error;
  *d = read(a, a_count, a_base, m0, 2);
  return GOC_SUCCESS;
}

int goc_s_movreld_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t d_count, uint32_t d_base, uint32_t m0) {
  if (int error = validate(flags, instruction_flags))
    return error;
  if (in_range(d_base, m0, d_count, 1))
    d[d_base + m0] = a;
  return GOC_SUCCESS;
}

int goc_s_movreld_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
                      uint32_t d_count, uint32_t d_base, uint32_t m0) {
  if (int error = validate(flags, instruction_flags))
    return error;
  if (in_range(d_base, m0, d_count, 2)) {
    d[d_base + m0] = uint32_t(a);
    d[d_base + m0 + 1] = uint32_t(a >> 32);
  }
  return GOC_SUCCESS;
}

int goc_s_movrelsd_2_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, const uint32_t *a,
                         uint32_t d_count, uint32_t a_count, uint32_t d_base, uint32_t a_base,
                         uint32_t m0) {
  if (int error = validate(flags, instruction_flags))
    return error;
  uint32_t dst_offset = (m0 >> 16) & 1023;
  if (in_range(d_base, dst_offset, d_count, 1))
    d[d_base + dst_offset] = uint32_t(read(a, a_count, a_base, m0 & 1023, 1));
  return GOC_SUCCESS;
}

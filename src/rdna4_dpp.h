// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

// Writes all 32 permuted source words; out must not overlap a. Inactive sources
// become positive zero unless FI is set. Does not access destination operands.
void dpp8_source(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *out, const uint32_t *a);
void dpp8_x86_64_v3(uint32_t mask, uint32_t selectors, bool fi, uint32_t *out, const uint32_t *a);
void dpp8_x86_64_v4(uint32_t mask, uint32_t selectors, bool fi, uint32_t *out, const uint32_t *a);

// Checks descriptor fields and the RDNA4 control encoding; low 32 bits are
// validated by the instruction. Zero denotes no DPP modifier.
bool valid_dpp(uint64_t mode);

// Writes all permuted source words and returns the destination EXEC mask after
// row/bank/source filtering. out must not overlap a. Valid descriptor required.
uint32_t dpp16_source(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *out,
                      const uint32_t *a);
uint32_t dpp16_x86_64_v3(uint32_t mask, uint32_t control, uint32_t valid_row, bool fi,
                         uint32_t *out, const uint32_t *a);
uint32_t dpp16_x86_64_v4(uint32_t mask, uint32_t control, uint32_t valid_row, bool fi,
                         uint32_t *out, const uint32_t *a);

} // namespace goc

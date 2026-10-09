// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

// Writes all 32 permuted source words; out must not overlap a. Inactive sources
// become positive zero unless FI is set. Does not access destination operands.
void dpp8_source(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *out, const uint32_t *a);
void dpp8_x86_64_v3(uint32_t mask, uint32_t selectors, bool fi, uint32_t *out, const uint32_t *a);
void dpp8_x86_64_v4(uint32_t mask, uint32_t selectors, bool fi, uint32_t *out, const uint32_t *a);

} // namespace goc

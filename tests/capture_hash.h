// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

static const uint64_t capture_hash_seed = 14695981039346656037ULL;

// Mix one whole word, as used by word-wise hardware capture digests.
inline uint64_t capture_hash_word(uint64_t hash, uint64_t word) {
  return (hash ^ word) * 1099511628211ULL;
}

// Mix the low byte_count bytes in low-to-high order, independent of host endian.
// byte_count must not exceed eight. This differs from mixing one whole word.
inline uint64_t capture_hash_bytes(uint64_t hash, uint64_t word, unsigned byte_count) {
  for (unsigned byte = 0; byte < byte_count; ++byte)
    hash = capture_hash_word(hash, (word >> (8 * byte)) & 255);
  return hash;
}

} // namespace goc_test

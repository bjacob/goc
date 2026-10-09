// SPDX-License-Identifier: MIT

#ifndef GOC_TEST_RDNA4_SHIFT_REFERENCE_H_
#define GOC_TEST_RDNA4_SHIFT_REFERENCE_H_

#include <stdint.h>

namespace goc_test {

// Independent bit-by-bit reference: op 0=left, 1=logical right, 2=arithmetic
// right. Build each output bit from its source position or the extended sign.
inline uint64_t shift_reference(int bits, int op, uint32_t count, uint64_t input) {
  const int distance = int(count % unsigned(bits));
  uint64_t result = 0;
  for (int bit = 0; bit < bits; ++bit) {
    int source = op == 0 ? bit - distance : bit + distance;
    bool value = source >= 0 && source < bits ? (input >> source) & 1
                 : source >= bits && op == 2  ? (input >> (bits - 1)) & 1
                                              : false;
    if (value)
      result |= UINT64_C(1) << bit;
  }
  return result;
}

} // namespace goc_test

#endif

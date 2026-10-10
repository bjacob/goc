// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "ds_swizzle_hardware.h"
#include "goc/goc.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

int run(unsigned lanes, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, uint16_t offset) {
  return lanes == 32 ? goc_ds_swizzle_b32(exact, uint32_t(exec_mask), mode, d, a, offset)
                     : goc_ds_swizzle_b32_wave64(exact, exec_mask, mode, d, a, offset);
}

// Independent bit-by-bit routing reference. The FFT lookup is the manual's
// published mask=0 order (converted from one-based to zero-based lane indices).
unsigned source_lane(unsigned lane, unsigned offset) {
  static const unsigned fft[] = {0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                                 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31};
  unsigned local = lane % 32, source = 0;
  if (offset >= 0xe000) {
    unsigned divisor = 1;
    for (unsigned bit = 0; bit < 5; ++bit)
      if (offset & (1U << bit))
        divisor *= 2;
    source = fft[local] / divisor;
    for (unsigned bit = 0; bit < 5; ++bit)
      if ((offset & local) & (1U << bit))
        source |= 1U << bit;
  } else if (offset >= 0xc000) {
    int delta = (offset / 32) % 32;
    if (offset & 1024)
      delta = -delta;
    unsigned rotated = (int(local) + delta + 32) % 32;
    for (unsigned bit = 0; bit < 5; ++bit)
      source |= (((offset >> bit) & 1 ? local : rotated) & (1U << bit));
  } else if (offset & 0x8000) {
    source = (local / 4) * 4 + ((offset >> (2 * (local % 4))) & 3);
  } else {
    for (unsigned bit = 0; bit < 5; ++bit) {
      bool value = ((local >> bit) & 1) && ((offset >> bit) & 1);
      value = value || ((offset >> (bit + 5)) & 1);
      value = value != bool((offset >> (bit + 10)) & 1);
      source |= unsigned(value) << bit;
    }
  }
  return lane / 32 * 32 + source;
}

} // namespace

TEST(DsSwizzle, AllOffsetEncodingsBothWavesAndInPlace) {
  for (unsigned lanes : {32U, 64U})
    for (unsigned offset = 0; offset < 65536; ++offset) {
      uint32_t words[66];
      for (unsigned lane = 0; lane < 66; ++lane)
        words[lane] = 0x80000000U + lane;
      uint32_t *d = words + 1;
      ASSERT_EQ(run(lanes, UINT64_MAX, 0, &d, &d, offset), GOC_SUCCESS);
      EXPECT_EQ(words[0], 0x80000000U);
      for (unsigned lane = 0; lane < lanes; ++lane)
        ASSERT_EQ(d[lane], 0x80000001U + source_lane(lane, offset))
            << lanes << '/' << offset << '/' << lane;
      EXPECT_EQ(words[lanes + 1], 0x80000001U + lanes);
    }
}

TEST(DsSwizzle, Rx9070HardwareCorpora) {
  std::vector<uint16_t> offsets;
  for (unsigned offset = 0xc000; offset < 0xc800; ++offset)
    offsets.push_back(offset);
  for (unsigned offset = 0xe000; offset < 0xe020; ++offset)
    offsets.push_back(offset);
  for (unsigned offset = 0x8000; offset < 0x8100; ++offset)
    offsets.push_back(offset);
  for (unsigned i = 0; i < 32; ++i)
    offsets.push_back(31 | (i << 10));
  const uint64_t masks[2][8] = {
      {0xffffffffU, 0, 0xaaaaaaaaU, 0x55555555U, 1, 0x80000000U, 0xffff, 0xffff0000U},
      {UINT64_MAX, 0, 0xaaaaaaaaaaaaaaaaULL, 0x5555555555555555ULL, 1, 1ULL << 63, 0xffffffffULL,
       0xffffffff00000000ULL}};
  for (unsigned wave = 0; wave < 2; ++wave)
    for (unsigned m = 0; m < 8; ++m) {
      uint64_t hash = goc_test::capture_hash_seed;
      unsigned lanes = wave == 0 ? 32 : 64;
      for (uint16_t offset : offsets) {
        uint32_t input[64], output[64];
        for (unsigned lane = 0; lane < lanes; ++lane) {
          input[lane] = lane + 1;
          output[lane] = 0xdead0000U + lane;
        }
        const uint32_t *a = input;
        uint32_t *d = output;
        ASSERT_EQ(run(lanes, masks[wave][m], 0, &d, &a, offset), GOC_SUCCESS);
        for (unsigned lane = 0; lane < lanes; ++lane)
          hash = goc_test::capture_hash_word(hash, output[lane]);
      }
      EXPECT_EQ(hash, goc_test::ds_swizzle_hashes[wave][m]) << wave << '/' << m;
    }
}

TEST(DsSwizzle, MaskedAliasesValidationAndEmptyExec) {
  for (unsigned lanes : {32U, 64U}) {
    for (unsigned active = 0; active < lanes; ++active) {
      uint32_t words[64];
      std::fill_n(words, 64, 0xdeadbeefU);
      uint32_t *d = words;
      ASSERT_EQ(run(lanes, 1ULL << active, 0, &d, &d, 0xc020), GOC_SUCCESS);
      for (unsigned lane = 0; lane < lanes; ++lane)
        EXPECT_EQ(words[lane], lane == active ? 0 : 0xdeadbeefU);
    }
    for (unsigned bit = 0; bit < 64; ++bit)
      EXPECT_EQ(run(lanes, UINT64_MAX, 1ULL << bit, nullptr, nullptr, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(run(lanes, 0, 0, nullptr, nullptr, 65535), GOC_SUCCESS);
  }
}

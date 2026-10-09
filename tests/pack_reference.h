// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

// gfx1201 captures, 2026-10-09. Two SAT destination halves, then 64 PACK
// combinations (ABS A/B, NEG A/B, HIGH A/B). Each digest covers 65536 inputs:
// A=i|((65535-i)<<16), B=(i*0x7395a831)^0xa7925163, initial D=0xcafebeef.
// FNV-1a consumes four little-endian bytes per result. PACK quiets sNaNs.
static const uint64_t pack_capture_digests[66] = {
    0xbbed16c4bfb9f925ULL, 0xa81ec3accda5c525ULL, 0x0b4bd1b7f33ae4a5ULL, 0xe9e1dc3821d04da5ULL,
    0x649252bd168e8aa5ULL, 0xd02ef5cc793585a5ULL, 0xda082f3f3403c0a5ULL, 0xfb7224bf056e57a5ULL,
    0xcd25e6bb59f334a5ULL, 0x618943abf74c39a5ULL, 0x6cd528a9b6efcaa5ULL, 0xc140a2655fbeb3a5ULL,
    0x5083b0472433cca5ULL, 0xd5e5ba8d8261d7a5ULL, 0x0473dbe23257c0a5ULL, 0xb00862268988d7a5ULL,
    0x4b97ef9b066e24a5ULL, 0xc635e554a84019a5ULL, 0x81e6d07afa7ad275ULL, 0x82766813fc91eb75ULL,
    0xc62c82d460e2f475ULL, 0x51c36001927f9575ULL, 0x08ededb1e6dad675ULL, 0x085e5618e4c3bd75ULL,
    0x26214ae6f7d2ce75ULL, 0x9a8a6db9c6362d75ULL, 0xe9d9eb319a144f75ULL, 0xa0bb6cdab356a475ULL,
    0x1dafcdca169aef75ULL, 0xbcbba4fb21d6e475ULL, 0x3bd128db9ec60975ULL, 0x84efa7328583b475ULL,
    0xfcde1b79d4c2fb75ULL, 0x5dd24448c9870675ULL, 0xdc3e3f5e0b74a44eULL, 0x91cf894ceea11b4eULL,
    0x75667d3971261cceULL, 0x63211f9e8fed69ceULL, 0x339dbd725ee8644eULL, 0x7e0c73837bbbed4eULL,
    0x65a74540ccc7ceceULL, 0x77eca2dbae0081ceULL, 0x23f07e6136f7cf4eULL, 0x928cd6004cd23e4eULL,
    0x3f12f7d76eba53ceULL, 0x1fc346b4f7c668ceULL, 0xe387014ff1c98d4eULL, 0x74eaa9b0dbef1e4eULL,
    0xf02b96699a80c3ceULL, 0x0f7b478c1174aeceULL, 0x7040fe72a61e1bdaULL, 0x5cd241d0810ecfdaULL,
    0xeb8bebc35387065aULL, 0x9863f1ac4a6fba5aULL, 0x4e36ef36668437daULL, 0x61a5abd88b9383daULL,
    0x90a0d8fb72306a5aULL, 0xe3c8d3127b47b65aULL, 0xfbe9cf6a6e45afdaULL, 0x4fa1e0e198f64fdaULL,
    0x86c5c41bb588765aULL, 0xbf97c40aaf28425aULL, 0x39b4104f37dd7fdaULL, 0xe5fbfed80d2cdfdaULL,
    0x12b12cfe0886a65aULL, 0xd9df2d0f0ee6da5aULL,
};

inline uint32_t pack_mode(unsigned variant) {
  if (variant < 2)
    return variant ? GOC_ALU_HIGH_D : 0;
  unsigned m = variant - 2;
  return (m & 1 ? GOC_ALU_ABS_A : 0) | (m & 2 ? GOC_ALU_ABS_B : 0) | (m & 4 ? GOC_ALU_NEG_A : 0) |
         (m & 8 ? GOC_ALU_NEG_B : 0) | (m & 16 ? GOC_ALU_HIGH_A : 0) |
         (m & 32 ? GOC_ALU_HIGH_B : 0);
}

// Independently classify signed halves and manipulate individual FP16 fields.
inline uint32_t pack_reference(unsigned variant, uint32_t a, uint32_t b, uint32_t d) {
  if (variant < 2) {
    uint32_t packed = 0;
    for (unsigned h = 0; h < 2; ++h) {
      unsigned x = (a >> (16 * h)) & 65535;
      unsigned byte = (x & 32768) ? 0 : (x > 255 ? 255 : x);
      packed |= byte << (8 * h);
    }
    return variant ? (d & 65535) | (packed << 16) : (d & 0xffff0000) | packed;
  }
  unsigned m = variant - 2;
  uint32_t out = 0;
  for (unsigned h = 0; h < 2; ++h) {
    unsigned x = ((h ? b : a) >> ((m & (16u << h)) ? 16 : 0)) & 65535;
    unsigned sign = x >> 15, exponent = (x >> 10) & 31, fraction = x & 1023;
    if (exponent == 31 && fraction)
      fraction |= 512;
    if (m & (1u << h))
      sign = 0;
    if (m & (4u << h))
      sign ^= 1;
    out |= (sign * 32768 + exponent * 1024 + fraction) << (16 * h);
  }
  return out;
}

inline int pack_call(unsigned variant, uint64_t flags, uint32_t exec_mask, uint64_t mode,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return variant < 2 ? goc_v_sat_pk_u8_i16(flags, exec_mask, mode, d, a)
                     : goc_v_pack_b32_f16(flags, exec_mask, mode, d, a, b, nullptr);
}

} // namespace goc_test

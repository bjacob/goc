// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

#if defined(GOC_HAVE_X86_64_V3)
void half_dot_x86_64_v3(bool bf16, bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c);

void fp8_wmma_x86_64_v3(bool bf8_a, bool bf8_b, uint32_t exec_mask, uint32_t modifiers,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);

void fp8_dot_x86_64_v3(bool bf8_a, bool bf8_b, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                       const uint32_t *a, const uint32_t *b, const uint32_t *c);

void dot2_x86_64_v3(bool bf16, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                    const uint32_t *a, const uint32_t *b, const uint32_t *c);

void integer_dot_x86_64_v3(int bits, bool unsigned_acc, uint32_t exec_mask, uint32_t modifiers,
                           uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

void integer_wmma_x86_64_v3(int bits, int k, uint32_t exec_mask, uint32_t modifiers,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

void wmma_f16_x86_64_v3(uint32_t exec_mask, uint32_t modifiers, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);

void wmma_bf16_x86_64_v3(uint32_t exec_mask, uint32_t modifiers, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);

template <bool Multiply>
void literal_fma_x86_64_v3(uint32_t exec_mask, uint32_t literal, uint32_t *d, const uint32_t *a,
                           const uint32_t *b);

void fma_x86_64_v3(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a,
                   const uint32_t *b, const uint32_t *c);

void fma_dx9_zero_x86_64_v3(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a,
                            const uint32_t *b, const uint32_t *c);
#endif

#if defined(GOC_HAVE_X86_64_V4)
template <bool Multiply>
void literal_fma_x86_64_v4(uint32_t exec_mask, uint32_t literal, uint32_t *d, const uint32_t *a,
                           const uint32_t *b);

void fma_x86_64_v4(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a,
                   const uint32_t *b, const uint32_t *c);

void fma_dx9_zero_x86_64_v4(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a,
                            const uint32_t *b, const uint32_t *c);
#endif

#if defined(GOC_HAVE_AVX512BF16)
// Whether all inputs satisfy the conservative DPBF16 fast-path bounds.
bool wmma_inputs_avx512bf16(const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

void wmma_avx512bf16(uint32_t exec_mask, uint32_t modifiers, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);
#endif

#if defined(GOC_HAVE_AVX512VNNI)
void integer_wmma_avx512vnni(int bits, int k, uint32_t exec_mask, uint32_t modifiers,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c);
#endif

} // namespace goc

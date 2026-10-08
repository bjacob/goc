// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

#if defined(GOC_HAVE_X86_64_V3)
void integer_wmma_x86_64_v3(int bits, int k, uint32_t mask, uint32_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

void wmma_f16_x86_64_v3(uint32_t mask, uint32_t modifiers, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);

void wmma_bf16_x86_64_v3(uint32_t mask, uint32_t modifiers, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);

void fma_x86_64_v3(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                   const uint32_t *c);
#endif

#if defined(GOC_HAVE_X86_64_V4)
void fma_x86_64_v4(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                   const uint32_t *c);
#endif

#if defined(GOC_HAVE_AVX512BF16)
// Whether all inputs satisfy the conservative DPBF16 fast-path bounds.
bool wmma_inputs_avx512bf16(const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

void wmma_avx512bf16(uint32_t mask, uint32_t modifiers, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);
#endif

#if defined(GOC_HAVE_AVX512VNNI)
void integer_wmma_avx512vnni(int bits, int k, uint32_t mask, uint32_t modifiers, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c);
#endif

} // namespace goc

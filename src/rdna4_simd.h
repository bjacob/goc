// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

#if defined(GOC_HAVE_X86_64_V3)
void wmma_f16_x86_64_v3(uint32_t mask, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);

void fma_x86_64_v3(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                   const uint32_t *c);
#endif

#if defined(GOC_HAVE_X86_64_V4)
void fma_x86_64_v4(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                   const uint32_t *c);
#endif

#if defined(GOC_HAVE_AVX512BF16)
void wmma_avx512bf16(uint32_t mask, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c);
#endif

} // namespace goc

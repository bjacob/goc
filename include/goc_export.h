// SPDX-License-Identifier: MIT

#ifndef GOC_EXPORT_H_
#define GOC_EXPORT_H_

// Public entry points remain visible when implementation visibility is hidden.
#if defined(__GNUC__) || defined(__clang__)
#ifdef __cplusplus
#define GOC_API [[gnu::visibility("default")]]
#else
#define GOC_API __attribute__((visibility("default")))
#endif
#else
#define GOC_API
#endif

#endif

// SPDX-License-Identifier: MIT

#ifndef GOC_EXPORT_H_
#define GOC_EXPORT_H_

// Static consumers keep GoC private; shared consumers import its public API.
#if defined(GOC_STATIC_DEFINE)
#define GOC_API
#elif defined(_WIN32)
#if defined(GOC_BUILDING_LIBRARY)
#define GOC_API __declspec(dllexport)
#else
#define GOC_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#ifdef __cplusplus
#define GOC_API [[gnu::visibility("default")]]
#else
#define GOC_API __attribute__((visibility("default")))
#endif
#else
#define GOC_API
#endif

#endif

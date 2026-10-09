// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// GFX1201, HIP 7.13, 4096 boundary/random input pairs per variant.
// Order: incoming SCC, variant; each digest folds result low/high and SCC.
// Full, empty, and alternating EXEC captures were identical.
// Variant 0..17: ordinary ops; 18..28: ADDK; 29..39: MULK literals.
namespace goc_test {

inline const uint64_t scalar_integer_hardware[2][40] = {
    {
        0xa75ccf4aa71301b6ULL, 0x0131a2ddde88d554ULL, 0xbcfad5781b0c0254ULL, 0x2ad38622d9b72518ULL,
        0xa75ccf4aa71301b6ULL, 0x0131a2ddde88d554ULL, 0x432158f91cc0b94bULL, 0x6ec8325414e51fe5ULL,
        0x3267d8276705ea8eULL, 0x818c0d41cba15ea7ULL, 0xda02679d5705ec36ULL, 0x412a897a83efa627ULL,
        0xe6856ec806acc1d8ULL, 0xecbb94cc1fea53d3ULL, 0xd77cad446a7054c6ULL, 0x1cc475c66d8df846ULL,
        0xd01bbd764edf2c2cULL, 0x981e4594d1add184ULL, 0xe050fbef7f272625ULL, 0x1d33187e0304ce25ULL,
        0xe423ca59d588cb05ULL, 0x810d80300f91b265ULL, 0x09660bed10c55b05ULL, 0x9428afed41d03245ULL,
        0xeb584e6757080825ULL, 0x6ee86681f2b94445ULL, 0xea62601c2010ce85ULL, 0x170f9dc257bdea45ULL,
        0x5e1bc55f417ec7c5ULL, 0x6e431751526de325ULL, 0xe050fbef7f272625ULL, 0x92bc97acea3fc825ULL,
        0xebf00cd5500cc825ULL, 0x5e2c26c1aa8ee325ULL, 0xdeac7164fb233725ULL, 0x99785871d2422625ULL,
        0x5edf513c6788e125ULL, 0xa85e1d4081543725ULL, 0x34689e779dd44125ULL, 0xd851910ef9854325ULL,
    },
    {
        0xa75ccf4aa71301b6ULL, 0x0131a2ddde88d554ULL, 0xbcfad5781b0c0254ULL, 0x2ad38622d9b72518ULL,
        0xfb8a346d5f90467cULL, 0x1a4349b70cc89f68ULL, 0x432158f91cc0b94bULL, 0x6ec8325414e51fe5ULL,
        0x3267d8276705ea8eULL, 0x818c0d41cba15ea7ULL, 0xda02679d5705ec36ULL, 0x412a897a83efa627ULL,
        0x4aaab6888813cafcULL, 0xe803192df0a6d937ULL, 0x5fc2721aeb3742daULL, 0x020c4b4d6688b7c2ULL,
        0x031f9e2858406288ULL, 0xaf6ec68499595b5cULL, 0xe050fbef7f272625ULL, 0x1d33187e0304ce25ULL,
        0xe423ca59d588cb05ULL, 0x810d80300f91b265ULL, 0x09660bed10c55b05ULL, 0x9428afed41d03245ULL,
        0xeb584e6757080825ULL, 0x6ee86681f2b94445ULL, 0xea62601c2010ce85ULL, 0x170f9dc257bdea45ULL,
        0x5e1bc55f417ec7c5ULL, 0xb299d820475d3325ULL, 0x9a391ab4a76a7f25ULL, 0x141db7d69fbfdc25ULL,
        0x32d46cab362ddc25ULL, 0x42cb48395b703325ULL, 0xc743dd85dc42cd25ULL, 0x25adeba81cd27f25ULL,
        0xaa42d2fc521509a5ULL, 0xb9246fb0c27dcd25ULL, 0x6a5ffc60ff6f7a25ULL, 0xb357eeb97bce8a25ULL,
    },
};

} // namespace goc_test

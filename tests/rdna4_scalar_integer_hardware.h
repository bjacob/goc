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
        UINT64_C(0xa75ccf4aa71301b6), UINT64_C(0x0131a2ddde88d554), UINT64_C(0xbcfad5781b0c0254),
        UINT64_C(0x2ad38622d9b72518), UINT64_C(0xa75ccf4aa71301b6), UINT64_C(0x0131a2ddde88d554),
        UINT64_C(0x432158f91cc0b94b), UINT64_C(0x6ec8325414e51fe5), UINT64_C(0x3267d8276705ea8e),
        UINT64_C(0x818c0d41cba15ea7), UINT64_C(0xda02679d5705ec36), UINT64_C(0x412a897a83efa627),
        UINT64_C(0xe6856ec806acc1d8), UINT64_C(0xecbb94cc1fea53d3), UINT64_C(0xd77cad446a7054c6),
        UINT64_C(0x1cc475c66d8df846), UINT64_C(0xd01bbd764edf2c2c), UINT64_C(0x981e4594d1add184),
        UINT64_C(0xe050fbef7f272625), UINT64_C(0x1d33187e0304ce25), UINT64_C(0xe423ca59d588cb05),
        UINT64_C(0x810d80300f91b265), UINT64_C(0x09660bed10c55b05), UINT64_C(0x9428afed41d03245),
        UINT64_C(0xeb584e6757080825), UINT64_C(0x6ee86681f2b94445), UINT64_C(0xea62601c2010ce85),
        UINT64_C(0x170f9dc257bdea45), UINT64_C(0x5e1bc55f417ec7c5), UINT64_C(0x6e431751526de325),
        UINT64_C(0xe050fbef7f272625), UINT64_C(0x92bc97acea3fc825), UINT64_C(0xebf00cd5500cc825),
        UINT64_C(0x5e2c26c1aa8ee325), UINT64_C(0xdeac7164fb233725), UINT64_C(0x99785871d2422625),
        UINT64_C(0x5edf513c6788e125), UINT64_C(0xa85e1d4081543725), UINT64_C(0x34689e779dd44125),
        UINT64_C(0xd851910ef9854325),
    },
    {
        UINT64_C(0xa75ccf4aa71301b6), UINT64_C(0x0131a2ddde88d554), UINT64_C(0xbcfad5781b0c0254),
        UINT64_C(0x2ad38622d9b72518), UINT64_C(0xfb8a346d5f90467c), UINT64_C(0x1a4349b70cc89f68),
        UINT64_C(0x432158f91cc0b94b), UINT64_C(0x6ec8325414e51fe5), UINT64_C(0x3267d8276705ea8e),
        UINT64_C(0x818c0d41cba15ea7), UINT64_C(0xda02679d5705ec36), UINT64_C(0x412a897a83efa627),
        UINT64_C(0x4aaab6888813cafc), UINT64_C(0xe803192df0a6d937), UINT64_C(0x5fc2721aeb3742da),
        UINT64_C(0x020c4b4d6688b7c2), UINT64_C(0x031f9e2858406288), UINT64_C(0xaf6ec68499595b5c),
        UINT64_C(0xe050fbef7f272625), UINT64_C(0x1d33187e0304ce25), UINT64_C(0xe423ca59d588cb05),
        UINT64_C(0x810d80300f91b265), UINT64_C(0x09660bed10c55b05), UINT64_C(0x9428afed41d03245),
        UINT64_C(0xeb584e6757080825), UINT64_C(0x6ee86681f2b94445), UINT64_C(0xea62601c2010ce85),
        UINT64_C(0x170f9dc257bdea45), UINT64_C(0x5e1bc55f417ec7c5), UINT64_C(0xb299d820475d3325),
        UINT64_C(0x9a391ab4a76a7f25), UINT64_C(0x141db7d69fbfdc25), UINT64_C(0x32d46cab362ddc25),
        UINT64_C(0x42cb48395b703325), UINT64_C(0xc743dd85dc42cd25), UINT64_C(0x25adeba81cd27f25),
        UINT64_C(0xaa42d2fc521509a5), UINT64_C(0xb9246fb0c27dcd25), UINT64_C(0x6a5ffc60ff6f7a25),
        UINT64_C(0xb357eeb97bce8a25),
    },
};

} // namespace goc_test

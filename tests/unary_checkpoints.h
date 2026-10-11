// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// Cumulative word-wise FNV after each 2^20 inputs in the dense domains.
// Generated from rocjitsu commit ffc144c564c450b7847ae0528410dc63af9725a8.
static const uint64_t unary_normalized_checkpoints[5][32] = {
    {
        0xc6ef36d60ddef8beULL,
        0x418bc5565a0b1f34ULL,
        0xc2f8db7451efa48fULL,
        0xaef3bc444bf250cbULL,
        0xccae965f44df62a8ULL,
        0x3a6a8e7ce1dc4bd9ULL,
        0xca0e335c2d0d2ee2ULL,
        0xd54ec24992572df9ULL,
    },
    {
        0x3ccaa5c665226d1fULL,
        0xefb1f5bf96cc704cULL,
        0xc0d19d152c815b79ULL,
        0xceb47eb4a6cf94bdULL,
        0xa07a3cf5b5d3f64dULL,
        0x492acbaa013926bfULL,
        0xeb3f8d0382a2837eULL,
        0xe7ee2e05159b3446ULL,
        0xf824ff248d964102ULL,
        0x9a7ca3ac2234acb0ULL,
        0x9c843983472f16cfULL,
        0xb0fa7b4d286e2f46ULL,
        0x46d70f9c8264c173ULL,
        0x805cf8a9a82ae794ULL,
        0xdc54b2388002068bULL,
        0x514d49d31f12176bULL,
    },
    {
        0x73f1f9d417ad77e8ULL,
        0x767ea3fc495e6935ULL,
        0xa4935cd3d822d5acULL,
        0x4be8094cd7c6f3ecULL,
        0x97541021749d02cfULL,
        0x2a0143ddc90cf893ULL,
        0x91677eab02e1c9f1ULL,
        0xf9cba0d07818ac98ULL,
        0x795334e37dc25a7bULL,
        0x1b8021e053133925ULL,
        0x2b9b5cf577d20356ULL,
        0x85e959704fe77d42ULL,
        0xab72723f1c860b45ULL,
        0xe53bccc3d264b8d7ULL,
        0x4860c5e2f67ccd3bULL,
        0x010bc79eb6e48cafULL,
    },
    {
        0xc72337a74bef768bULL, 0x8955b56152652967ULL, 0x4b117c52b9b2e234ULL, 0x2bdbb128f5f39d54ULL,
        0xd6f881859c811609ULL, 0x10363f450494ed5aULL, 0x499e1d11c67d2e3eULL, 0x0820cd2c217e3539ULL,
        0x2f27a76c6818229aULL, 0x42b3c4a07455a04fULL, 0x8506318e4e2aac02ULL, 0xb0621156790db21eULL,
        0x32fb680fae66e0a3ULL, 0x98a2994283c5679bULL, 0x35950097aaa3203eULL, 0x3622759937fb8914ULL,
        0xbd4cb8c992401e60ULL, 0xe0da3f3541275fa2ULL, 0x198421ddb5fe18a3ULL, 0x113a56c43a46f00bULL,
        0xea2fd7a04d188a5bULL, 0xd6be39d587150c91ULL, 0xea5e17507a24be18ULL, 0x06809a7692248ae2ULL,
        0xde6da9af9385d6fcULL, 0xe1407f47be4ff173ULL, 0xfdba91d8c7e9a979ULL, 0x99861ad6106fac8cULL,
        0xa41def8e09ba1d34ULL, 0xe96cd2f64e88eecdULL, 0xe3261a72442094a0ULL, 0x6d5d8b062135ef78ULL,
    },
    {
        0x8858e4dce2b1a75dULL,
        0xb343451030b02c12ULL,
        0x8796b2f0a8d77e83ULL,
        0x955a42945c87cafaULL,
        0x847ec06ff400f029ULL,
        0xeba9555ddab7616aULL,
        0x0575747f4656585eULL,
        0xebf1ee45bcf3963bULL,
        0x501a120481e3116dULL,
        0x62722ee7310fd4e7ULL,
        0x009588aca8fa40dcULL,
        0x9a552b0453b0037cULL,
        0x63a8386fc85f146aULL,
        0xa4c50780e9c26455ULL,
        0x264e898fad67f31dULL,
        0xc33a54efbae098beULL,
    },
};

} // namespace goc_test

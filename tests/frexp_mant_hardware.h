// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// gfx1201 capture_unary_exact.py frexp_mant --exceptions: 8192 inputs per
// modifier, four denormal policies. The two output-denormal policies give
// identical pairs. Each row contains (canonical-NaN result hash, raw pair hash).
inline constexpr uint64_t frexp_mant_hardware[2][32][2] = {
    {
        {0xd54cd6ede638beULL, 0x3d366df9f5340418ULL},
        {0x983f2ffbede638beULL, 0xeea5ff17f5340418ULL},
        {0x18fe397ede638beULL, 0xa290a3ecf5340418ULL},
        {0x356730dcede638beULL, 0xcdc2f0a8f5340418ULL},
        {0x2fb99fe6336638beULL, 0x965510c823b40418ULL},
        {0xd54de186b36638beULL, 0xc80a260a3b40418ULL},
        {0xbf10afb1b36638beULL, 0x435b6463a3b40418ULL},
        {0x30449b61336638beULL, 0xf51f7f7723b40418ULL},
        {0xe4c7a47282e638beULL, 0x7231e4df04340418ULL},
        {0x1639ac2302e638beULL, 0x3e11835184340418ULL},
        {0xc78e851102e638beULL, 0x549d2de384340418ULL},
        {0x4b9d0bfc82e638beULL, 0x1188d04d04340418ULL},
        {0x33dfc128de6638beULL, 0xecc4bbb44cb40418ULL},
        {0xce720f6a5e6638beULL, 0xa59b32fcccb40418ULL},
        {0x47d69995e6638beULL, 0xc6401470ccb40418ULL},
        {0xb93cb1c7de6638beULL, 0xa5ea3b084cb40418ULL},
        {0xd01fb792feaa4de6ULL, 0xf361e93dd04ef95eULL},
        {0x82c5139f7e7270bdULL, 0x6a0371bd67e77a0dULL},
        {0x47086aa6506638beULL, 0x7c060bc3d1652166ULL},
        {0xb9d103fd6854a325ULL, 0x9c1bda7f8c872325ULL},
        {0x56ed03a74354a325ULL, 0xe529155b14872325ULL},
        {0x1ea31c9c7d4a325ULL, 0x53be999a65072325ULL},
        {0x59356c90ed4a325ULL, 0x52d47782bb072325ULL},
        {0xb9d103fd6854a325ULL, 0x9c1bda7f8c872325ULL},
        {0x56ed03a74354a325ULL, 0xe529155b14872325ULL},
        {0x1ea31c9c7d4a325ULL, 0x53be999a65072325ULL},
        {0x59356c90ed4a325ULL, 0x52d47782bb072325ULL},
        {0xb9d103fd6854a325ULL, 0x9c1bda7f8c872325ULL},
        {0x348f273b272a4de6ULL, 0x63eba6bdfccef95eULL},
        {0x613d9087077270bdULL, 0xe636c11d90e77a0dULL},
        {0xb4c3eec9b8e638beULL, 0x9cec5fb7f8e52166ULL},
        {0xb9d103fd6854a325ULL, 0x9c1bda7f8c872325ULL},
    },
    {
        {0x49e95cb15866b094ULL, 0xf7bf6a971fd10174ULL},
        {0x7d39d6d05866b094ULL, 0xf5d8dc3d1fd10174ULL},
        {0x337a46895866b094ULL, 0x9cbf4d201fd10174ULL},
        {0xc52403865866b094ULL, 0x591bc1701fd10174ULL},
        {0xc063edaa6866b094ULL, 0x7c04d29e19d10174ULL},
        {0xd9c45e7d6866b094ULL, 0xee3f451419d10174ULL},
        {0x58983e816866b094ULL, 0x825347a19d10174ULL},
        {0xdfb61f386866b094ULL, 0xe47c1d5019d10174ULL},
        {0x12163664ae66b094ULL, 0x4b00645de3d10174ULL},
        {0xccc0aacdae66b094ULL, 0x7da3c026e3d10174ULL},
        {0x92a25ecfae66b094ULL, 0xa0beca6de3d10174ULL},
        {0x817a7d60ae66b094ULL, 0xfdc246c6e3d10174ULL},
        {0x5192ffd81266b094ULL, 0x8c8b71748dd10174ULL},
        {0xdf6904481266b094ULL, 0xb632be3e8dd10174ULL},
        {0x28f11e711266b094ULL, 0x98d7f228dd10174ULL},
        {0xa251b09d1266b094ULL, 0x4f5098608dd10174ULL},
        {0x466ddb6b6571f82aULL, 0x73f21020d591ee96ULL},
        {0x61b33e6efff57bf3ULL, 0x6840e426ab218a43ULL},
        {0x27c6e82d4a66b094ULL, 0x8d706d5c2915b010ULL},
        {0xb9d103fd6854a325ULL, 0x9c1bda7f8c872325ULL},
        {0xd541939add54a325ULL, 0x351c8c0b61872325ULL},
        {0x7a94d8586254a325ULL, 0x3e82a55406872325ULL},
        {0x9398e88a0554a325ULL, 0xa416ec713d872325ULL},
        {0xb9d103fd6854a325ULL, 0x9c1bda7f8c872325ULL},
        {0xd541939add54a325ULL, 0x351c8c0b61872325ULL},
        {0x7a94d8586254a325ULL, 0x3e82a55406872325ULL},
        {0x9398e88a0554a325ULL, 0xa416ec713d872325ULL},
        {0xb9d103fd6854a325ULL, 0x9c1bda7f8c872325ULL},
        {0x5af175c249f1f82aULL, 0x250cea94e511ee96ULL},
        {0xd5485e4a65757bf3ULL, 0x5404d13896a18a43ULL},
        {0x95cbb05caa66b094ULL, 0x7f8e9ffa9315b010ULL},
        {0xb9d103fd6854a325ULL, 0x9c1bda7f8c872325ULL},
    },
};

} // namespace goc_test

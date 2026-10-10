// SPDX-License-Identifier: MIT
// Standalone HIP capture: hipcc --offload-arch=gfx1201 -mwavefrontsize64.
#include "integer_wmma_capture.h"

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>

#define HIP(call)                                                                                  \
  do {                                                                                             \
    auto e = (call);                                                                               \
    if (e != hipSuccess) {                                                                         \
      std::fprintf(stderr, "%s\n", hipGetErrorString(e));                                          \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)
using Int4 = int __attribute__((ext_vector_type(4)));

template <unsigned Shape, unsigned Mode>
__global__ void capture(const uint32_t *input, uint32_t *output) {
  unsigned lane = threadIdx.x, sample = blockIdx.x;
  const uint32_t *v = input + sample * 15 * 64;
  uint32_t a = v[lane], index = v[14 * 64 + lane];
  uint64_t b = uint64_t(v[2 * 64 + lane]) | (uint64_t(v[3 * 64 + lane]) << 32);
  Int4 d;
  for (unsigned reg = 0; reg < 4; ++reg)
    d[reg] = v[(6 + reg) * 64 + lane];
  if constexpr (Shape == 0) {
    if constexpr (Mode & 4)
      asm volatile("v_swmmac_i32_16x16x32_iu8 %0, %1, %2, %3 index_key:%6 neg_lo:[%4,%5,0] clamp"
                   : "+v"(d)
                   : "v"(a), "v"(b), "v"(index), "n"(bool(Mode & 1)), "n"(bool(Mode & 2)),
                     "n"(Mode >> 3));
    else
      asm volatile("v_swmmac_i32_16x16x32_iu8 %0, %1, %2, %3 index_key:%6 neg_lo:[%4,%5,0]"
                   : "+v"(d)
                   : "v"(a), "v"(b), "v"(index), "n"(bool(Mode & 1)), "n"(bool(Mode & 2)),
                     "n"(Mode >> 3));
  } else if constexpr (Shape == 1) {
    if constexpr (Mode & 4)
      asm volatile("v_swmmac_i32_16x16x32_iu4 %0, %1, %2, %3 index_key:%6 neg_lo:[%4,%5,0] clamp"
                   : "+v"(d)
                   : "v"(a), "v"(uint32_t(b)), "v"(index), "n"(bool(Mode & 1)), "n"(bool(Mode & 2)),
                     "n"(Mode >> 3));
    else
      asm volatile("v_swmmac_i32_16x16x32_iu4 %0, %1, %2, %3 index_key:%6 neg_lo:[%4,%5,0]"
                   : "+v"(d)
                   : "v"(a), "v"(uint32_t(b)), "v"(index), "n"(bool(Mode & 1)), "n"(bool(Mode & 2)),
                     "n"(Mode >> 3));
  } else if constexpr (Shape == 2) {
    if constexpr (Mode & 4)
      asm volatile("v_swmmac_i32_16x16x64_iu4 %0, %1, %2, %3 index_key:%6 neg_lo:[%4,%5,0] clamp"
                   : "+v"(d)
                   : "v"(a), "v"(b), "v"(index), "n"(bool(Mode & 1)), "n"(bool(Mode & 2)),
                     "n"(Mode >> 3));
    else
      asm volatile("v_swmmac_i32_16x16x64_iu4 %0, %1, %2, %3 index_key:%6 neg_lo:[%4,%5,0]"
                   : "+v"(d)
                   : "v"(a), "v"(b), "v"(index), "n"(bool(Mode & 1)), "n"(bool(Mode & 2)),
                     "n"(Mode >> 3));
  }
  for (unsigned reg = 0; reg < 4; ++reg)
    output[sample * 256 + reg * 64 + lane] = d[reg];
}

template <unsigned Shape, unsigned Mode> void run(const uint32_t *input, uint32_t *device_output) {
  hipLaunchKernelGGL((capture<Shape, Mode>), dim3(16), dim3(64), 0, 0, input, device_output);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  uint32_t output[16][256];
  HIP(hipMemcpy(output, device_output, sizeof output, hipMemcpyDeviceToHost));
  for (unsigned sample = 0; sample < 16; ++sample) {
    uint64_t hash = 14695981039346656037ULL;
    for (auto word : output[sample])
      hash = (hash ^ word) * 1099511628211ULL;
    std::printf("%u %u %u %016llx\n", Shape, sample, Mode, (unsigned long long)hash);
  }
}

int main() {
  uint32_t input[16][15][64];
  for (unsigned sample = 0; sample < 16; ++sample)
    goc_test::integer_wmma_capture_inputs(sample, input[sample]);
  uint32_t *device_input, *device_output;
  HIP(hipMalloc(&device_input, sizeof input));
  HIP(hipMalloc(&device_output, 16 * 256 * sizeof(uint32_t)));
  HIP(hipMemcpy(device_input, input, sizeof input, hipMemcpyHostToDevice));
  run<0, 0>(device_input, device_output);
  run<0, 1>(device_input, device_output);
  run<0, 2>(device_input, device_output);
  run<0, 3>(device_input, device_output);
  run<0, 4>(device_input, device_output);
  run<0, 5>(device_input, device_output);
  run<0, 6>(device_input, device_output);
  run<0, 7>(device_input, device_output);
  run<0, 8>(device_input, device_output);
  run<0, 9>(device_input, device_output);
  run<0, 10>(device_input, device_output);
  run<0, 11>(device_input, device_output);
  run<0, 12>(device_input, device_output);
  run<0, 13>(device_input, device_output);
  run<0, 14>(device_input, device_output);
  run<0, 15>(device_input, device_output);
  run<0, 16>(device_input, device_output);
  run<0, 17>(device_input, device_output);
  run<0, 18>(device_input, device_output);
  run<0, 19>(device_input, device_output);
  run<0, 20>(device_input, device_output);
  run<0, 21>(device_input, device_output);
  run<0, 22>(device_input, device_output);
  run<0, 23>(device_input, device_output);
  run<0, 24>(device_input, device_output);
  run<0, 25>(device_input, device_output);
  run<0, 26>(device_input, device_output);
  run<0, 27>(device_input, device_output);
  run<0, 28>(device_input, device_output);
  run<0, 29>(device_input, device_output);
  run<0, 30>(device_input, device_output);
  run<0, 31>(device_input, device_output);
  run<1, 0>(device_input, device_output);
  run<1, 1>(device_input, device_output);
  run<1, 2>(device_input, device_output);
  run<1, 3>(device_input, device_output);
  run<1, 4>(device_input, device_output);
  run<1, 5>(device_input, device_output);
  run<1, 6>(device_input, device_output);
  run<1, 7>(device_input, device_output);
  run<1, 8>(device_input, device_output);
  run<1, 9>(device_input, device_output);
  run<1, 10>(device_input, device_output);
  run<1, 11>(device_input, device_output);
  run<1, 12>(device_input, device_output);
  run<1, 13>(device_input, device_output);
  run<1, 14>(device_input, device_output);
  run<1, 15>(device_input, device_output);
  run<2, 0>(device_input, device_output);
  run<2, 1>(device_input, device_output);
  run<2, 2>(device_input, device_output);
  run<2, 3>(device_input, device_output);
  run<2, 4>(device_input, device_output);
  run<2, 5>(device_input, device_output);
  run<2, 6>(device_input, device_output);
  run<2, 7>(device_input, device_output);
  run<2, 8>(device_input, device_output);
  run<2, 9>(device_input, device_output);
  run<2, 10>(device_input, device_output);
  run<2, 11>(device_input, device_output);
  run<2, 12>(device_input, device_output);
  run<2, 13>(device_input, device_output);
  run<2, 14>(device_input, device_output);
  run<2, 15>(device_input, device_output);
  HIP(hipFree(device_input));
  HIP(hipFree(device_output));
}

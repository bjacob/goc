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
  int a = v[lane], b = v[2 * 64 + lane];
  Int4 c;
  for (unsigned reg = 0; reg < 4; ++reg)
    c[reg] = v[(6 + reg) * 64 + lane];
  Int4 d;
  if constexpr (Shape == 0)
    d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12(bool(Mode & 1), a, bool(Mode & 2), b, c,
                                                         bool(Mode & 4));
  else if constexpr (Shape == 1)
    d = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64_gfx12(bool(Mode & 1), a, bool(Mode & 2), b, c,
                                                         bool(Mode & 4));
  else
    d = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12(bool(Mode & 1), a, bool(Mode & 2), b, c,
                                                         bool(Mode & 4));
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
  run<1, 0>(device_input, device_output);
  run<1, 1>(device_input, device_output);
  run<1, 2>(device_input, device_output);
  run<1, 3>(device_input, device_output);
  run<1, 4>(device_input, device_output);
  run<1, 5>(device_input, device_output);
  run<1, 6>(device_input, device_output);
  run<1, 7>(device_input, device_output);
  run<2, 0>(device_input, device_output);
  run<2, 1>(device_input, device_output);
  run<2, 2>(device_input, device_output);
  run<2, 3>(device_input, device_output);
  run<2, 4>(device_input, device_output);
  run<2, 5>(device_input, device_output);
  run<2, 6>(device_input, device_output);
  run<2, 7>(device_input, device_output);
  HIP(hipFree(device_input));
  HIP(hipFree(device_output));
}

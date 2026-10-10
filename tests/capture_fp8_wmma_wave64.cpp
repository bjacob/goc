// SPDX-License-Identifier: MIT
// Standalone HIP capture: hipcc --offload-arch=gfx1201 -mwavefrontsize64.
// Small integer factors make every product and sum exactly representable.

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>

#define HIP(call)                                                                                  \
  do {                                                                                             \
    auto error = (call);                                                                           \
    if (error != hipSuccess) {                                                                     \
      std::fprintf(stderr, "%s\n", hipGetErrorString(error));                                      \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

using Float4 = float __attribute__((ext_vector_type(4)));

__global__ void capture(uint32_t *output) {
  unsigned lane = threadIdx.x;
  const unsigned fp8[] = {0x38, 0x40, 0x48, 0x50, 0xb8, 0xc0, 0xc8, 0xd0};
  const unsigned bf8[] = {0x3c, 0x40, 0x44, 0x48, 0xbc, 0xc0, 0xc4, 0xc8};
  unsigned a = 0, b = 0;
  unsigned format = blockIdx.x;
  for (unsigned byte = 0; byte < 4; ++byte) {
    unsigned x = (lane * 3 + lane / 7 + byte * 5) % 8;
    unsigned y = (lane * 5 + lane / 3 + byte * 3) % 8;
    a |= (format & 2 ? bf8[x] : fp8[x]) << (byte * 8);
    b |= (format & 1 ? bf8[y] : fp8[y]) << (byte * 8);
  }
  Float4 c;
  for (unsigned reg = 0; reg < 4; ++reg)
    c[reg] = int(lane) - 32 + 3 * int(reg);
  Float4 d;
  if (format == 0)
    d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12(a, b, c);
  else if (format == 1)
    d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w64_gfx12(a, b, c);
  else if (format == 2)
    d = __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w64_gfx12(a, b, c);
  else
    d = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w64_gfx12(a, b, c);
  for (unsigned reg = 0; reg < 4; ++reg)
    output[format * 256 + reg * 64 + lane] = __float_as_uint(d[reg]);
}

int main() {
  uint32_t *device, output[1024];
  HIP(hipMalloc(&device, sizeof output));
  hipLaunchKernelGGL(capture, dim3(4), dim3(64), 0, 0, device);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  HIP(hipMemcpy(output, device, sizeof output, hipMemcpyDeviceToHost));
  for (unsigned format = 0; format < 4; ++format) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned i = 0; i < 256; ++i)
      hash = (hash ^ output[format * 256 + i]) * 1099511628211ULL;
    std::printf("%u %016llx\n", format, (unsigned long long)hash);
  }
  HIP(hipFree(device));
}

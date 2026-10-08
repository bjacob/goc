// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_dense_golden.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdint.h>
#include <system_error>

namespace {

using Wmma = decltype(&goc_rdna4_v_wmma_f32_16x16x16_f16);

uint32_t bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

struct Registers {
  uint32_t data[24][32] = {};
  uint32_t *v[24];

  explicit Registers(bool bf16) {
    for (int reg = 0; reg < 24; ++reg)
      v[reg] = data[reg];
    const uint16_t fp16_values[] = {0xc000, 0xbc00, 0, 0x3c00, 0x4000};
    const uint16_t bf16_values[] = {0xc000, 0xbf80, 0, 0x3f80, 0x4000};
    const auto *values = bf16 ? bf16_values : fp16_values;
    std::minstd_rand random(7);
    for (int row = 0; row < 16; ++row)
      for (int k = 0; k < 16; ++k)
        data[k % 8 / 2][row + 16 * (k / 8)] |= uint32_t(values[random() % 5]) << (16 * (k % 2));
    for (int k = 0; k < 16; ++k)
      for (int col = 0; col < 16; ++col)
        data[4 + k % 8 / 2][col + 16 * (k / 8)] |= uint32_t(values[random() % 5]) << (16 * (k % 2));
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 16; ++col)
        data[8 + row % 8][col + 16 * (row / 8)] = bits(float(int(random() % 5) - 2));
  }

  bool correct() const {
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 16; ++col)
        if (data[16 + row % 8][col + 16 * (row / 8)] != bits(float(kDenseGolden[row * 16 + col])))
          return false;
    return true;
  }
};

// Returns median nanoseconds per wave, or a negative value on an API/result error.
double measure(Wmma fn, uint64_t flags, Registers &r, int iterations) {
  const auto call = [&] { return fn(flags, UINT32_MAX, 0, r.v + 16, r.v, r.v + 4, r.v + 8); };
  for (int warmup = 0; warmup < 32; ++warmup)
    if (call() != GOC_SUCCESS)
      return -1;
  if (!r.correct())
    return -1;

  std::array<double, 7> samples;
  for (double &sample : samples) {
    int status = GOC_SUCCESS;
    auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
      status |= call();
    auto end = std::chrono::steady_clock::now();
    if (status != GOC_SUCCESS || !r.correct())
      return -1;
    sample = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

bool benchmark(bool bf16, uint64_t cpu, int iterations) {
  Registers r(bf16);
  const char *format = bf16 ? "BF16" : "FP16";
  Wmma fn = bf16 ? goc_rdna4_v_wmma_f32_16x16x16_bf16 : goc_rdna4_v_wmma_f32_16x16x16_f16;
  double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations);
  if (scalar < 0)
    return false;
  std::printf("%-6s %-8s %-16s %12.1f %10.2fx\n", format, "loose", "scalar", scalar, 1.0);

  const auto accelerated = [&](const char *path, uint64_t level) {
    double time = measure(fn, level, r, iterations);
    if (time < 0)
      return false;
    std::printf("%-6s %-8s %-16s %12.1f %10.2fx\n", format, "loose", path, time, scalar / time);
    return true;
  };
  // Only label a SIMD path when both this build and the host support it.
  bool ran_simd = false;
#if defined(GOC_BENCH_HAVE_X86_64_V3)
  if (cpu >= GOC_CPU_X86_64_V3) {
    if (!accelerated("x86-64-v3", GOC_CPU_X86_64_V3))
      return false;
    ran_simd = true;
  }
#endif
#if defined(GOC_BENCH_HAVE_AVX512BF16)
  if (bf16 && cpu >= GOC_CPU_ZEN4) {
    if (!accelerated("Zen4 / AVX512BF16", GOC_CPU_ZEN4))
      return false;
    ran_simd = true;
  }
#endif
  (void)cpu;
  (void)accelerated;
  if (!ran_simd)
    std::printf("%-6s SIMD unavailable in this build or on this host; skipped.\n", format);

  double exact = measure(
      fn, GOC_CPU_BASELINE | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, r, iterations);
  if (exact < 0)
    return false;
  std::printf("%-6s %-8s %-16s %12.1f %11s\n", format, "exact", "scalar", exact, "--");
  return true;
}

} // namespace

int main(int argc, char **argv) {
  int iterations = 10000;
  if (argc > 2) {
    std::fprintf(stderr, "Usage: %s [positive iterations per sample]\n", argv[0]);
    return 2;
  }
  if (argc == 2) {
    const char *end = argv[1] + std::strlen(argv[1]);
    auto parsed = std::from_chars(argv[1], end, iterations);
    if (parsed.ec != std::errc{} || parsed.ptr != end || iterations <= 0) {
      std::fprintf(stderr, "Iterations must be a positive integer.\n");
      return 2;
    }
  }

  uint64_t cpu = goc_init_cpu_flags();
  std::printf("RDNA4 wave32 WMMA, 16x16x16, FP32 output; CPU flags 0x%llx\n",
              static_cast<unsigned long long>(cpu));
  std::printf("Median of 7 samples, %d calls/sample, after warmup.\n", iterations);
  std::puts(
      "Full EXEC, no modifiers, fixed small-integer inputs; independent dense goldens checked.");
  std::puts("Timings include public API dispatch and stores; buffers stay hot in cache.");
  std::puts("Speedups compare loose paths of the SAME format; exact is reported separately.");
  std::printf("%-6s %-8s %-16s %12s %11s\n", "Input", "Mode", "CPU path", "ns/wave", "Speedup");
  if (!benchmark(false, cpu, iterations) || !benchmark(true, cpu, iterations)) {
    std::fprintf(stderr, "Benchmark failed: API error or result mismatch.\n");
    return 1;
  }
  return 0;
}

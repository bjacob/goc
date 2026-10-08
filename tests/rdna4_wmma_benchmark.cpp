// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_dense_golden.h"
#include "rdna4_subbyte_golden.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
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
  uint32_t expected[256];

  Registers() {
    for (int reg = 0; reg < 24; ++reg)
      v[reg] = data[reg];
  }

  explicit Registers(bool bf16) : Registers() {
    for (int i = 0; i < 256; ++i)
      expected[i] = bits(float(kDenseGolden[i]));
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

  Registers(int shape, int mode) : Registers() {
    std::copy(kIntegerDense[shape][mode], kIntegerDense[shape][mode] + 256, expected);
    const int width = shape == 0 ? 8 : 4, k_size = shape == 2 ? 32 : 16;
    std::minstd_rand random(12056925);
    for (int operand = 0; operand < 2; ++operand)
      for (int i = 0; i < 16 * k_size; ++i) {
        int index = operand ? i % 16 : i / k_size;
        int k = operand ? i / 16 : i % k_size;
        int group = k / 8;
        int reg = 4 * operand + group / 2 * (width / 4) + k % 8 / (32 / width);
        int lane = index + 16 * (group % 2), shift = k % (32 / width) * width;
        data[reg][lane] |= (random() % (1u << width)) << shift;
      }
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 16; ++col) {
        int i = row * 16 + col;
        data[8 + row % 8][col + 16 * (row / 8)] = i % 4 == 0   ? 0x7ffffff0
                                                  : i % 4 == 1 ? 0x80000010
                                                               : random();
      }
  }

  bool correct() const {
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 16; ++col)
        if (data[16 + row % 8][col + 16 * (row / 8)] != expected[row * 16 + col])
          return false;
    return true;
  }
};

bool positive_integer(const char *text, int &value) {
  const char *end = text + std::strlen(text);
  auto parsed = std::from_chars(text, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end && value > 0;
}

// Returns median nanoseconds per wave, or a negative value on an API/result error
// or iteration-count overflow. Every accepted sample spans at least min_ms.
double measure(Wmma fn, uint64_t flags, Registers &r, int initial_iterations, int min_ms,
               uint32_t modifiers = 0) {
  uint64_t iterations = uint64_t(initial_iterations);
  const auto call = [&] {
    return fn(flags, UINT32_MAX, modifiers, r.v + 16, r.v, r.v + 4, r.v + 8);
  };
  for (int warmup = 0; warmup < 32; ++warmup)
    if (call() != GOC_SUCCESS)
      return -1;
  if (!r.correct())
    return -1;

  std::array<double, 7> samples;
  for (double &sample : samples) {
    for (;;) {
      int status = GOC_SUCCESS;
      auto start = std::chrono::steady_clock::now();
      for (uint64_t iteration = 0; iteration < iterations; ++iteration)
        status |= call();
      auto elapsed = std::chrono::steady_clock::now() - start;
      if (status != GOC_SUCCESS || !r.correct())
        return -1;
      if (elapsed >= std::chrono::milliseconds(min_ms)) {
        sample = std::chrono::duration<double, std::nano>(elapsed).count() / iterations;
        break;
      }
      if (iterations > UINT64_MAX / 2)
        return -1;
      iterations *= 2;
    }
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// Print one table row using the same column widths for headings and results.
void print_columns(const char *input, const char *mode, const char *path, const char *time,
                   const char *speedup) {
  std::printf("%-10s %-10s %-20s %12s %11s\n", input, mode, path, time, speedup);
}

// Print a result with fixed decimal precision; a negative speedup prints "--".
void print_result(const char *input, const char *mode, const char *path, double time,
                  double speedup) {
  char time_text[64], speedup_text[64];
  std::snprintf(time_text, sizeof(time_text), "%.1f", time);
  if (speedup < 0)
    std::snprintf(speedup_text, sizeof(speedup_text), "--");
  else
    std::snprintf(speedup_text, sizeof(speedup_text), "%.2fx", speedup);
  print_columns(input, mode, path, time_text, speedup_text);
}

bool benchmark(bool bf16, uint64_t cpu, int iterations, int min_ms) {
  Registers r(bf16);
  const char *format = bf16 ? "BF16" : "FP16";
  Wmma fn = bf16 ? goc_rdna4_v_wmma_f32_16x16x16_bf16 : goc_rdna4_v_wmma_f32_16x16x16_f16;
  double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms);
  if (scalar < 0)
    return false;
  print_result(format, "loose", "scalar", scalar, 1.0);

  const auto accelerated = [&](const char *path, uint64_t level) {
    double time = measure(fn, level, r, iterations, min_ms);
    if (time < 0)
      return false;
    print_result(format, "loose", path, time, scalar / time);
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

  double exact =
      measure(fn, GOC_CPU_BASELINE | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, r,
              iterations, min_ms);
  if (exact < 0)
    return false;
  print_result(format, "exact", "scalar", exact, -1);
  return true;
}

bool benchmark_integer(int shape, int mode, uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_wmma_i32_16x16x16_iu8, goc_rdna4_v_wmma_i32_16x16x16_iu4,
                            goc_rdna4_v_wmma_i32_16x16x32_iu4};
  const char *names[] = {"INT8/K16", "INT4/K16", "INT4/K32"};
  const char *mode_name = mode == 0 ? "u/u wrap" : "s/s clamp";
  const uint32_t modifiers = (mode & 3) | ((mode & 4) ? GOC_WMMA_CLAMP : 0);
  Registers r(shape, mode);
  double scalar = 0;
  const auto run = [&](const char *path, uint64_t level) {
    double time =
        measure(functions[shape], level | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, r,
                iterations, min_ms, modifiers);
    if (time < 0)
      return false;
    if (level == GOC_CPU_BASELINE)
      scalar = time;
    print_result(names[shape], mode_name, path, time, scalar / time);
    return true;
  };
  if (!run("scalar", GOC_CPU_BASELINE))
    return false;
#if defined(GOC_BENCH_HAVE_X86_64_V3)
  if (cpu >= GOC_CPU_X86_64_V3 && !run("x86-64-v3", GOC_CPU_X86_64_V3))
    return false;
#endif
#if defined(GOC_BENCH_HAVE_AVX512VNNI)
  if (cpu >= GOC_CPU_ZEN4 && !run("Zen4 / AVX512VNNI", GOC_CPU_ZEN4))
    return false;
#endif
  (void)cpu;
  return true;
}

} // namespace

int main(int argc, char **argv) {
  int iterations = 128;
  int min_ms = 10;
  if (argc > 2) {
    std::fprintf(stderr, "Usage: %s [positive initial iterations]\n", argv[0]);
    return 2;
  }
  if (argc == 2 && !positive_integer(argv[1], iterations)) {
    std::fprintf(stderr, "Initial iterations must be a positive integer.\n");
    return 2;
  }
  if (const char *value = std::getenv("GOC_BENCH_MIN_MS")) {
    if (!positive_integer(value, min_ms)) {
      std::fprintf(stderr, "GOC_BENCH_MIN_MS must be a positive integer in milliseconds.\n");
      return 2;
    }
  }

  uint64_t cpu = goc_init_cpu_flags();
  std::printf("RDNA4 wave32 WMMA, 16x16 output; CPU flags 0x%llx\n",
              static_cast<unsigned long long>(cpu));
  std::printf("Median of 7 samples, each at least %d ms, after warmup.\n", min_ms);
  std::printf("Start at %d calls; double until the minimum duration is reached.\n", iterations);
  std::puts(
      "Full EXEC, fixed inputs, separate C/D, hot buffers; independent dense goldens checked.");
  std::puts("Timings include public API dispatch, input conversions and output stores.");
  std::puts("FP rows: loose speedups, exact scalar separately. Integer rows: exact, speedups "
            "within each mode.");
  print_columns("Input", "Mode", "CPU path", "ns/wave", "Speedup");
  if (!benchmark(false, cpu, iterations, min_ms) || !benchmark(true, cpu, iterations, min_ms)) {
    std::fprintf(stderr,
                 "Benchmark failed: API error, result mismatch or iteration-count overflow.\n");
    return 1;
  }
  for (int shape = 0; shape < 3; ++shape)
    for (int mode : {0, 7})
      if (!benchmark_integer(shape, mode, cpu, iterations, min_ms)) {
        std::fprintf(stderr, "Integer benchmark failed: API/result error or iteration overflow.\n");
        return 1;
      }
  return 0;
}

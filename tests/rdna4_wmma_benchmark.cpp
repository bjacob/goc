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
  int output_regs = 8;

  Registers() {
    for (int reg = 0; reg < 24; ++reg)
      v[reg] = data[reg];
  }

  explicit Registers(bool bf16, uint32_t modifiers = 0) : Registers() {
    for (int i = 0; i < 256; ++i)
      expected[i] = bits(float(kDenseGolden[i]));
    const uint16_t fp16_values[] = {0xc000, 0xbc00, 0, 0x3c00, 0x4000};
    const uint16_t bf16_values[] = {0xc000, 0xbf80, 0, 0x3f80, 0x4000};
    const auto *values = bf16 ? bf16_values : fp16_values;
    int left[16][16], right[16][16];
    std::minstd_rand random(7);
    for (int row = 0; row < 16; ++row)
      for (int k = 0; k < 16; ++k) {
        int index = random() % 5;
        left[row][k] = index - 2;
        data[k % 8 / 2][row + 16 * (k / 8)] |= uint32_t(values[index]) << (16 * (k % 2));
      }
    for (int k = 0; k < 16; ++k)
      for (int col = 0; col < 16; ++col) {
        int index = random() % 5;
        right[k][col] = index - 2;
        data[4 + k % 8 / 2][col + 16 * (k / 8)] |= uint32_t(values[index]) << (16 * (k % 2));
      }
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 16; ++col) {
        int c = int(random() % 5) - 2;
        data[8 + row % 8][col + 16 * (row / 8)] = bits(float(c));
        if (modifiers) {
          // Independent integer reference for the modified small-integer matrices.
          int acc = (modifiers & GOC_WMMA_ABS_C) ? std::abs(c) : c;
          if (modifiers & GOC_WMMA_NEG_C)
            acc = -acc;
          for (int k = 0; k < 16; ++k) {
            int a = left[row][k], b = right[k][col];
            if (modifiers & (k % 2 ? GOC_WMMA_NEG_HI_A : GOC_WMMA_NEG_LO_A))
              a = -a;
            if (modifiers & (k % 2 ? GOC_WMMA_NEG_HI_B : GOC_WMMA_NEG_LO_B))
              b = -b;
            acc += a * b;
          }
          expected[row * 16 + col] = bits(float(acc));
        }
      }
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

  void initialize_fp8_wmma(int format, uint32_t modifiers) {
    std::minstd_rand random(12056925);
    for (int operand = 0; operand < 2; ++operand) {
      bool bf8 = operand ? format & 1 : format & 2;
      for (int i = 0; i < 256; ++i) {
        uint32_t x = random();
        uint32_t code = (x & 128) | ((bf8 ? 48 : 32) + x % (bf8 ? 24 : 48));
        int index = operand ? i % 16 : i / 16, k = operand ? i / 16 : i % 16;
        data[4 * operand + (k % 8) / 4][index + 16 * (k / 8)] |= code << (8 * (k % 4));
      }
    }
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 16; ++col) {
        int c = int(random() % 33) - 16;
        data[8 + row % 8][col + 16 * (row / 8)] = bits(float(c));
        int adjusted = modifiers & GOC_WMMA_ABS_C ? std::abs(c) : c;
        if (modifiers & GOC_WMMA_NEG_C)
          adjusted = -adjusted;
        float golden;
        std::memcpy(&golden, &kFp8Dense[format][row * 16 + col], sizeof(golden));
        expected[row * 16 + col] = bits(golden + float(adjusted - c));
      }
  }

  void initialize_fma(uint32_t modifiers, bool dx9) {
    output_regs = 1;
    std::minstd_rand random(31);
    for (int lane = 0; lane < 32; ++lane) {
      int a = int(random() % 33) - 16, b = int(random() % 33) - 16, c = int(random() % 33) - 16;
      data[0][lane] = bits(float(a));
      data[4][lane] = bits(float(b));
      data[8][lane] = bits(float(c));
      if (modifiers) {
        a = -std::abs(a);
        c = -c;
      }
      float want = float(a * b + c);
      if (dx9 && (a == 0 || b == 0))
        want = modifiers && c == 0 ? -0.0f : float(c);
      if (modifiers)
        want *= 0.5f;
      expected[128 * (lane / 16) + lane % 16] = bits(want);
    }
    if (dx9) {
      // Zero times a positive factor selects C's negative zero unchanged.
      data[0][0] = 0;
      data[4][0] = 0x40000000;
      data[8][0] = 0x80000000;
      expected[0] = modifiers ? 0 : 0x80000000;
    }
  }

  bool correct() const {
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 16; ++col)
        if (row % 8 < output_regs) {
          int lane = col + 16 * (row / 8);
          if (data[16 + row % 8][lane] != expected[row * 16 + col])
            return false;
        }
    return true;
  }
};

bool nonnegative_integer(const char *text, int &value) {
  const char *end = text + std::strlen(text);
  auto parsed = std::from_chars(text, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end && value >= 0;
}

// Returns median nanoseconds per wave, or a negative value on an API/result error
// or iteration-count overflow. Every accepted sample spans at least min_ms.
template <typename Instruction>
double measure(Instruction fn, uint64_t flags, Registers &r, int initial_iterations, int min_ms,
               uint32_t modifiers) {
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
void print_columns(const char *input, const char *semantics, const char *instruction_flags,
                   const char *path, const char *time, const char *speedup) {
  std::printf("%-10s %-10s %-18s %-20s %12s %11s\n", input, semantics, instruction_flags, path,
              time, speedup);
}

// Print a result with fixed decimal precision; a negative speedup prints "--".
void print_result(const char *input, const char *semantics, const char *instruction_flags,
                  const char *path, double time, double speedup) {
  char time_text[64], speedup_text[64];
  std::snprintf(time_text, sizeof(time_text), "%.1f", time);
  if (speedup < 0)
    std::snprintf(speedup_text, sizeof(speedup_text), "--");
  else
    std::snprintf(speedup_text, sizeof(speedup_text), "%.2fx", speedup);
  print_columns(input, semantics, instruction_flags, path, time_text, speedup_text);
}

bool benchmark(bool bf16, uint64_t cpu, int iterations, int min_ms, uint32_t modifiers) {
  Registers r(bf16, modifiers);
  const char *instruction_flags = modifiers == 0                   ? "none"
                                  : modifiers == GOC_WMMA_NEG_LO_A ? "NEG_LO_A"
                                                                   : "mixed";
  const char *format = bf16 ? "bf16" : "fp16";
  Wmma fn = bf16 ? goc_rdna4_v_wmma_f32_16x16x16_bf16 : goc_rdna4_v_wmma_f32_16x16x16_f16;
  double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, modifiers);
  if (scalar < 0)
    return false;
  print_result(format, "loose", instruction_flags, "scalar", scalar, 1.0);

  const auto accelerated = [&](const char *path, uint64_t level) {
    double time = measure(fn, level, r, iterations, min_ms, modifiers);
    if (time < 0)
      return false;
    print_result(format, "loose", instruction_flags, path, time, scalar / time);
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
    if (!accelerated("Zen4 / avx512bf16", GOC_CPU_ZEN4))
      return false;
    ran_simd = true;
  }
#endif
  (void)cpu;
  (void)accelerated;
  if (!ran_simd)
    std::printf("%-6s SIMD unavailable in this build or on this host; skipped.\n", format);

  if (modifiers)
    return true;

  double exact =
      measure(fn, GOC_CPU_BASELINE | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, r,
              iterations, min_ms, 0);
  if (exact < 0)
    return false;
  print_result(format, "exact", "none", "scalar", exact, -1);
  return true;
}

bool benchmark_integer(int shape, int mode, uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_wmma_i32_16x16x16_iu8, goc_rdna4_v_wmma_i32_16x16x16_iu4,
                            goc_rdna4_v_wmma_i32_16x16x32_iu4};
  const char *names[] = {"int8/k16", "int4/k16", "int4/k32"};
  const char *instruction_flags = mode == 0 ? "u/u wrap" : "s/s clamp";
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
    print_result(names[shape], "exact", instruction_flags, path, time, scalar / time);
    return true;
  };
  if (!run("scalar", GOC_CPU_BASELINE))
    return false;
#if defined(GOC_BENCH_HAVE_X86_64_V3)
  if (cpu >= GOC_CPU_X86_64_V3 && !run("x86-64-v3", GOC_CPU_X86_64_V3))
    return false;
#endif
#if defined(GOC_BENCH_HAVE_AVX512VNNI)
  if (cpu >= GOC_CPU_ZEN4 && !run("Zen4 / avx512vnni", GOC_CPU_ZEN4))
    return false;
#endif
  (void)cpu;
  return true;
}

bool benchmark_fma(uint64_t cpu, int iterations, int min_ms, uint32_t modifiers, bool dx9) {
  Registers r;
  r.initialize_fma(modifiers, dx9);
  double scalar = 0;
  const auto run = [&](const char *path, uint64_t level) {
    auto fn = dx9 ? goc_rdna4_v_fma_dx9_zero_f32 : goc_rdna4_v_fma_f32;
    double time = measure(fn, level, r, iterations, min_ms, modifiers);
    if (time < 0)
      return false;
    if (level == GOC_CPU_BASELINE)
      scalar = time;
    print_result(dx9 ? "f32/fmadx9" : "f32/fma", "loose", modifiers ? "NEG/ABS/half" : "none", path,
                 time, scalar / time);
    return true;
  };
  if (!run("scalar", GOC_CPU_BASELINE))
    return false;
#if defined(GOC_BENCH_HAVE_X86_64_V3)
  if (cpu >= GOC_CPU_X86_64_V3 && !run("x86-64-v3", GOC_CPU_X86_64_V3))
    return false;
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
  if (cpu >= GOC_CPU_X86_64_V4 && !run("x86-64-v4", GOC_CPU_X86_64_V4))
    return false;
#endif
  (void)cpu;
  return true;
}

bool benchmark_unary(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_log_f32);
  const Unary functions[] = {goc_rdna4_v_trunc_f32, goc_rdna4_v_ceil_f32, goc_rdna4_v_rndne_f32,
                             goc_rdna4_v_floor_f32, goc_rdna4_v_sqrt_f32, goc_rdna4_v_rcp_f32,
                             goc_rdna4_v_rsq_f32,   goc_rdna4_v_exp_f32,  goc_rdna4_v_log_f32,
                             goc_rdna4_v_fract_f32};
  const char *names[] = {"f32/trunc", "f32/ceil", "f32/rndne", "f32/floor", "f32/sqrt",
                         "f32/rcp",   "f32/rsq",  "f32/exp",   "f32/log",   "f32/fract"};
  const float inputs[] = {0.25f, 1, 4, 16};
  const float exp_inputs[] = {0, 1, 2, 4};
  const float golden[][4] = {
      {0, 1, 4, 16},          {1, 1, 4, 16},       {0, 1, 4, 16}, {0, 1, 4, 16}, {0.5f, 1, 2, 4},
      {4, 1, 0.25f, 0.0625f}, {2, 1, 0.5f, 0.25f}, {1, 2, 4, 16}, {-2, 0, 2, 4}, {0.25f, 0, 0, 0}};
  for (int op = 0; op < 10; ++op)
    for (uint32_t modifiers : {UINT32_C(0), GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        float input = (op == 7 ? exp_inputs : inputs)[lane % 4];
        float want = golden[op][lane % 4];
        if (modifiers) {
          input = -input;
          want = std::min(1.0f, std::max(0.0f, want * 0.5f));
        }
        r.data[0][lane] = bits(input);
        r.expected[128 * (lane / 16) + lane % 16] = bits(want);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *,
                          const uint32_t *const *) {
        return functions[op](flags, mask, mode, d, a);
      };
      const char *mode = modifiers ? "ABS_A / half/clamp" : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, modifiers);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", mode, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if ((op < 7 || op == 9) && cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, modifiers);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", mode, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_binary(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_add_f32);
  const Binary functions[] = {
      goc_rdna4_v_add_f32,     goc_rdna4_v_sub_f32,     goc_rdna4_v_subrev_f32,
      goc_rdna4_v_mul_f32,     goc_rdna4_v_min_num_f32, goc_rdna4_v_max_num_f32,
      goc_rdna4_v_minimum_f32, goc_rdna4_v_maximum_f32, goc_rdna4_v_mul_dx9_zero_f32};
  const char *names[] = {"f32/add",    "f32/sub", "f32/subrev", "f32/mul",   "f32/minnum",
                         "f32/maxnum", "f32/min", "f32/max",    "f32/muldx9"};
  for (int op = 0; op < 9; ++op)
    for (uint32_t mode :
         {UINT32_C(0), GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        int a = lane - 16, b = lane % 7 - 3;
        r.data[0][lane] = bits(float(a));
        r.data[4][lane] = bits(float(b));
        if (mode) {
          a = std::abs(a);
          b = -b;
        }
        float want = float(op == 0 ? a + b : op == 1 ? a - b : op == 2 ? b - a : a * b);
        if (op >= 4 && op < 8)
          want = float((op == 5 || op == 7) ? std::max(a, b) : std::min(a, b));
        if (mode)
          want = std::min(1.0f, std::max(0.0f, want * 0.5f));
        // Multiplication retains the sign of zero.
        if (!mode && op == 3 && want == 0 && ((a < 0) != (b < 0)))
          want = -0.0f;
        r.expected[128 * (lane / 16) + lane % 16] = bits(want);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        return functions[op](flags, mask, modifiers, d, a, b);
      };
      const char *label = mode ? "ABS/NEG/half/clamp" : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_minmax3(uint64_t cpu, int iterations, int min_ms) {
  using Ternary = decltype(&goc_rdna4_v_fma_f32);
  const Ternary functions[] = {
      goc_rdna4_v_min3_num_f32,       goc_rdna4_v_max3_num_f32,       goc_rdna4_v_minmax_num_f32,
      goc_rdna4_v_maxmin_num_f32,     goc_rdna4_v_minimum3_f32,       goc_rdna4_v_maximum3_f32,
      goc_rdna4_v_minimummaximum_f32, goc_rdna4_v_maximumminimum_f32, goc_rdna4_v_med3_num_f32};
  const char *names[] = {"f32/min3n", "f32/max3n", "f32/mnmxn", "f32/mxmnn", "f32/min3",
                         "f32/max3",  "f32/mnmx",  "f32/mxmn",  "f32/med3n"};
  for (int op = 0; op < 9; ++op)
    for (uint32_t mode : {UINT32_C(0), GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_NEG_C |
                                           GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        int a = lane - 16, b = lane % 7 - 3, c = lane % 11 - 5;
        r.data[0][lane] = bits(float(a));
        r.data[4][lane] = bits(float(b));
        r.data[8][lane] = bits(float(c));
        if (mode) {
          a = std::abs(a);
          b = -b;
          c = -c;
        }
        int ab = (op % 4 == 1 || op % 4 == 3) ? std::max(a, b) : std::min(a, b);
        float want = float((op % 4 == 1 || op % 4 == 2) ? std::max(ab, c) : std::min(ab, c));
        if (op == 8) {
          int sorted[] = {a, b, c};
          std::sort(sorted, sorted + 3);
          want = float(sorted[1]);
        }
        if (mode)
          want = std::min(1.0f, std::max(0.0f, want * 0.5f));
        r.expected[128 * (lane / 16) + lane % 16] = bits(want);
      }
      auto fn = functions[op];
      const char *label = mode ? "ABS/NEG/half/clamp" : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_fp64(uint64_t cpu, int iterations, int min_ms) {
  const char *names[] = {"f64/add",    "f64/mul", "f64/fma", "f64/minnum",
                         "f64/maxnum", "f64/min", "f64/max"};
  for (int op = 0; op < 7; ++op)
    for (uint32_t mode :
         {UINT32_C(0), GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = 2;
      for (int lane = 0; lane < 32; ++lane) {
        int x[] = {lane - 16, lane % 7 - 3, lane % 11 - 5};
        for (int operand = 0; operand < 3; ++operand) {
          double value = x[operand];
          uint64_t raw;
          std::memcpy(&raw, &value, sizeof(raw));
          r.data[4 * operand][lane] = uint32_t(raw);
          r.data[4 * operand + 1][lane] = uint32_t(raw >> 32);
        }
        if (mode) {
          x[0] = std::abs(x[0]);
          x[1] = -x[1];
        }
        double want = op == 0 ? x[0] + x[1] : op == 1 ? x[0] * x[1] : x[0] * x[1] + x[2];
        if (op >= 3)
          want = (op == 4 || op == 6) ? std::max(x[0], x[1]) : std::min(x[0], x[1]);
        if (mode)
          want = std::min(1.0, std::max(0.0, want * 0.5));
        if (!mode && op == 1 && want == 0 && ((x[0] < 0) != (x[1] < 0)))
          want = -0.0;
        uint64_t raw;
        std::memcpy(&raw, &want, sizeof(raw));
        r.expected[128 * (lane / 16) + lane % 16] = uint32_t(raw);
        r.expected[128 * (lane / 16) + 16 + lane % 16] = uint32_t(raw >> 32);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
        if (op == 0)
          return goc_rdna4_v_add_f64(flags, mask, modifiers, d, a, b);
        if (op == 1)
          return goc_rdna4_v_mul_f64(flags, mask, modifiers, d, a, b);
        if (op == 3)
          return goc_rdna4_v_min_num_f64(flags, mask, modifiers, d, a, b);
        if (op == 4)
          return goc_rdna4_v_max_num_f64(flags, mask, modifiers, d, a, b);
        if (op == 5)
          return goc_rdna4_v_minimum_f64(flags, mask, modifiers, d, a, b);
        if (op == 6)
          return goc_rdna4_v_maximum_f64(flags, mask, modifiers, d, a, b);
        return goc_rdna4_v_fma_f64(flags, mask, modifiers, d, a, b, c);
      };
      const char *label = mode ? "ABS/NEG/half/clamp" : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_fp64_unary(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_sqrt_f64);
  const Unary functions[] = {goc_rdna4_v_trunc_f64, goc_rdna4_v_ceil_f64,  goc_rdna4_v_rndne_f64,
                             goc_rdna4_v_floor_f64, goc_rdna4_v_fract_f64, goc_rdna4_v_sqrt_f64,
                             goc_rdna4_v_rcp_f64,   goc_rdna4_v_rsq_f64};
  const char *names[] = {"f64/trunc", "f64/ceil", "f64/rndne", "f64/floor",
                         "f64/fract", "f64/sqrt", "f64/rcp",   "f64/rsq"};
  const double inputs[] = {0.25, 1, 4, 16};
  const double golden[][4] = {{0, 1, 4, 16},        {1, 1, 4, 16},    {0, 1, 4, 16},
                              {0, 1, 4, 16},        {0.25, 0, 0, 0},  {0.5, 1, 2, 4},
                              {4, 1, 0.25, 0.0625}, {2, 1, 0.5, 0.25}};
  for (int op = 0; op < 8; ++op)
    for (uint32_t mode : {UINT32_C(0), GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = 2;
      for (int lane = 0; lane < 32; ++lane) {
        double input = inputs[lane % 4], want = golden[op][lane % 4];
        if (mode) {
          input = -input;
          want = std::min(1.0, want * 0.5);
        }
        uint64_t raw;
        std::memcpy(&raw, &input, sizeof(raw));
        r.data[0][lane] = uint32_t(raw);
        r.data[1][lane] = uint32_t(raw >> 32);
        std::memcpy(&raw, &want, sizeof(raw));
        r.expected[128 * (lane / 16) + lane % 16] = uint32_t(raw);
        r.expected[128 * (lane / 16) + 16 + lane % 16] = uint32_t(raw >> 32);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *,
                          const uint32_t *const *) {
        return functions[op](flags, mask, modifiers, d, a);
      };
      const char *label = mode ? "ABS_A / half/clamp" : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_half_dot(uint64_t cpu, int iterations, int min_ms) {
  for (bool bf16 : {false, true})
    for (bool modified : {false, true}) {
      Registers r;
      r.output_regs = 1;
      const uint32_t mode =
          modified ? GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D : 0;
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = bf16 ? 0x40003f80 : 0x40003c00; // 1,2
        if (modified)
          r.data[0][lane] ^= 0x80008000;
        r.data[4][lane] = bf16 ? 0x40804040 : 0x44004200; // 3,4
        r.data[8][lane] = bf16 ? 0xc0003f80 : 0xc0003c00; // 1,-2
        r.data[16][lane] = 0xfacecafe;
        // Default: 1*3+2*4+1=12; modified: -1*3-2*4-2=-13.
        uint32_t code = bf16 ? (modified ? 0xc150 : 0x4140) : (modified ? 0xca80 : 0x4a00);
        r.expected[128 * (lane / 16) + lane % 16] =
            modified ? (code << 16) | 0xcafe : 0xface0000 | code;
      }
      Wmma fn = bf16 ? goc_rdna4_v_dot2_bf16_bf16 : goc_rdna4_v_dot2_f16_f16;
      const char *name = bf16 ? "dot2/b16" : "dot2/f16";
      const char *label = modified ? "ABS/NEG/hiC/hiD" : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(name, "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(name, "loose", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_fp8_wmma(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {
      goc_rdna4_v_wmma_f32_16x16x16_fp8_fp8, goc_rdna4_v_wmma_f32_16x16x16_fp8_bf8,
      goc_rdna4_v_wmma_f32_16x16x16_bf8_fp8, goc_rdna4_v_wmma_f32_16x16x16_bf8_bf8};
  const char *names[] = {"wmma/f8f8", "wmma/f8b8", "wmma/b8f8", "wmma/b8b8"};
  for (int format = 0; format < 4; ++format)
    for (uint32_t modifiers : {UINT32_C(0), GOC_WMMA_NEG_C | GOC_WMMA_ABS_C}) {
      Registers r;
      r.initialize_fp8_wmma(format, modifiers);
      const char *mode = modifiers ? "ABS_C/NEG_C" : "none";
      double scalar =
          measure(functions[format], GOC_CPU_BASELINE, r, iterations, min_ms, modifiers);
      if (scalar < 0)
        return false;
      print_result(names[format], "loose", mode, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd =
            measure(functions[format], GOC_CPU_X86_64_V3, r, iterations, min_ms, modifiers);
        if (simd < 0)
          return false;
        print_result(names[format], "loose", mode, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_fp8_dot(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_dot4_f32_fp8_fp8, goc_rdna4_v_dot4_f32_fp8_bf8,
                            goc_rdna4_v_dot4_f32_bf8_fp8, goc_rdna4_v_dot4_f32_bf8_bf8};
  const char *names[] = {"dot4/f8f8", "dot4/f8b8", "dot4/b8f8", "dot4/b8b8"};
  const int golden[] = {-6, 2, -3, 7};
  for (int op = 0; op < 4; ++op)
    for (uint32_t modifiers : {UINT32_C(0), GOC_DOT_NEG_C | GOC_DOT_ABS_C}) {
      Registers r;
      r.output_regs = 1;
      // A rotates [1,-2,2,-1]; B is [2,1,-2,2].
      uint32_t left = op >= 2 ? 0xbc40c03c : 0xb840c038;
      uint32_t right = op & 1 ? 0x40c03c40 : 0x40c03840;
      for (int lane = 0; lane < 32; ++lane) {
        unsigned shift = (lane % 4) * 8;
        r.data[0][lane] = shift ? (left >> shift) | (left << (32 - shift)) : left;
        r.data[4][lane] = right;
        int c = lane - 16;
        r.data[8][lane] = bits(float(c));
        if (modifiers)
          c = -std::abs(c);
        r.expected[128 * (lane / 16) + lane % 16] = bits(float(golden[lane % 4] + c));
      }
      const char *mode = modifiers ? "ABS_C/NEG_C" : "none";
      double scalar = measure(functions[op], GOC_CPU_BASELINE, r, iterations, min_ms, modifiers);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", mode, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(functions[op], GOC_CPU_X86_64_V3, r, iterations, min_ms, modifiers);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", mode, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_dot2(uint64_t cpu, int iterations, int min_ms) {
  for (bool bf16 : {false, true})
    for (uint32_t modifiers :
         {UINT32_C(0), GOC_DOT_NEG_LO_A, GOC_DOT_LO_A_HIGH | GOC_DOT_NEG_HI_B}) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = bf16 ? 0x40003f80 : 0x40003c00; // 1,2
        r.data[4][lane] = bf16 ? 0x40804040 : 0x44004200; // 3,4
        int c = lane - 16;
        r.data[8][lane] = bits(float(c));
        int dot = modifiers == 0 ? 11 : modifiers == GOC_DOT_NEG_LO_A ? 5 : -2;
        r.expected[128 * (lane / 16) + lane % 16] = bits(float(dot + c));
      }
      Wmma fn = bf16 ? goc_rdna4_v_dot2_f32_bf16 : goc_rdna4_v_dot2_f32_f16;
      const char *name = bf16 ? "dot2/bf16" : "dot2/fp16";
      const char *mode = modifiers == 0                  ? "none"
                         : modifiers == GOC_DOT_NEG_LO_A ? "NEG_LO_A"
                                                         : "select/NEG_HI_B";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, modifiers);
      if (scalar < 0)
        return false;
      print_result(name, "loose", mode, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, modifiers);
        if (simd < 0)
          return false;
        print_result(name, "loose", mode, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_integer_dot(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_dot4_i32_iu8, goc_rdna4_v_dot4_u32_u8,
                            goc_rdna4_v_dot8_i32_iu4, goc_rdna4_v_dot8_u32_u4};
  const char *names[] = {"dot4/i8", "dot4/u8", "dot8/i4", "dot8/u4"};
  const uint32_t a[] = {0xfedcba98, 0x80808080, 0x76543210, 0xffffffff};
  const uint32_t b[] = {0x76543210, 0x7f7f7f7f, 0xfedcba98, 0xffffffff};
  const uint32_t c[] = {0xfffffff0, 0x80000010, 0x7ffffff0, 0x00000000};
  // Independently calculated integer results, indexed by instruction and CLAMP.
  const uint32_t golden[][4] = {
      {0xffffdf08, 0x7fff0210, 0x7fffdf08, 0x00000004},
      {0xffffdf08, 0x80000000, 0x7fffdf08, 0x00000004},
      {0x0000eb08, 0x8000fe10, 0x8000eb08, 0x0003f804},
      {0xffffffff, 0x8000fe10, 0x8000eb08, 0x0003f804},
      {0xffffff9c, 0x7fffff30, 0x7fffff9c, 0x00000008},
      {0xffffff9c, 0x80000000, 0x7fffff9c, 0x00000008},
      {0x0000015c, 0x800000f0, 0x8000015c, 0x00000708},
      {0xffffffff, 0x800000f0, 0x8000015c, 0x00000708},
  };
  for (int op = 0; op < 4; ++op)
    for (int clamp = 0; clamp < 2; ++clamp) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = a[lane % 4];
        r.data[4][lane] = b[lane % 4];
        r.data[8][lane] = c[lane % 4];
        r.expected[128 * (lane / 16) + lane % 16] = golden[2 * op + clamp][lane % 4];
      }
      uint32_t modifiers =
          (op & 1 ? 0 : GOC_DOT_SIGNED_A | GOC_DOT_SIGNED_B) | (clamp ? GOC_DOT_CLAMP : 0);
      const char *mode =
          op & 1 ? (clamp ? "u/u clamp" : "u/u wrap") : (clamp ? "s/s clamp" : "s/s wrap");
      double scalar = measure(functions[op], GOC_CPU_BASELINE, r, iterations, min_ms, modifiers);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", mode, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(functions[op], GOC_CPU_X86_64_V3, r, iterations, min_ms, modifiers);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", mode, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
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
  if (argc == 2 && (!nonnegative_integer(argv[1], iterations) || iterations == 0)) {
    std::fprintf(stderr, "Initial iterations must be a positive integer.\n");
    return 2;
  }
  if (const char *value = std::getenv("GOC_BENCH_MIN_MS")) {
    if (!nonnegative_integer(value, min_ms)) {
      std::fprintf(stderr, "GOC_BENCH_MIN_MS must be a nonnegative integer in milliseconds.\n");
      return 2;
    }
  }

  uint64_t cpu = goc_init_cpu_flags();
  std::printf("RDNA4 wave32 WMMA (16x16 output), DOT, FMA and unary arithmetic; CPU flags 0x%llx\n",
              static_cast<unsigned long long>(cpu));
  std::printf("Median of 7 samples, each at least %d ms, after warmup.\n", min_ms);
  std::printf("Start at %d calls; double until the minimum duration is reached.\n", iterations);
  std::puts("Fixed inputs, full EXEC, separate C/D, hot buffers; all outputs checked.");
  std::puts("Timings include public API dispatch, input conversions and output stores.");
  std::puts("FP rows: loose speedups, exact scalar separately. Integer WMMA rows: exact. Speedups "
            "compare "
            "matching instruction-flags settings.");
  std::puts("mixed = NEG_HI_A | NEG_LO_B | ABS_C | NEG_C.");
  std::puts("DOT2 output widths: f16,b16 = 16-bit; fp16,bf16 = FP32.");
  print_columns("Input", "Semantics", "Instruction flags", "CPU path", "ns/wave", "Speedup");
  if (!benchmark_minmax3(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "Three-input min/max benchmark failed.\n");
    return 1;
  }
  if (!benchmark_fp64(cpu, iterations, min_ms) || !benchmark_fp64_unary(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "FP64 benchmark failed.\n");
    return 1;
  }
  if (!benchmark_binary(cpu, iterations, min_ms)) {
    std::fprintf(stderr,
                 "Binary arithmetic benchmark failed: API/result error or iteration overflow.\n");
    return 1;
  }
  if (!benchmark_half_dot(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "True16 DOT2 benchmark failed: API/result error or iteration overflow.\n");
    return 1;
  }
  if (!benchmark_fp8_wmma(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "FP8 WMMA benchmark failed: API/result error or iteration overflow.\n");
    return 1;
  }
  if (!benchmark_fp8_dot(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "FP8 DOT4 benchmark failed: API/result error or iteration overflow.\n");
    return 1;
  }
  if (!benchmark_dot2(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "DOT2 benchmark failed: API/result error or iteration overflow.\n");
    return 1;
  }
  if (!benchmark_integer_dot(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "Integer DOT benchmark failed: API/result error or iteration overflow.\n");
    return 1;
  }
  if (!benchmark_unary(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "Unary benchmark failed: API/result error or iteration overflow.\n");
    return 1;
  }
  for (uint32_t modifiers :
       {UINT32_C(0), GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_NEG_C | GOC_ALU_OMOD_HALF})
    if (!benchmark_fma(cpu, iterations, min_ms, modifiers, false) ||
        !benchmark_fma(cpu, iterations, min_ms, modifiers, true)) {
      std::fprintf(stderr, "FMA benchmark failed: API/result error or iteration overflow.\n");
      return 1;
    }
  for (uint32_t modifiers :
       {UINT32_C(0), GOC_WMMA_NEG_LO_A,
        GOC_WMMA_NEG_HI_A | GOC_WMMA_NEG_LO_B | GOC_WMMA_ABS_C | GOC_WMMA_NEG_C})
    if (!benchmark(false, cpu, iterations, min_ms, modifiers) ||
        !benchmark(true, cpu, iterations, min_ms, modifiers)) {
      std::fprintf(stderr, "Benchmark failed: API/result error or iteration overflow.\n");
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

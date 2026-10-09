// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_bit_count_reference.h"
#include "rdna4_bitfield_reference.h"
#include "rdna4_boolean_reference.h"
#include "rdna4_dense_golden.h"
#include "rdna4_half_reference.h"
#include "rdna4_integer16_reference.h"
#include "rdna4_integer16_ternary_reference.h"
#include "rdna4_integer_mad_reference.h"
#include "rdna4_integer_ternary_reference.h"
#include "rdna4_mixed_fma_reference.h"
#include "rdna4_packed_integer_reference.h"
#include "rdna4_packed_mad_reference.h"
#include "rdna4_sad_reference.h"
#include "rdna4_shift_reference.h"
#include "rdna4_subbyte_golden.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
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
  // Nonzero only for approximate FP32 math benchmarks. Other rows remain bit-exact.
  float absolute_tolerance = 0;

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
          uint32_t actual_bits = data[16 + row % 8][lane];
          uint32_t expected_bits = expected[row * 16 + col];
          if (actual_bits != expected_bits) {
            if (absolute_tolerance == 0)
              return false;
            float actual, want;
            std::memcpy(&actual, &actual_bits, sizeof(actual));
            std::memcpy(&want, &expected_bits, sizeof(want));
            if (!std::isfinite(actual) || !std::isfinite(want) ||
                std::abs(actual - want) > absolute_tolerance)
              return false;
          }
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
template <typename Instruction, typename RegisterFile>
double measure(Instruction fn, uint64_t flags, RegisterFile &r, int initial_iterations, int min_ms,
               uint32_t modifiers, uint64_t mask = UINT32_MAX) {
  uint64_t iterations = uint64_t(initial_iterations);
  const auto call = [&] { return fn(flags, mask, modifiers, r.v + 16, r.v, r.v + 4, r.v + 8); };
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

bool benchmark_integer_mul(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_mul_lo_u32);
  const Binary functions[] = {goc_rdna4_v_mul_lo_u32,     goc_rdna4_v_mul_hi_u32,
                              goc_rdna4_v_mul_hi_i32,     goc_rdna4_v_mul_i32_i24,
                              goc_rdna4_v_mul_hi_i32_i24, goc_rdna4_v_mul_u32_u24,
                              goc_rdna4_v_mul_hi_u32_u24};
  const char *names[] = {"u32/mullo", "u32/mulhi", "i32/mulhi", "i24/mul",
                         "i24/mulhi", "u24/mul",   "u24/mulhi"};
  const uint32_t inputs[][4] = {{0x007fffff, 0xff800000, 0xffffffff, 65535},
                                {256, 0x00800000, 0xffffffff, 65536}};
  for (int op = 0; op < 7; ++op)
    for (int clamp = 0; clamp <= int(op == 3 || op == 5); ++clamp) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        uint64_t a = inputs[0][lane % 4], b = inputs[1][lane % 4];
        r.data[0][lane] = uint32_t(a);
        r.data[4][lane] = uint32_t(b);
        bool signed_op = op == 2 || op == 3 || op == 4;
        uint64_t modulus = UINT64_C(1) << (op < 3 ? 32 : 24);
        a %= modulus;
        b %= modulus;
        bool na = signed_op && a >= modulus / 2, nb = signed_op && b >= modulus / 2;
        uint64_t product = (na ? modulus - a : a) * (nb ? modulus - b : b);
        if (clamp)
          product =
              std::min(product, signed_op ? (na != nb ? UINT64_C(0x80000000) : UINT64_C(0x7fffffff))
                                          : uint64_t(UINT32_MAX));
        if (na != nb)
          product = UINT64_C(0) - product;
        bool high = op == 1 || op == 2 || op == 4 || op == 6;
        r.expected[128 * (lane / 16) + lane % 16] = uint32_t(high ? product >> 32 : product);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        return functions[op](flags, mask, mode, d, a, b);
      };
      const uint32_t mode = clamp ? GOC_ALU_CLAMP : 0;
      const char *label = clamp ? "clamp" : "none";
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
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

template <auto Function>
int integer_binary(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
  return Function(flags, mask, mode, d, a, b);
}

bool benchmark_half_binary(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {
      integer_binary<goc_rdna4_v_add_f16>,     integer_binary<goc_rdna4_v_sub_f16>,
      integer_binary<goc_rdna4_v_subrev_f16>,  integer_binary<goc_rdna4_v_mul_f16>,
      integer_binary<goc_rdna4_v_min_num_f16>, integer_binary<goc_rdna4_v_max_num_f16>,
      integer_binary<goc_rdna4_v_minimum_f16>, integer_binary<goc_rdna4_v_maximum_f16>};
  const char *names[] = {"f16/add",    "f16/sub",    "f16/subrev", "f16/mul",
                         "f16/minnum", "f16/maxnum", "f16/minim",  "f16/maxim"};
  const uint32_t modified = GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D | GOC_ALU_ABS_A |
                            GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  for (int op = 0; op < 8; ++op)
    for (uint32_t mode : {UINT32_C(0), modified}) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        double a = (lane % 11 - 5) * 0.25, b = (lane % 7 - 3) * 0.25;
        r.data[0][lane] = goc_test::half_bits(a) | (uint32_t(goc_test::half_bits(-a)) << 16);
        r.data[4][lane] = goc_test::half_bits(b) | (uint32_t(goc_test::half_bits(-b)) << 16);
        r.data[16][lane] = 0xfacecafe;
        if (mode) {
          a = std::abs(-a);
          // HIGH_B selects -b; NEG_B restores b.
        }
        double want = goc_test::half_binary(op, a, b);
        if (mode)
          want = std::clamp(want * 0.5, 0.0, 1.0);
        // CLAMP produces positive zero.
        if (mode && want == 0)
          want = 0.0;
        uint32_t half = goc_test::half_bits(want);
        r.expected[128 * (lane / 16) + lane % 16] =
            mode ? (half << 16) | 0xcafe : 0xface0000 | half;
      }
      const char *label = mode ? "ABS/NEG/hi/half/cl" : "none";
      double scalar = measure(functions[op], GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(functions[op], GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_integer_add(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {
      integer_binary<goc_rdna4_v_add_nc_u32>,    integer_binary<goc_rdna4_v_sub_nc_u32>,
      integer_binary<goc_rdna4_v_subrev_nc_u32>, integer_binary<goc_rdna4_v_add_nc_i32>,
      integer_binary<goc_rdna4_v_sub_nc_i32>,    goc_rdna4_v_add3_u32};
  const char *names[] = {"u32/add", "u32/sub", "u32/subrev", "i32/add", "i32/sub", "u32/add3"};
  const uint32_t inputs[][4] = {
      {0xffffffff, 0x7fffffff, 0x80000000, 0}, {1, 1, 0xffffffff, 1}, {0xffffffff, 2, 3, 4}};
  for (int op = 0; op < 6; ++op)
    for (int clamp = 0; clamp <= int(op != 5); ++clamp) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        int64_t a = inputs[0][lane % 4], b = inputs[1][lane % 4], c = inputs[2][lane % 4];
        r.data[0][lane] = uint32_t(a);
        r.data[4][lane] = uint32_t(b);
        r.data[8][lane] = uint32_t(c);
        bool signed_op = op == 3 || op == 4;
        if (signed_op) {
          if (a > INT32_MAX)
            a -= INT64_C(4294967296);
          if (b > INT32_MAX)
            b -= INT64_C(4294967296);
        }
        int64_t value = op == 1 || op == 4 ? a - b : op == 2 ? b - a : a + b;
        if (op == 5)
          value += c;
        if (clamp)
          value = std::clamp(value, signed_op ? int64_t(INT32_MIN) : INT64_C(0),
                             signed_op ? int64_t(INT32_MAX) : int64_t(UINT32_MAX));
        r.expected[128 * (lane / 16) + lane % 16] = uint32_t(value);
      }
      const uint32_t mode = clamp ? GOC_ALU_CLAMP : 0;
      const char *label = clamp ? "clamp" : "none";
      double scalar = measure(functions[op], GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (clamp && cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(functions[op], GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(functions[op], GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_integer_minmax(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {integer_binary<goc_rdna4_v_min_i32>,
                            integer_binary<goc_rdna4_v_max_i32>,
                            goc_rdna4_v_min3_i32,
                            goc_rdna4_v_max3_i32,
                            goc_rdna4_v_minmax_i32,
                            goc_rdna4_v_maxmin_i32,
                            goc_rdna4_v_med3_i32,
                            integer_binary<goc_rdna4_v_min_u32>,
                            integer_binary<goc_rdna4_v_max_u32>,
                            goc_rdna4_v_min3_u32,
                            goc_rdna4_v_max3_u32,
                            goc_rdna4_v_minmax_u32,
                            goc_rdna4_v_maxmin_u32,
                            goc_rdna4_v_med3_u32};
  const char *names[] = {"i32/min",    "i32/max",    "i32/min3",   "i32/max3", "i32/minmax",
                         "i32/maxmin", "i32/med3",   "u32/min",    "u32/max",  "u32/min3",
                         "u32/max3",   "u32/minmax", "u32/maxmin", "u32/med3"};
  const uint32_t input[3][4] = {{1, 0xffffffff, 0x80000000, 0x7fffffff},
                                {2, 1, 0x7fffffff, 0x80000000},
                                {3, 0x80000000, 0, 0xffffffff}};
  for (int op = 0; op < 14; ++op) {
    Registers r;
    r.output_regs = 1;
    for (int lane = 0; lane < 32; ++lane) {
      int64_t values[3];
      for (int reg = 0; reg < 3; ++reg) {
        uint32_t word = input[reg][lane % 4];
        r.data[4 * reg][lane] = word;
        values[reg] = word;
        if (op < 7 && word > INT32_MAX)
          values[reg] -= INT64_C(4294967296);
      }
      std::sort(values, values + 2);
      int operation = op % 7;
      int64_t want;
      if (operation < 2) {
        want = values[operation];
      } else if (operation == 4) {
        want = std::max(values[0], values[2]);
      } else if (operation == 5) {
        want = std::min(values[1], values[2]);
      } else {
        std::sort(values, values + 3);
        want = values[operation == 2 ? 0 : operation == 3 ? 2 : 1];
      }
      r.expected[128 * (lane / 16) + lane % 16] = uint32_t(want);
    }
    double scalar = measure(functions[op], GOC_CPU_BASELINE, r, iterations, min_ms, 0);
    if (scalar < 0)
      return false;
    print_result(names[op], "loose", "none", "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3) {
      double simd = measure(functions[op], GOC_CPU_X86_64_V3, r, iterations, min_ms, 0);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", "none", "x86-64-v3", simd, scalar / simd);
    }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
    if (cpu >= GOC_CPU_X86_64_V4) {
      double simd = measure(functions[op], GOC_CPU_X86_64_V4, r, iterations, min_ms, 0);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", "none", "x86-64-v4", simd, scalar / simd);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_ldexp(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_ldexp_f32);
  const Binary functions[] = {goc_rdna4_v_ldexp_f32, goc_rdna4_v_ldexp_f64};
  const double inputs[] = {0.75, -0.5, 1.5, -2};
  const int powers[] = {-2, -1, 1, 2};
  const double golden[] = {0.1875, -0.25, 3, -8};
  for (int fp64 = 0; fp64 < 2; ++fp64)
    for (uint32_t mode : {UINT32_C(0), GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = fp64 ? 2 : 1;
      for (int lane = 0; lane < 32; ++lane) {
        double input = inputs[lane % 4], want = golden[lane % 4];
        if (mode)
          want = std::min(1.0, std::abs(want) * 0.5);
        uint64_t input_bits, output_bits;
        std::memcpy(&input_bits, &input, sizeof(input));
        std::memcpy(&output_bits, &want, sizeof(want));
        if (!fp64) {
          input_bits = bits(float(input));
          output_bits = bits(float(want));
        }
        r.data[0][lane] = uint32_t(input_bits);
        r.data[1][lane] = uint32_t(input_bits >> 32);
        r.data[4][lane] = uint32_t(powers[lane % 4]);
        int index = 128 * (lane / 16) + lane % 16;
        r.expected[index] = uint32_t(output_bits);
        if (fp64)
          r.expected[index + 16] = uint32_t(output_bits >> 32);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        return functions[fp64](flags, mask, modifiers, d, a, b);
      };
      const char *name = fp64 ? "f64/ldexp" : "f32/ldexp";
      const char *label = mode ? "ABS_A / half/clamp" : "none";
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
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(name, "loose", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_frexp_exp(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_frexp_exp_i32_f32);
  const Unary functions[] = {goc_rdna4_v_frexp_exp_i32_f32, goc_rdna4_v_frexp_exp_i32_f64};
  const uint64_t inputs[][4] = {{0x3e800000, 0xbf800000, 0x40800000, 0x41800000},
                                {UINT64_C(0x3fd0000000000000), UINT64_C(0xbff0000000000000),
                                 UINT64_C(0x4010000000000000), UINT64_C(0x4030000000000000)}};
  const int exponents[] = {-1, 1, 3, 5};
  for (int fp64 = 0; fp64 < 2; ++fp64)
    for (uint32_t mode :
         {UINT32_C(0), GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = uint32_t(inputs[fp64][lane % 4]);
        r.data[1][lane] = uint32_t(inputs[fp64][lane % 4] >> 32);
        r.expected[128 * (lane / 16) + lane % 16] = uint32_t(exponents[lane % 4]);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *,
                          const uint32_t *const *) {
        return functions[fp64](flags, mask, modifiers, d, a);
      };
      const char *name = fp64 ? "f64/frexp" : "f32/frexp";
      const char *label = mode ? "NEG/ABS/half/clamp" : "none";
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

bool benchmark_half_exponent(uint64_t cpu, int iterations, int min_ms) {
  const uint16_t inputs[] = {0x3400, 0xbc00, 0x4400, 0x4c00, 1, 0x3ff, 0x7bff, 0};
  const int adjustments[] = {-2, -1, 0, 1, 24, 0, -1, 32767};
  const int exponents[] = {-1, 1, 3, 5, -23, -14, 16, 0};
  for (bool ldexp : {false, true})
    for (bool modified : {false, true}) {
      uint32_t mode = modified
                          ? GOC_ALU_ABS_A | GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | GOC_ALU_OMOD_HALF |
                                GOC_ALU_CLAMP | (ldexp ? GOC_ALU_HIGH_B : 0)
                          : 0;
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        auto half = inputs[lane % 8];
        int exponent = adjustments[lane % 8];
        r.data[0][lane] = uint32_t(half) | (uint32_t(half ^ 0x8000) << 16);
        r.data[4][lane] = uint16_t(exponent) | (uint32_t(uint16_t(exponent)) << 16);
        r.data[16][lane] = 0xfacecafe;
        uint32_t result = uint16_t(exponents[lane % 8]);
        if (ldexp) {
          double x = goc_test::half_value(half);
          if (modified)
            x = std::abs(x);
          double value = std::ldexp(x, exponent);
          if (modified)
            value = !(value > 0) ? 0 : std::min(value * 0.5, 1.0);
          result = goc_test::half_bits(value);
        }
        r.expected[128 * (lane / 16) + lane % 16] =
            modified ? (result << 16) | 0xcafe : 0xface0000 | result;
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        return ldexp ? goc_rdna4_v_ldexp_f16(flags, mask, modifiers, d, a, b)
                     : goc_rdna4_v_frexp_exp_i16_f16(flags, mask, modifiers, d, a);
      };
      const char *name = ldexp ? "f16/ldexp" : "f16/frexp";
      const char *label = modified ? "ABS/hi/half/clamp" : "none";
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

bool benchmark_half_unary(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_log_f16);
  const Unary functions[] = {
      goc_rdna4_v_trunc_f16, goc_rdna4_v_ceil_f16,      goc_rdna4_v_rndne_f16,
      goc_rdna4_v_floor_f16, goc_rdna4_v_sqrt_f16,      goc_rdna4_v_rcp_f16,
      goc_rdna4_v_rsq_f16,   goc_rdna4_v_exp_f16,       goc_rdna4_v_log_f16,
      goc_rdna4_v_fract_f16, goc_rdna4_v_frexp_mant_f16};
  const char *names[] = {"f16/trunc", "f16/ceil", "f16/rndne", "f16/floor", "f16/sqrt", "f16/rcp",
                         "f16/rsq",   "f16/exp",  "f16/log",   "f16/fract", "f16/mant"};
  const float inputs[] = {0.25f, 1, 4, 16};
  const float exp_inputs[] = {0, 1, 2, 4};
  const float golden[][4] = {{0, 1, 4, 16},       {1, 1, 4, 16},       {0, 1, 4, 16},
                             {0, 1, 4, 16},       {0.5f, 1, 2, 4},     {4, 1, 0.25f, 0.0625f},
                             {2, 1, 0.5f, 0.25f}, {1, 2, 4, 16},       {-2, 0, 2, 4},
                             {0.25f, 0, 0, 0},    {0.5, 0.5, 0.5, 0.5}};
  for (int op = 0; op < 11; ++op)
    for (uint32_t modifiers : {UINT32_C(0), GOC_ALU_ABS_A | GOC_ALU_HIGH_A | GOC_ALU_HIGH_D |
                                                GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        float input = (op == 7 ? exp_inputs : inputs)[lane % 4];
        float want = golden[op][lane % 4];
        if (modifiers) {
          input = -input;
          want = std::min(1.0f, std::max(0.0f, want * 0.5f));
        }
        uint32_t code = goc_test::half_bits(input), result = goc_test::half_bits(want);
        r.data[0][lane] = modifiers ? (code << 16) | 0xbeef : 0xdead0000 | code;
        r.data[16][lane] = 0xfacecafe;
        r.expected[128 * (lane / 16) + lane % 16] =
            modifiers ? (result << 16) | 0xcafe : 0xface0000 | result;
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *,
                          const uint32_t *const *) {
        return functions[op](flags, mask, mode, d, a);
      };
      const char *mode = modifiers ? "ABS/hi/half/clamp" : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, modifiers);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", mode, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if ((op < 7 || op >= 9) && cpu >= GOC_CPU_X86_64_V3) {
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

bool benchmark_mixed_fma(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_fma_mix_f32, goc_rdna4_v_fma_mixlo_f16,
                            goc_rdna4_v_fma_mixhi_f16};
  const char *names[] = {"f32/mix", "f16/mixlo", "f16/mixhi"};
  const uint32_t modes[] = {0,
                            GOC_MIX_F16_A | GOC_MIX_F16_C | GOC_ALU_HIGH_A | GOC_ALU_NEG_A |
                                GOC_ALU_ABS_B | GOC_ALU_CLAMP,
                            GOC_MIX_F16_A | GOC_MIX_F16_B | GOC_MIX_F16_C | GOC_ALU_HIGH_B};
  const char *labels[] = {"none", "fp16 A/C+mods", "fp16 all"};
  for (int op = 0; op < 3; ++op)
    for (int variant = 0; variant < 3; ++variant) {
      uint32_t mode = modes[variant];
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        for (int source = 0; source < 3; ++source) {
          uint32_t raw = bits(float(lane - source * 8) * 0.0625f);
          if (mode & (GOC_MIX_F16_A << source)) {
            uint16_t lo = goc_test::half_bits(double(lane - source * 8) * 0.0625);
            uint16_t hi = goc_test::half_bits(double(source * 3 - lane) * 0.125);
            raw = lo | (uint32_t(hi) << 16);
          }
          r.data[source * 4][lane] = raw;
        }
        r.data[16][lane] = 0xfacecafe;
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::mixed_fma_reference::evaluate(
            op, r.data[0][lane], r.data[4][lane], r.data[8][lane], r.data[16][lane], mode, true);
      }
      double scalar =
          measure(functions[op], GOC_CPU_BASELINE | GOC_FP16_OVFL, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", labels[variant], "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd =
            measure(functions[op], GOC_CPU_X86_64_V3 | GOC_FP16_OVFL, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", labels[variant], "x86-64-v3", simd, scalar / simd);
      }
#endif
      if (op) {
        double exact = measure(functions[op],
                               GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT | GOC_FP16_OVFL,
                               r, iterations, min_ms, mode);
        if (exact < 0)
          return false;
        print_result(names[op], "exact", labels[variant], "scalar", exact, 1);
      }
    }
  (void)cpu;
  return true;
}

bool benchmark_bit_count(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_bcnt_u32_b32);
  const Binary functions[] = {
      goc_test::count_leading,         goc_test::count_trailing,       goc_test::count_sign,
      goc_rdna4_v_bcnt_u32_b32,        goc_rdna4_v_mbcnt_lo_u32_b32,   goc_rdna4_v_mbcnt_hi_u32_b32,
      goc_rdna4w64_v_mbcnt_lo_u32_b32, goc_rdna4w64_v_mbcnt_hi_u32_b32};
  const char *names[] = {"u32/clz",   "b32/ctz",   "i32/cls",    "u32/bcnt",
                         "u32/mbclo", "u32/mbchi", "u32/mbcl64", "u32/mbch64"};

  struct CountRegisters {
    uint32_t data[3][64] = {}, expected[64] = {};
    uint32_t *v[24] = {};

    CountRegisters() {
      v[0] = data[0];
      v[4] = data[1];
      v[16] = data[2];
    }

    bool correct() const { return std::equal(data[2], data[2] + 64, expected); }
  };

  for (int op = 0; op < 8; ++op) {
    CountRegisters r;
    std::mt19937 random(935);
    unsigned lanes = op < 6 ? 32 : 64;
    for (unsigned lane = 0; lane < 64; ++lane) {
      r.data[0][lane] = random();
      r.data[1][lane] = UINT32_MAX - lane;
      r.data[2][lane] = 0xfacecafe;
      r.expected[lane] =
          lane < lanes ? goc_test::bit_count_reference(op, r.data[0][lane], r.data[1][lane], lane)
                       : 0xfacecafe;
    }
    uint64_t mask = op < 6 ? UINT32_MAX : UINT64_MAX;
    const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *) {
      return functions[op](flags, mask, modifiers, d, a, b);
    };
    const char *label = "none";
    double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, 0, mask);
    if (scalar < 0)
      return false;
    print_result(names[op], "loose", label, "scalar", scalar, 1);

#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3 && op != 5) {
      double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, 0, mask);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", label, "x86-64-v3", simd, scalar / simd);
    }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
    if (cpu >= GOC_CPU_X86_64_V4) {
      double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, 0, mask);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", label, "x86-64-v4", simd, scalar / simd);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_boolean(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_and_b32);
  const Binary functions[] = {goc_rdna4_v_and_b32,     goc_rdna4_v_or_b32,     goc_rdna4_v_xor_b32,
                              goc_test::boolean_not32, goc_rdna4_v_and_b16,    goc_rdna4_v_or_b16,
                              goc_rdna4_v_xor_b16,     goc_test::boolean_not16};
  const char *names[] = {"b32/and", "b32/or", "b32/xor", "b32/not",
                         "b16/and", "b16/or", "b16/xor", "b16/not"};
  for (int op = 0; op < 8; ++op)
    for (bool modified : {false, true}) {
      if (op < 4 && modified)
        continue;
      uint32_t mode = goc_test::boolean_mode(op, modified ? 7 : 0);
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0x83171521u * (lane + 1);
        r.data[4][lane] = 0xb43d7357u * (lane + 3);
        r.data[16][lane] = 0xfacecafe;
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::boolean_reference(
            op, r.data[0][lane], r.data[4][lane], r.data[16][lane], mode);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        return functions[op](flags, mask, modifiers, d, a, b);
      };
      const char *label = modified ? "high" : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", label, "scalar", scalar, 1);

#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_bitfield(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_bfe_u32, goc_rdna4_v_bfe_i32, goc_rdna4_v_bfi_b32,
                            goc_test::bitfield_mask, goc_test::bitfield_reverse};
  const char *names[] = {"u32/bfe", "i32/bfe", "b32/bfi", "b32/bfm", "b32/bfrev"};
  for (int op = 0; op < 5; ++op) {
    Registers r;
    r.output_regs = 1;
    std::mt19937 random(452);
    for (int lane = 0; lane < 32; ++lane) {
      for (int reg : {0, 4, 8})
        r.data[reg][lane] = random();
      r.expected[128 * (lane / 16) + lane % 16] =
          goc_test::bitfield_reference(op, r.data[0][lane], r.data[4][lane], r.data[8][lane]);
    }
    double scalar = measure(functions[op], GOC_CPU_BASELINE, r, iterations, min_ms, 0);
    if (scalar < 0)
      return false;
    print_result(names[op], "loose", "none", "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3 && op != 2) {
      double simd = measure(functions[op], GOC_CPU_X86_64_V3, r, iterations, min_ms, 0);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", "none", "x86-64-v3", simd, scalar / simd);
    }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
    if (cpu >= GOC_CPU_X86_64_V4) {
      double simd = measure(functions[op], GOC_CPU_X86_64_V4, r, iterations, min_ms, 0);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", "none", "x86-64-v4", simd, scalar / simd);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_integer_ternary(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_lshl_add_u32, goc_rdna4_v_add_lshl_u32,
                            goc_rdna4_v_lshl_or_b32,  goc_rdna4_v_and_or_b32,
                            goc_rdna4_v_or3_b32,      goc_rdna4_v_xor3_b32,
                            goc_rdna4_v_xad_u32,      goc_rdna4_v_lerp_u8};
  const char *names[] = {"u32/shladd", "u32/addshl", "b32/shlor", "b32/andor",
                         "b32/or3",    "b32/xor3",   "u32/xad",   "u8/lerp"};
  for (int op = 0; op < 8; ++op) {
    Registers r;
    r.output_regs = 1;
    std::mt19937 random(452);
    for (int lane = 0; lane < 32; ++lane) {
      for (int reg : {0, 4, 8})
        r.data[reg][lane] = random();
      r.expected[128 * (lane / 16) + lane % 16] = goc_test::integer_ternary_reference(
          op, r.data[0][lane], r.data[4][lane], r.data[8][lane]);
    }
    double scalar = measure(functions[op], GOC_CPU_BASELINE, r, iterations, min_ms, 0);
    if (scalar < 0)
      return false;
    print_result(names[op], "loose", "none", "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3 && (op < 3 || op == 7)) {
      double simd = measure(functions[op], GOC_CPU_X86_64_V3, r, iterations, min_ms, 0);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", "none", "x86-64-v3", simd, scalar / simd);
    }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
    if (cpu >= GOC_CPU_X86_64_V4) {
      double simd = measure(functions[op], GOC_CPU_X86_64_V4, r, iterations, min_ms, 0);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", "none", "x86-64-v4", simd, scalar / simd);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_sad(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_sad_u8,          goc_rdna4_v_sad_hi_u8,
                            goc_rdna4_v_sad_u16,         goc_rdna4_v_sad_u32,
                            goc_rdna4_v_msad_u8,         goc_rdna4_v_qsad_pk_u16_u8,
                            goc_rdna4_v_mqsad_pk_u16_u8, goc_rdna4_v_mqsad_u32_u8};
  const char *names[] = {"u8/sad",  "u8/sadhi", "u16/sad",   "u32/sad",
                         "u8/msad", "u16/qsad", "u16/mqsad", "u32/mqsad"};
  for (int op = 0; op < 8; ++op)
    for (uint32_t mode : {UINT32_C(0), GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = op == 7 ? 4 : op >= 5 ? 2 : 1;
      std::mt19937 random(830);
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = random();
        r.data[1][lane] = random();
        r.data[4][lane] = random() & (lane % 2 ? UINT32_MAX : 0x00ff00ff);
        uint32_t c[4];
        for (int reg = 0; reg < 4; ++reg)
          r.data[8 + reg][lane] = c[reg] = lane % 2 ? UINT32_MAX - random() % 2048 : random();
        auto expected = goc_test::sad_reference(op, r.data[0][lane], r.data[1][lane],
                                                r.data[4][lane], c, bool(mode));
        for (int reg = 0; reg < r.output_regs; ++reg)
          r.expected[128 * (lane / 16) + 16 * reg + lane % 16] = expected[reg];
      }
      double scalar = measure(functions[op], GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", mode ? "clamp" : "none", "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(functions[op], GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", mode ? "clamp" : "none", "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_shift(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_lshlrev_b32);
  const Binary functions[] = {goc_rdna4_v_lshlrev_b32, goc_rdna4_v_lshrrev_b32,
                              goc_rdna4_v_ashrrev_i32, goc_rdna4_v_lshlrev_b64,
                              goc_rdna4_v_lshrrev_b64, goc_rdna4_v_ashrrev_i64};
  const char *names[] = {"b32/shl", "b32/shr", "i32/ashr", "b64/shl", "b64/shr", "i64/ashr"};
  for (int op = 0; op < 6; ++op) {
    Registers r;
    int width = op < 3 ? 32 : 64;
    r.output_regs = width / 32;
    std::mt19937 random(861);
    for (int lane = 0; lane < 32; ++lane) {
      r.data[0][lane] = (random() & ~UINT32_C(63)) | uint32_t((lane * 7) % 64);
      r.data[4][lane] = random();
      r.data[5][lane] = random();
      uint64_t value = r.data[4][lane] | (uint64_t(r.data[5][lane]) << 32);
      uint64_t expected = goc_test::shift_reference(width, op % 3, r.data[0][lane], value);
      int index = 128 * (lane / 16) + lane % 16;
      r.expected[index] = uint32_t(expected);
      if (width == 64)
        r.expected[index + 16] = uint32_t(expected >> 32);
    }
    const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *) {
      return functions[op](flags, mask, mode, d, a, b);
    };
    double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, 0);
    if (scalar < 0)
      return false;
    print_result(names[op], "loose", "none", "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3) {
      double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, 0);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", "none", "x86-64-v3", simd, scalar / simd);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_half_trig(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_sin_f16);
  const Unary functions[] = {goc_rdna4_v_sin_f16, goc_rdna4_v_cos_f16};
  const char *names[] = {"f16/sin", "f16/cos"};
  for (int op = 0; op < 2; ++op)
    for (uint32_t modifiers : {UINT32_C(0), GOC_ALU_NEG_A | GOC_ALU_HIGH_A | GOC_ALU_HIGH_D |
                                                GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        double input = double(lane * 37 - 600) / 512;
        uint16_t code = goc_test::half_bits(input);
        r.data[0][lane] = modifiers ? (uint32_t(code) << 16) | 0x7c01 : 0x7c010000 | code;
        r.data[16][lane] = 0xfacecafe;
        double phase = std::remainder(modifiers ? -input : input, 1.0);
        double angle = phase * 6.283185307179586476925286766559;
        uint16_t want = goc_test::half_bits(op ? std::cos(angle) : std::sin(angle));
        if (modifiers) {
          double rounded = goc_test::half_value(want);
          rounded = std::abs(rounded) < 0x1p-14 ? 0 : rounded * 0.5;
          want = goc_test::half_bits(rounded);
          if ((want & 0x7fff) < 0x400)
            want = 0;
          want = goc_test::half_bits(std::min(1.0, std::max(0.0, goc_test::half_value(want))));
        }
        r.expected[128 * (lane / 16) + lane % 16] =
            modifiers ? (uint32_t(want) << 16) | 0xcafe : 0xface0000 | want;
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *,
                          const uint32_t *const *) {
        return functions[op](flags, mask, mode, d, a);
      };
      const char *mode = modifiers ? "NEG/hi/half/clamp" : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, modifiers);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", mode, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
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

bool benchmark_trig(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_sin_f32);
  const Unary functions[] = {goc_rdna4_v_sin_f32, goc_rdna4_v_cos_f32};
  const char *names[] = {"f32/sin", "f32/cos"};
  for (int op = 0; op < 2; ++op)
    for (uint32_t modifiers : {UINT32_C(0), GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
      Registers r;
      r.output_regs = 1;
      r.absolute_tolerance = 3e-7f;
      for (int lane = 0; lane < 32; ++lane) {
        float input = float(lane * 193 - 3021) / 1024;
        r.data[0][lane] = bits(input);
        double phase = std::remainder(double(modifiers ? -input : input), 1.0);
        double angle = phase * 6.283185307179586476925286766559;
        double want = op ? std::cos(angle) : std::sin(angle);
        if (modifiers)
          want = std::min(1.0, std::max(0.0, want * 0.5));
        r.expected[128 * (lane / 16) + lane % 16] = bits(float(want));
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *,
                          const uint32_t *const *) {
        return functions[op](flags, mask, mode, d, a);
      };
      const char *mode = modifiers ? "NEG_A / half/clamp" : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, modifiers);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", mode, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
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

bool benchmark_unary(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_log_f32);
  const Unary functions[] = {
      goc_rdna4_v_trunc_f32, goc_rdna4_v_ceil_f32,      goc_rdna4_v_rndne_f32,
      goc_rdna4_v_floor_f32, goc_rdna4_v_sqrt_f32,      goc_rdna4_v_rcp_f32,
      goc_rdna4_v_rsq_f32,   goc_rdna4_v_exp_f32,       goc_rdna4_v_log_f32,
      goc_rdna4_v_fract_f32, goc_rdna4_v_frexp_mant_f32};
  const char *names[] = {"f32/trunc", "f32/ceil", "f32/rndne", "f32/floor", "f32/sqrt", "f32/rcp",
                         "f32/rsq",   "f32/exp",  "f32/log",   "f32/fract", "f32/mant"};
  const float inputs[] = {0.25f, 1, 4, 16};
  const float exp_inputs[] = {0, 1, 2, 4};
  const float golden[][4] = {{0, 1, 4, 16},       {1, 1, 4, 16},       {0, 1, 4, 16},
                             {0, 1, 4, 16},       {0.5f, 1, 2, 4},     {4, 1, 0.25f, 0.0625f},
                             {2, 1, 0.5f, 0.25f}, {1, 2, 4, 16},       {-2, 0, 2, 4},
                             {0.25f, 0, 0, 0},    {0.5, 0.5, 0.5, 0.5}};
  for (int op = 0; op < 11; ++op)
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
      if ((op < 7 || op >= 9) && cpu >= GOC_CPU_X86_64_V3) {
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

bool benchmark_integer_mad(uint64_t cpu, int iterations, int min_ms) {
  using Fn = decltype(&goc_rdna4_v_mad_i32_i16);
  const Fn functions[] = {goc_rdna4_v_mad_u32_u16, goc_rdna4_v_mad_i32_i16, goc_rdna4_v_mad_u32_u24,
                          goc_rdna4_v_mad_i32_i24};
  const char *names[] = {"u16/mad32", "i16/mad32", "u24/mad32", "i24/mad32"};
  for (int op = 0; op < 4; ++op)
    for (bool modified : {false, true}) {
      uint32_t mode = modified ? GOC_ALU_CLAMP | (op < 2 ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B : 0) : 0;
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0x81171521u * (lane + 1);
        r.data[4][lane] = 0xb43d7357u * (lane + 3);
        r.data[8][lane] = 0x6b271231u * (lane + 7);
        r.data[16][lane] = 0xfacecafe;
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::integer_mad_reference(
            op, r.data[0][lane], r.data[4][lane], r.data[8][lane], mode);
      }
      const char *name = names[op];
      const char *label = modified ? (op < 2 ? "high/clamp" : "clamp") : "none";
      double scalar = measure(functions[op], GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(name, "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(functions[op], GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(name, "loose", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_integer16_ternary(uint64_t cpu, int iterations, int min_ms) {
  using Fn = decltype(&goc_rdna4_v_mad_i16);
  const Fn functions[] = {goc_rdna4_v_mad_u16,  goc_rdna4_v_mad_i16,  goc_rdna4_v_min3_u16,
                          goc_rdna4_v_min3_i16, goc_rdna4_v_max3_u16, goc_rdna4_v_max3_i16,
                          goc_rdna4_v_med3_u16, goc_rdna4_v_med3_i16};
  const char *names[] = {"u16/mad",  "i16/mad",  "u16/min3", "i16/min3",
                         "u16/max3", "i16/max3", "u16/med3", "i16/med3"};
  for (int op = 0; op < 8; ++op)
    for (bool modified : {false, true}) {
      uint32_t mode =
          modified ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D | (op < 2 ? GOC_ALU_CLAMP : 0)
                   : 0;
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0x81171521u * (lane + 1);
        r.data[4][lane] = 0xb43d7357u * (lane + 3);
        r.data[8][lane] = 0x6b271231u * (lane + 7);
        r.data[16][lane] = 0xfacecafe;
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::integer16_ternary_reference(
            op, r.data[0][lane], r.data[4][lane], r.data[8][lane], mode);
      }
      const char *name = names[op];
      const char *label = modified ? (op < 2 ? "high/clamp" : "high") : "none";
      double scalar = measure(functions[op], GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(name, "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(functions[op], GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(name, "loose", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_packed_mad(uint64_t cpu, int iterations, int min_ms) {
  using Fn = decltype(&goc_rdna4_v_pk_mad_i16);
  const Fn functions[] = {goc_rdna4_v_pk_mad_u16, goc_rdna4_v_pk_mad_i16};
  for (int sign = 0; sign < 2; ++sign)
    for (bool modified : {false, true}) {
      uint32_t mode = modified ? GOC_PK_LO_A_HIGH | GOC_PK_HI_A_LOW | GOC_PK_LO_B_HIGH |
                                     GOC_PK_HI_C_LOW | GOC_PK_CLAMP
                               : 0;
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0x81171521u * (lane + 1);
        r.data[4][lane] = 0xb43d7357u * (lane + 3);
        r.data[8][lane] = 0x6b271231u * (lane + 7);
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::packed_mad_reference(
            sign, r.data[0][lane], r.data[4][lane], r.data[8][lane], mode);
      }
      const char *name = sign ? "i16/pmad" : "u16/pmad";
      const char *label = modified ? "select/clamp" : "none";
      double scalar = measure(functions[sign], GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(name, "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(functions[sign], GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(name, "loose", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_integer16(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_add_nc_i16);
  const Binary functions[] = {
      goc_rdna4_v_add_nc_i16,  goc_rdna4_v_sub_nc_i16,  goc_rdna4_v_add_nc_u16,
      goc_rdna4_v_sub_nc_u16,  goc_rdna4_v_min_i16,     goc_rdna4_v_max_i16,
      goc_rdna4_v_min_u16,     goc_rdna4_v_max_u16,     goc_rdna4_v_mul_lo_u16,
      goc_rdna4_v_lshlrev_b16, goc_rdna4_v_lshrrev_b16, goc_rdna4_v_ashrrev_i16};
  const char *names[] = {"i16/add", "i16/sub", "u16/add", "u16/sub", "i16/min", "i16/max",
                         "u16/min", "u16/max", "u16/mul", "u16/shl", "u16/shr", "i16/shr"};
  for (int op = 0; op < 12; ++op)
    for (bool modified : {false, true}) {
      uint32_t mode =
          modified ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D | (op < 4 ? GOC_ALU_CLAMP : 0)
                   : 0;
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0x83171521u * (lane + 1);
        r.data[4][lane] = 0xb43d7357u * (lane + 3);
        r.data[16][lane] = 0xfacecafe;
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::integer16_reference(
            op, r.data[0][lane], r.data[4][lane], r.data[16][lane], mode);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        return functions[op](flags, mask, modifiers, d, a, b);
      };
      const char *label = modified ? (op < 4 ? "high/clamp" : "high") : "none";
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

bool benchmark_packed_integer(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_pk_add_i16);
  const Binary functions[] = {
      goc_rdna4_v_pk_add_i16,     goc_rdna4_v_pk_sub_i16,     goc_rdna4_v_pk_add_u16,
      goc_rdna4_v_pk_sub_u16,     goc_rdna4_v_pk_min_i16,     goc_rdna4_v_pk_max_i16,
      goc_rdna4_v_pk_min_u16,     goc_rdna4_v_pk_max_u16,     goc_rdna4_v_pk_mul_lo_u16,
      goc_rdna4_v_pk_lshlrev_b16, goc_rdna4_v_pk_lshrrev_b16, goc_rdna4_v_pk_ashrrev_i16};
  const char *names[] = {"i16/padd", "i16/psub", "u16/padd", "u16/psub", "i16/pmin", "i16/pmax",
                         "u16/pmin", "u16/pmax", "u16/pmul", "u16/pshl", "u16/pshr", "i16/pshr"};
  for (int op = 0; op < 12; ++op)
    for (bool modified : {false, true}) {
      uint32_t mode =
          modified ? GOC_PK_LO_A_HIGH | GOC_PK_HI_A_LOW | GOC_PK_LO_B_HIGH | GOC_PK_CLAMP : 0;
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0x83171521u * (lane + 1);
        r.data[4][lane] = 0xb43d7357u * (lane + 3);
        r.expected[128 * (lane / 16) + lane % 16] =
            goc_test::packed_integer_reference(op, r.data[0][lane], r.data[4][lane], mode);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        return functions[op](flags, mask, modifiers, d, a, b);
      };
      const char *label = modified ? "select/clamp" : "none";
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

bool benchmark_packed_binary(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_pk_add_f16);
  const Binary functions[] = {goc_rdna4_v_pk_add_f16,     goc_rdna4_v_pk_mul_f16,
                              goc_rdna4_v_pk_min_num_f16, goc_rdna4_v_pk_max_num_f16,
                              goc_rdna4_v_pk_minimum_f16, goc_rdna4_v_pk_maximum_f16};
  const char *names[] = {"f16/padd", "f16/pmul", "f16/pminn", "f16/pmaxn", "f16/pmin", "f16/pmax"};
  for (int op = 0; op < 6; ++op)
    for (bool modified : {false, true}) {
      uint32_t mode = modified ? GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A | GOC_PK_LO_A_HIGH |
                                     GOC_PK_HI_A_LOW | GOC_PK_LO_B_HIGH | GOC_PK_CLAMP
                               : 0;
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        float al = float(lane - 16), ah = float(lane % 9 - 4), bl = float(lane % 7 - 3),
              bh = float(lane % 5 - 2);
        r.data[0][lane] = goc_test::half_bits(al) | (uint32_t(goc_test::half_bits(ah)) << 16);
        r.data[4][lane] = goc_test::half_bits(bl) | (uint32_t(goc_test::half_bits(bh)) << 16);
        uint32_t expected = 0;
        for (int half = 0; half < 2; ++half) {
          float x = modified ? (half ? -al : ah) : (half ? ah : al);
          float y = modified ? (half ? bh : -bh) : (half ? bh : bl);
          float value = op == 0    ? x + y
                        : op == 1  ? x * y
                        : (op & 1) ? std::max(x, y)
                                   : std::min(x, y);
          if (modified)
            value = std::min(1.0f, std::max(0.0f, value));
          expected |= uint32_t(goc_test::half_bits(value)) << (16 * half);
        }
        r.expected[128 * (lane / 16) + lane % 16] = expected;
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        return functions[op](flags, mask, modifiers, d, a, b);
      };
      const char *label = modified ? "NEG/select/clamp" : "none";
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

bool benchmark_packed_fma(uint64_t cpu, int iterations, int min_ms) {
  for (bool accumulate : {false, true})
    for (int modified = 0; modified < (accumulate ? 1 : 2); ++modified) {
      uint32_t mode = modified
                          ? GOC_PK_NEG_HI_A | GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_C | GOC_PK_LO_A_HIGH |
                                GOC_PK_HI_A_LOW | GOC_PK_LO_B_HIGH | GOC_PK_HI_C_LOW | GOC_PK_CLAMP
                          : 0;
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        int al = lane - 16, ah = lane % 9 - 4, bl = lane % 7 - 3, bh = lane % 5 - 2,
            cl = lane % 11 - 5, ch = lane % 13 - 6;
        r.data[0][lane] = goc_test::half_bits(al) | (uint32_t(goc_test::half_bits(ah)) << 16);
        r.data[4][lane] = goc_test::half_bits(bl) | (uint32_t(goc_test::half_bits(bh)) << 16);
        r.data[8][lane] = goc_test::half_bits(cl) | (uint32_t(goc_test::half_bits(ch)) << 16);
        int low = modified ? ah * -bh + cl : al * bl + cl;
        int high = modified ? -al * bh - cl : ah * bh + ch;
        if (modified) {
          low = std::min(1, std::max(0, low));
          high = std::min(1, std::max(0, high));
        }
        r.expected[128 * (lane / 16) + lane % 16] =
            goc_test::half_bits(low) | (uint32_t(goc_test::half_bits(high)) << 16);
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
        if (accumulate) {
          // Include the fixed-input accumulator reset in all FMAC timings.
          std::memcpy(d[0], c[0], 32 * sizeof(uint32_t));
          return goc_rdna4_v_pk_fmac_f16(flags, mask, modifiers, d, a, b);
        }
        return goc_rdna4_v_pk_fma_f16(flags, mask, modifiers, d, a, b, c);
      };
      const char *name = accumulate ? "f16/pkfmac" : "f16/pkfma";
      const char *label = modified ? "NEG/select/clamp" : "none";
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
      double exact =
          measure(fn, GOC_CPU_BASELINE | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, r,
                  iterations, min_ms, mode);
      if (exact < 0)
        return false;
      print_result(name, "exact", label, "scalar", exact, 1);
    }
  (void)cpu;
  return true;
}

bool benchmark_literal_fma(uint64_t cpu, int iterations, int min_ms) {
  for (bool half : {false, true})
    for (bool multiply : {false, true})
      for (int selection = 0; selection < (half ? 2 : 1); ++selection) {
        uint32_t mode = selection ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D : 0;
        uint32_t literal = half ? 0xb800 : 0xbf000000;
        Registers r;
        r.output_regs = 1;
        for (int lane = 0; lane < 32; ++lane) {
          int a = lane - 16, b = lane % 7 - 3;
          float want = multiply ? a * -0.5f + b : a * b - 0.5f;
          r.data[16][lane] = 0xfacecafe;
          if (half) {
            uint32_t ha = goc_test::half_bits(a), hb = goc_test::half_bits(b);
            r.data[0][lane] = selection ? (ha << 16) | 0x7c01 : 0x7c010000 | ha;
            r.data[4][lane] = selection ? (hb << 16) | 0xfc00 : 0xfc000000 | hb;
            uint32_t result = goc_test::half_bits(want);
            r.expected[128 * (lane / 16) + lane % 16] =
                selection ? (result << 16) | 0xcafe : 0xface0000 | result;
          } else {
            r.data[0][lane] = bits(float(a));
            r.data[4][lane] = bits(float(b));
            r.expected[128 * (lane / 16) + lane % 16] = bits(want);
          }
        }
        const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *) {
          if (half)
            return multiply
                       ? goc_rdna4_v_fmamk_f16(flags, mask, modifiers, d, a, uint16_t(literal), b)
                       : goc_rdna4_v_fmaak_f16(flags, mask, modifiers, d, a, b, uint16_t(literal));
          return multiply ? goc_rdna4_v_fmamk_f32(flags, mask, modifiers, d, a, literal, b)
                          : goc_rdna4_v_fmaak_f32(flags, mask, modifiers, d, a, b, literal);
        };
        const char *name =
            half ? (multiply ? "f16/fmamk" : "f16/fmaak") : (multiply ? "f32/fmamk" : "f32/fmaak");
        const char *label = selection ? "hi" : "none";
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
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (!half && cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(name, "loose", label, "x86-64-v4", simd, scalar / simd);
        }
#endif
        if (half) {
          double exact =
              measure(fn, GOC_CPU_BASELINE | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                      r, iterations, min_ms, mode);
          if (exact < 0)
            return false;
          print_result(name, "exact", label, "scalar", exact, 1);
        }
      }
  (void)cpu;
  return true;
}

bool benchmark_fmac(uint64_t cpu, int iterations, int min_ms) {
  for (bool half : {false, true})
    for (bool modified : {false, true}) {
      uint32_t mode =
          modified ? GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP : 0;
      if (half && modified)
        mode |= GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        int a = lane - 16, b = lane % 7 - 3, c = lane % 11 - 5;
        float want = modified ? float(std::abs(a) * -b + c) : float(a * b + c);
        if (modified)
          want = std::min(1.0f, std::max(0.0f, want * 0.5f));
        if (half) {
          uint32_t ha = goc_test::half_bits(a), hb = goc_test::half_bits(b),
                   hc = goc_test::half_bits(c);
          r.data[0][lane] = modified ? (ha << 16) | 0xbeef : 0xdead0000 | ha;
          r.data[4][lane] = modified ? (hb << 16) | 0x7c01 : 0xfc010000 | hb;
          r.data[8][lane] = modified ? (hc << 16) | 0xcafe : 0xface0000 | hc;
          uint32_t result = goc_test::half_bits(want);
          r.expected[128 * (lane / 16) + lane % 16] =
              modified ? (result << 16) | 0xcafe : 0xface0000 | result;
        } else {
          r.data[0][lane] = bits(float(a));
          r.data[4][lane] = bits(float(b));
          r.data[8][lane] = bits(float(c));
          r.expected[128 * (lane / 16) + lane % 16] = bits(want);
        }
      }
      auto operation = half ? goc_rdna4_v_fmac_f16 : goc_rdna4_v_fmac_f32;
      const auto fn = [&](uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
        // Keep the accumulator input fixed, avoiding drift during repeated FMAC.
        // This reset is included in timings for every CPU path.
        std::memcpy(d[0], c[0], 32 * sizeof(uint32_t));
        return operation(flags, mask, modifiers, d, a, b);
      };
      const char *name = half ? "f16/fmac" : "f32/fmac";
      const char *label = modified ? (half ? "ABS/NEG/hi/half/cl" : "ABS/NEG/half/cl") : "none";
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
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (!half && cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(name, "loose", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
      if (half) {
        double exact =
            measure(fn, GOC_CPU_BASELINE | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, r,
                    iterations, min_ms, mode);
        if (exact < 0)
          return false;
        print_result(name, "exact", label, "scalar", exact, 1);
      }
    }
  (void)cpu;
  return true;
}

bool benchmark_half_fma(uint64_t cpu, int iterations, int min_ms) {
  for (uint32_t mode : {UINT32_C(0), GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_NEG_C |
                                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
                                         GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D}) {
    Registers r;
    r.output_regs = 1;
    for (int lane = 0; lane < 32; ++lane) {
      int a = lane - 16, b = lane % 7 - 3, c = lane % 11 - 5;
      uint32_t ha = goc_test::half_bits(a), hb = goc_test::half_bits(b),
               hc = goc_test::half_bits(c);
      r.data[0][lane] = mode ? (ha << 16) | 0xbeef : 0xdead0000 | ha;
      r.data[4][lane] = mode ? (hb << 16) | 0x7c01 : 0xfc010000 | hb;
      r.data[8][lane] = mode ? (hc << 16) | 0x7c00 : 0xfc000000 | hc;
      r.data[16][lane] = 0xfacecafe;
      float want = mode ? float(std::abs(a) * -b - c) : float(a * b + c);
      if (mode)
        want = std::min(1.0f, std::max(0.0f, want * 0.5f));
      uint32_t result = goc_test::half_bits(want);
      r.expected[128 * (lane / 16) + lane % 16] =
          mode ? (result << 16) | 0xcafe : 0xface0000 | result;
    }
    auto fn = goc_rdna4_v_fma_f16;
    const char *label = mode ? "ABS/NEG/hi/half/cl" : "none";
    double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
    if (scalar < 0)
      return false;
    print_result("f16/fma", "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3) {
      double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("f16/fma", "loose", label, "x86-64-v3", simd, scalar / simd);
    }
#endif
    double exact =
        measure(fn, GOC_CPU_BASELINE | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, r,
                iterations, min_ms, mode);
    if (exact < 0)
      return false;
    print_result("f16/fma", "exact", label, "scalar", exact, 1);
  }
  (void)cpu;
  return true;
}

bool benchmark_half_minmax3(uint64_t cpu, int iterations, int min_ms) {
  using Ternary = decltype(&goc_rdna4_v_min3_num_f16);
  const Ternary functions[] = {
      goc_rdna4_v_min3_num_f16,       goc_rdna4_v_max3_num_f16,       goc_rdna4_v_minmax_num_f16,
      goc_rdna4_v_maxmin_num_f16,     goc_rdna4_v_minimum3_f16,       goc_rdna4_v_maximum3_f16,
      goc_rdna4_v_minimummaximum_f16, goc_rdna4_v_maximumminimum_f16, goc_rdna4_v_med3_num_f16};
  const char *names[] = {"f16/min3n", "f16/max3n", "f16/mnmxn", "f16/mxmnn", "f16/min3",
                         "f16/max3",  "f16/mnmx",  "f16/mxmn",  "f16/med3n"};
  for (int op = 0; op < 9; ++op)
    for (uint32_t mode : {UINT32_C(0), GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_NEG_C |
                                           GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
                                           GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D}) {
      Registers r;
      r.output_regs = 1;
      for (int lane = 0; lane < 32; ++lane) {
        int a = lane - 16, b = lane % 7 - 3, c = lane % 11 - 5;
        uint32_t ha = goc_test::half_bits(a), hb = goc_test::half_bits(b),
                 hc = goc_test::half_bits(c);
        r.data[0][lane] = mode ? (ha << 16) | 0xbeef : 0xdead0000 | ha;
        r.data[4][lane] = mode ? (hb << 16) | 0x7c01 : 0xfc010000 | hb;
        r.data[8][lane] = mode ? (hc << 16) | 0x7c00 : 0xfc000000 | hc;
        r.data[16][lane] = 0xfacecafe;
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
        uint32_t result = goc_test::half_bits(want);
        r.expected[128 * (lane / 16) + lane % 16] =
            mode ? (result << 16) | 0xcafe : 0xface0000 | result;
      }
      auto fn = functions[op];
      const char *label = mode ? "ABS/NEG/hi/half/cl" : "none";
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
  const Unary functions[] = {
      goc_rdna4_v_trunc_f64, goc_rdna4_v_ceil_f64,  goc_rdna4_v_rndne_f64,
      goc_rdna4_v_floor_f64, goc_rdna4_v_fract_f64, goc_rdna4_v_sqrt_f64,
      goc_rdna4_v_rcp_f64,   goc_rdna4_v_rsq_f64,   goc_rdna4_v_frexp_mant_f64};
  const char *names[] = {"f64/trunc", "f64/ceil", "f64/rndne", "f64/floor", "f64/fract",
                         "f64/sqrt",  "f64/rcp",  "f64/rsq",   "f64/mant"};
  const double inputs[] = {0.25, 1, 4, 16};
  const double golden[][4] = {{0, 1, 4, 16},        {1, 1, 4, 16},     {0, 1, 4, 16},
                              {0, 1, 4, 16},        {0.25, 0, 0, 0},   {0.5, 1, 2, 4},
                              {4, 1, 0.25, 0.0625}, {2, 1, 0.5, 0.25}, {0.5, 0.5, 0.5, 0.5}};
  for (int op = 0; op < 9; ++op)
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
  if (!benchmark_integer_mad(cpu, iterations, min_ms) ||
      !benchmark_integer16_ternary(cpu, iterations, min_ms) ||
      !benchmark_integer16(cpu, iterations, min_ms) ||
      !benchmark_packed_mad(cpu, iterations, min_ms) ||
      !benchmark_packed_integer(cpu, iterations, min_ms) ||
      !benchmark_packed_binary(cpu, iterations, min_ms) ||
      !benchmark_packed_fma(cpu, iterations, min_ms) ||
      !benchmark_literal_fma(cpu, iterations, min_ms) || !benchmark_fmac(cpu, iterations, min_ms) ||
      !benchmark_half_fma(cpu, iterations, min_ms) ||
      !benchmark_half_minmax3(cpu, iterations, min_ms) ||
      !benchmark_minmax3(cpu, iterations, min_ms)) {
    std::fprintf(stderr,
                 "Packed arithmetic, FMA family or three-input min/max benchmark failed.\n");
    return 1;
  }
  if (!benchmark_fp64(cpu, iterations, min_ms) || !benchmark_fp64_unary(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "FP64 benchmark failed.\n");
    return 1;
  }
  if (!benchmark_binary(cpu, iterations, min_ms) || !benchmark_ldexp(cpu, iterations, min_ms)) {
    std::fprintf(stderr,
                 "Binary arithmetic benchmark failed: API/result error or iteration overflow.\n");
    return 1;
  }
  if (!benchmark_half_binary(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "FP16 binary benchmark failed: API/result error or iteration overflow.\n");
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
  if (!benchmark_integer_add(cpu, iterations, min_ms) ||
      !benchmark_integer_mul(cpu, iterations, min_ms) ||
      !benchmark_integer_dot(cpu, iterations, min_ms) ||
      !benchmark_integer_minmax(cpu, iterations, min_ms)) {
    std::fprintf(stderr, "Integer DOT benchmark failed: API/result error or iteration overflow.\n");
    return 1;
  }
  if (!benchmark_half_exponent(cpu, iterations, min_ms) ||
      !benchmark_integer_ternary(cpu, iterations, min_ms) ||
      !benchmark_mixed_fma(cpu, iterations, min_ms) ||
      !benchmark_bit_count(cpu, iterations, min_ms) ||
      !benchmark_boolean(cpu, iterations, min_ms) || !benchmark_bitfield(cpu, iterations, min_ms) ||
      !benchmark_sad(cpu, iterations, min_ms) || !benchmark_shift(cpu, iterations, min_ms) ||
      !benchmark_half_trig(cpu, iterations, min_ms) || !benchmark_trig(cpu, iterations, min_ms) ||
      !benchmark_half_unary(cpu, iterations, min_ms) || !benchmark_unary(cpu, iterations, min_ms) ||
      !benchmark_frexp_exp(cpu, iterations, min_ms)) {
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

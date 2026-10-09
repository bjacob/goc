// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_bit_count_reference.h"
#include "rdna4_bitfield_reference.h"
#include "rdna4_boolean_reference.h"
#include "rdna4_byte_conversion_reference.h"
#include "rdna4_carry_reference.h"
#include "rdna4_class_reference.h"
#include "rdna4_cndmask_reference.h"
#include "rdna4_conversion16_reference.h"
#include "rdna4_conversion32_reference.h"
#include "rdna4_conversion64_reference.h"
#include "rdna4_cube_reference.h"
#include "rdna4_dense_golden.h"
#include "rdna4_div_fixup_reference.h"
#include "rdna4_dpp16_reference.h"
#include "rdna4_dpp_arithmetic_reference.h"
#include "rdna4_dpp_half_fma_reference.h"
#include "rdna4_dpp_integer_reference.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_float_compare_reference.h"
#include "rdna4_fp8_conversion_reference.h"
#include "rdna4_fp8_narrow_reference.h"
#include "rdna4_half_binary_reference.h"
#include "rdna4_half_minmax_reference.h"
#include "rdna4_half_reference.h"
#include "rdna4_half_unary_reference.h"
#include "rdna4_integer16_reference.h"
#include "rdna4_integer16_ternary_reference.h"
#include "rdna4_integer_add_reference.h"
#include "rdna4_integer_compare_reference.h"
#include "rdna4_integer_conversion_reference.h"
#include "rdna4_integer_mad_reference.h"
#include "rdna4_integer_minmax_reference.h"
#include "rdna4_integer_mul_reference.h"
#include "rdna4_integer_ternary_reference.h"
#include "rdna4_interp16_reference.h"
#include "rdna4_interp32_reference.h"
#include "rdna4_mad64_reference.h"
#include "rdna4_mixed_fma_reference.h"
#include "rdna4_mullit_reference.h"
#include "rdna4_normalized_reference.h"
#include "rdna4_pack_reference.h"
#include "rdna4_packed_conversion_reference.h"
#include "rdna4_packed_integer_reference.h"
#include "rdna4_packed_mad_reference.h"
#include "rdna4_permlane_reference.h"
#include "rdna4_pseudo_scalar_hardware.h"
#include "rdna4_pseudo_scalar_reference.h"
#include "rdna4_rcp_iflag_reference.h"
#include "rdna4_sad_reference.h"
#include "rdna4_scalar_bits_reference.h"
#include "rdna4_scalar_compare_reference.h"
#include "rdna4_scalar_convert_reference.h"
#include "rdna4_scalar_field_reference.h"
#include "rdna4_scalar_fma_reference.h"
#include "rdna4_scalar_fp_reference.h"
#include "rdna4_scalar_integer_reference.h"
#include "rdna4_scalar_pack_reference.h"
#include "rdna4_scalar_round_reference.h"
#include "rdna4_shift_reference.h"
#include "rdna4_subbyte_golden.h"
#include "rdna4_swmmac16_hardware.h"
#include "rdna4_swmmac8_hardware.h"
#include "rdna4_swmmac_integer_hardware.h"
#include "rdna4_trig_preop_hardware.h"
#include "rdna4_unary_reference.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
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
        want = float(c);
      if (modifiers)
        want *= 0.5f;
      expected[128 * (lane / 16) + lane % 16] = bits(want);
    }
    if (dx9) {
      // The positive zero product adds to negative zero, producing +0 in RNE.
      data[0][0] = 0;
      data[4][0] = 0x40000000;
      data[8][0] = 0x80000000;
      expected[0] = 0;
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
               uint64_t modifiers, uint64_t mask = UINT32_MAX) {
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

bool csv_output = false;

// Print one CSV field, quoting embedded commas, quotes, and newlines.
void print_csv_field(const char *text) {
  bool quoted = std::strpbrk(text, ",\"\r\n") != nullptr;
  if (quoted)
    std::putchar('"');
  for (; *text; ++text) {
    if (*text == '"')
      std::putchar('"');
    std::putchar(*text);
  }
  if (quoted)
    std::putchar('"');
}

// Print a header or result row in the selected output format.
void print_columns(const char *instruction, const char *semantics, const char *instruction_flags,
                   const char *path, const char *time, const char *speedup, const char *wave,
                   const char *fp_state = "none") {
  if (csv_output) {
    const char *fields[] = {instruction, wave, semantics, instruction_flags,
                            fp_state,    path, time,      speedup};
    for (unsigned i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
      if (i)
        std::putchar(',');
      print_csv_field(fields[i]);
    }
    std::putchar('\n');
    return;
  }
  // Longest current mnemonic: 27 characters, with three spare. Other widths
  // accommodate headers, flag combinations, CPU names, and timing headroom.
  std::printf("%-30s %4s %-9s %-20s %-14s %-18s %10s %9s\n", instruction, wave, semantics,
              instruction_flags, fp_state, path, time, speedup);
}

// Print fixed-precision timing and speedup; negative speedup is absent in CSV
// and displayed as "--" in the table.
void print_result(const char *instruction, const char *semantics, const char *instruction_flags,
                  const char *path, double time, double speedup, int wave = 32,
                  const char *fp_state = "none") {
  char time_text[64], speedup_text[64];
  std::snprintf(time_text, sizeof(time_text), "%.1f", time);
  if (speedup < 0)
    std::snprintf(speedup_text, sizeof(speedup_text), "%s", csv_output ? "" : "--");
  else
    std::snprintf(speedup_text, sizeof(speedup_text), csv_output ? "%.2f" : "%.2fx", speedup);
  print_columns(instruction, semantics, instruction_flags, path, time_text, speedup_text,
                wave == 64 ? "64" : "32", fp_state);
}

bool benchmark(bool bf16, uint64_t cpu, int iterations, int min_ms, uint32_t modifiers) {
  Registers r(bf16, modifiers);
  const char *instruction_flags = modifiers == 0                   ? "none"
                                  : modifiers == GOC_WMMA_NEG_LO_A ? "NEG_LO_A"
                                                                   : "mixed";
  const char *format = bf16 ? "v_wmma_f32_16x16x16_bf16" : "v_wmma_f32_16x16x16_f16";
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
    std::fprintf(csv_output ? stderr : stdout,
                 "%s SIMD unavailable in this build or on this host; skipped.\n", format);

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
  const char *names[] = {"v_wmma_i32_16x16x16_iu8", "v_wmma_i32_16x16x16_iu4",
                         "v_wmma_i32_16x16x32_iu4"};
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
    print_result(dx9 ? "v_fma_dx9_zero_f32" : "v_fma_f32", "loose",
                 modifiers ? "NEG/ABS/half" : "none", path, time, scalar / time);
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
  const char *names[] = {"v_mul_lo_u32",     "v_mul_hi_u32",  "v_mul_hi_i32",    "v_mul_i32_i24",
                         "v_mul_hi_i32_i24", "v_mul_u32_u24", "v_mul_hi_u32_u24"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
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
int integer_binary(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
  return Function(flags, mask, mode, d, a, b);
}

bool benchmark_half_binary(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {
      integer_binary<goc_rdna4_v_add_f16>,     integer_binary<goc_rdna4_v_sub_f16>,
      integer_binary<goc_rdna4_v_subrev_f16>,  integer_binary<goc_rdna4_v_mul_f16>,
      integer_binary<goc_rdna4_v_min_num_f16>, integer_binary<goc_rdna4_v_max_num_f16>,
      integer_binary<goc_rdna4_v_minimum_f16>, integer_binary<goc_rdna4_v_maximum_f16>};
  const char *names[] = {"v_add_f16",     "v_sub_f16",     "v_subrev_f16",  "v_mul_f16",
                         "v_min_num_f16", "v_max_num_f16", "v_minimum_f16", "v_maximum_f16"};
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
  const char *names[] = {"v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32",
                         "v_add_nc_i32", "v_sub_nc_i32", "v_add3_u32"};
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
  const char *names[] = {"v_min_i32",    "v_max_i32",    "v_min3_i32", "v_max3_i32",
                         "v_minmax_i32", "v_maxmin_i32", "v_med3_i32", "v_min_u32",
                         "v_max_u32",    "v_min3_u32",   "v_max3_u32", "v_minmax_u32",
                         "v_maxmin_u32", "v_med3_u32"};
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
  for (int fp64 = 0; fp64 < 2; ++fp64)
    for (int descriptor : {-1, 0, 5})
      for (uint32_t low : {UINT32_C(0), GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
        if (fp64 && descriptor >= 0)
          continue;
        uint64_t mode = low | (descriptor < 0 ? UINT64_C(0) : goc_test::dpp_modes[descriptor]);
        Registers r;
        r.output_regs = fp64 ? 2 : 1;
        for (int lane = 0; lane < 32; ++lane) {
          int source = lane;
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          double input = inputs[lane % 4];
          double want = std::ldexp(source < 0 ? 0.0 : inputs[source % 4], powers[lane % 4]);
          if (low)
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
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *) {
          return functions[fp64](flags, mask, modifiers, d, a, b);
        };
        const char *name = fp64 ? "v_ldexp_f64" : "v_ldexp_f32";
        const char *label = descriptor < 0    ? (low ? "ABS_A / half/clamp" : "none")
                            : descriptor == 0 ? (low ? "DPP8/modifiers" : "DPP8")
                                              : (low ? "DPP16/modifiers" : "DPP16");
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
    for (int descriptor : {-1, 0, 5})
      for (uint32_t low :
           {UINT32_C(0), GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP}) {
        if (fp64 && descriptor >= 0)
          continue;
        uint64_t mode = low | (descriptor < 0 ? UINT64_C(0) : goc_test::dpp_modes[descriptor]);
        Registers r;
        r.output_regs = 1;
        for (int lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = uint32_t(inputs[fp64][lane % 4]);
          r.data[1][lane] = uint32_t(inputs[fp64][lane % 4] >> 32);
          int source = lane;
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          r.expected[128 * (lane / 16) + lane % 16] =
              source < 0 ? 0 : uint32_t(exponents[source % 4]);
        }
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *,
                            const uint32_t *const *) {
          return functions[fp64](flags, mask, modifiers, d, a);
        };
        const char *name = fp64 ? "v_frexp_exp_i32_f64" : "v_frexp_exp_i32_f32";
        const char *label = descriptor < 0    ? (low ? "NEG/ABS/half/clamp" : "none")
                            : descriptor == 0 ? (low ? "DPP8/all modifiers" : "DPP8")
                                              : (low ? "DPP16/all modifiers" : "DPP16");
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
        if (descriptor >= 0 && cpu >= GOC_CPU_X86_64_V4) {
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

bool benchmark_half_exponent(uint64_t cpu, int iterations, int min_ms) {
  const uint16_t inputs[] = {0x3400, 0xbc00, 0x4400, 0x4c00, 1, 0x3ff, 0x7bff, 0};
  const int adjustments[] = {-2, -1, 0, 1, 24, 0, -1, 32767};
  const int exponents[] = {-1, 1, 3, 5, -23, -14, 16, 0};
  for (bool ldexp : {false, true})
    for (int descriptor : {-1, 0, 5})
      for (bool modified : {false, true}) {
        uint64_t mode = modified
                            ? GOC_ALU_ABS_A | GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | GOC_ALU_OMOD_HALF |
                                  GOC_ALU_CLAMP | (ldexp ? GOC_ALU_HIGH_B : 0)
                            : 0;
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
        Registers r;
        r.output_regs = 1;
        for (int lane = 0; lane < 32; ++lane) {
          auto half = inputs[lane % 8];
          int exponent = adjustments[lane % 8];
          r.data[0][lane] = uint32_t(half) | (uint32_t(half ^ 0x8000) << 16);
          r.data[4][lane] = uint16_t(exponent) | (uint32_t(uint16_t(exponent)) << 16);
          r.data[16][lane] = 0xfacecafe;
          int source = lane;
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          half = source < 0 ? 0 : inputs[source % 8];
          uint32_t result = source < 0 ? 0 : uint16_t(exponents[source % 8]);
          if (ldexp) {
            double x = goc_test::half_value(half);
            if (modified)
              x = std::abs(x);
            double value = std::ldexp(x, exponent);
            if (modified && std::abs(value) < 0x1p-13)
              value = 0;
            if (modified)
              value = !(value > 0) ? 0 : std::min(value * 0.5, 1.0);
            result = goc_test::half_bits(value);
          }
          r.expected[128 * (lane / 16) + lane % 16] =
              modified ? (result << 16) | 0xcafe : 0xface0000 | result;
        }
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *) {
          return ldexp ? goc_rdna4_v_ldexp_f16(flags, mask, modifiers, d, a, b)
                       : goc_rdna4_v_frexp_exp_i16_f16(flags, mask, modifiers, d, a);
        };
        const char *name = ldexp ? "v_ldexp_f16" : "v_frexp_exp_i16_f16";
        const char *label = descriptor < 0    ? (modified ? "ABS/hi/half/clamp" : "none")
                            : descriptor == 0 ? (modified ? "DPP8/modifiers" : "DPP8")
                                              : (modified ? "DPP16/modifiers" : "DPP16");
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
        if (descriptor >= 0 && cpu >= GOC_CPU_X86_64_V4) {
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

bool benchmark_half_unary(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_log_f16);
  const Unary functions[] = {
      goc_rdna4_v_trunc_f16, goc_rdna4_v_ceil_f16,      goc_rdna4_v_rndne_f16,
      goc_rdna4_v_floor_f16, goc_rdna4_v_sqrt_f16,      goc_rdna4_v_rcp_f16,
      goc_rdna4_v_rsq_f16,   goc_rdna4_v_exp_f16,       goc_rdna4_v_log_f16,
      goc_rdna4_v_fract_f16, goc_rdna4_v_frexp_mant_f16};
  const char *names[] = {"v_trunc_f16", "v_ceil_f16",  "v_rndne_f16",     "v_floor_f16",
                         "v_sqrt_f16",  "v_rcp_f16",   "v_rsq_f16",       "v_exp_f16",
                         "v_log_f16",   "v_fract_f16", "v_frexp_mant_f16"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
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

bool benchmark_mixed_fma(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_fma_mix_f32, goc_rdna4_v_fma_mixlo_f16,
                            goc_rdna4_v_fma_mixhi_f16};
  const char *names[] = {"v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16"};
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
      print_result(names[op], "loose", labels[variant], "scalar", scalar, 1, 32, "fp16-ovfl");
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd =
            measure(functions[op], GOC_CPU_X86_64_V3 | GOC_FP16_OVFL, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "loose", labels[variant], "x86-64-v3", simd, scalar / simd, 32,
                     "fp16-ovfl");
      }
#endif
      if (op) {
        double exact = measure(functions[op],
                               GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT | GOC_FP16_OVFL,
                               r, iterations, min_ms, mode);
        if (exact < 0)
          return false;
        print_result(names[op], "exact", labels[variant], "scalar", exact, 1, 32, "fp16-ovfl");
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
  const char *names[] = {"v_clz_i32_u32",      "v_ctz_i32_b32",      "v_cls_i32",
                         "v_bcnt_u32_b32",     "v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32",
                         "v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32"};

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
    const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *) {
      return functions[op](flags, mask, modifiers, d, a, b);
    };
    const char *label = "none";
    double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, 0, mask);
    if (scalar < 0)
      return false;
    print_result(names[op], "loose", label, "scalar", scalar, 1, op >= 6 ? 64 : 32);

#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3 && op != 5) {
      double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, 0, mask);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", label, "x86-64-v3", simd, scalar / simd, op >= 6 ? 64 : 32);
    }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
    if (cpu >= GOC_CPU_X86_64_V4) {
      double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, 0, mask);
      if (simd < 0)
        return false;
      print_result(names[op], "loose", label, "x86-64-v4", simd, scalar / simd, op >= 6 ? 64 : 32);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_boolean(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_and_b32);
  const Binary functions[] = {
      goc_rdna4_v_and_b32,     goc_rdna4_v_or_b32,      goc_rdna4_v_xor_b32,
      goc_test::boolean_not32, goc_rdna4_v_and_b16,     goc_rdna4_v_or_b16,
      goc_rdna4_v_xor_b16,     goc_test::boolean_not16, goc_rdna4_v_xnor_b32};
  const char *names[] = {"v_and_b32", "v_or_b32",  "v_xor_b32", "v_not_b32", "v_and_b16",
                         "v_or_b16",  "v_xor_b16", "v_not_b16", "v_xnor_b32"};
  for (int op = 0; op < 9; ++op)
    for (bool modified : {false, true}) {
      if ((op < 4 || op == 8) && modified)
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
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
  const Wmma functions[] = {goc_rdna4_v_bfe_u32,        goc_rdna4_v_bfe_i32,
                            goc_rdna4_v_bfi_b32,        goc_test::bitfield_mask,
                            goc_test::bitfield_reverse, goc_rdna4_v_alignbit_b32,
                            goc_rdna4_v_alignbyte_b32,  goc_rdna4_v_perm_b32};
  const char *names[] = {"v_bfe_u32",   "v_bfe_i32",      "v_bfi_b32",       "v_bfm_b32",
                         "v_bfrev_b32", "v_alignbit_b32", "v_alignbyte_b32", "v_perm_b32"};
  for (int op = 0; op < 8; ++op) {
    Registers r;
    r.output_regs = 1;
    std::mt19937 random(452);
    for (int lane = 0; lane < 32; ++lane) {
      for (int reg : {0, 4, 8})
        r.data[reg][lane] = random();
      if (op == 7)
        r.data[8][lane] &= 0x0f0f0f0f;
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
  const char *names[] = {"v_lshl_add_u32", "v_add_lshl_u32", "v_lshl_or_b32", "v_and_or_b32",
                         "v_or3_b32",      "v_xor3_b32",     "v_xad_u32",     "v_lerp_u8"};
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
  const char *names[] = {"v_sad_u8",  "v_sad_hi_u8",      "v_sad_u16",         "v_sad_u32",
                         "v_msad_u8", "v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8", "v_mqsad_u32_u8"};
  for (int op = 0; op < 8; ++op)
    for (int descriptor : {-1, 0, 5})
      for (uint32_t modifier : {UINT32_C(0), GOC_ALU_CLAMP}) {
        if (op >= 5 && descriptor >= 0)
          continue;
        uint64_t mode = modifier;
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
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
                                                  r.data[4][lane], c, bool(modifier));
          for (int reg = 0; reg < r.output_regs; ++reg)
            r.expected[128 * (lane / 16) + 16 * reg + lane % 16] = expected[reg];
        }
        if (descriptor >= 0)
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source = 0;
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
            uint32_t accumulator = r.data[8][lane];
            r.expected[128 * (lane / 16) + lane % 16] =
                goc_test::sad_reference(op, source < 0 ? 0 : r.data[0][source], 0, r.data[4][lane],
                                        &accumulator, bool(modifier))[0];
          }
        const char *label = descriptor < 0    ? (modifier ? "clamp" : "none")
                            : descriptor == 0 ? (modifier ? "DPP8/clamp" : "DPP8")
                                              : (modifier ? "DPP16/clamp" : "DPP16");
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

bool benchmark_shift(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_lshlrev_b32);
  const Binary functions[] = {goc_rdna4_v_lshlrev_b32, goc_rdna4_v_lshrrev_b32,
                              goc_rdna4_v_ashrrev_i32, goc_rdna4_v_lshlrev_b64,
                              goc_rdna4_v_lshrrev_b64, goc_rdna4_v_ashrrev_i64};
  const char *names[] = {"v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32",
                         "v_lshlrev_b64", "v_lshrrev_b64", "v_ashrrev_i64"};
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
    const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
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
  const char *names[] = {"v_sin_f16", "v_cos_f16"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
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
  const char *names[] = {"v_sin_f32", "v_cos_f32"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
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

bool benchmark_dpp_integer_add(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 2u, 3u})
    for (unsigned descriptor : {0u, 5u})
      for (bool modified : {false, true}) {
        uint32_t low = modified ? GOC_ALU_CLAMP : 0;
        uint64_t mode = goc_test::dpp_modes[descriptor] | low;
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = 0x7ffffff0u + lane;
          r.data[4][lane] = 0xffffffffu - lane;
          r.data[8][lane] = lane;
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          r.expected[128 * (lane / 16) + lane % 16] =
              goc_test::integer_add_reference(op,
                                              op == 2   ? r.data[0][lane]
                                              : src < 0 ? 0
                                                        : r.data[0][src],
                                              op != 2   ? r.data[4][lane]
                                              : src < 0 ? 0
                                                        : r.data[4][src],
                                              r.data[8][lane], low);
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
          return goc_test::integer_add_functions[op](flags, mask, mode, d, a, b, c);
        };
        const char *label = descriptor == 0 ? (modified ? "DPP8/CLAMP" : "DPP8")
                                            : (modified ? "DPP16/CLAMP" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::integer_add_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::integer_add_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::integer_add_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_integer_mad(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 1u, 2u, 3u})
    for (unsigned descriptor : {0u, 5u})
      for (bool modified : {false, true}) {
        uint32_t low =
            modified ? GOC_ALU_CLAMP | (op < 2 ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B : 0) : 0;
        uint64_t mode = goc_test::dpp_modes[descriptor] | low;
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = 0x1234abcd + lane * 0x01010101u;
          r.data[4][lane] = 0x98760123 - lane * 0x01010101u;
          r.data[8][lane] = 0x7ffffff0u + lane;
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          r.expected[128 * (lane / 16) + lane % 16] = goc_test::integer_mad_reference(
              op, src < 0 ? 0 : r.data[0][src], r.data[4][lane], r.data[8][lane], low);
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
          return goc_test::integer_mad_functions[op](flags, mask, mode, d, a, b, c);
        };
        const char *label =
            descriptor == 0
                ? (modified ? (op < 2 ? "DPP8/HIGH_AB/CLAMP" : "DPP8/CLAMP") : "DPP8")
                : (modified ? (op < 2 ? "DPP16/HIGH_AB/CLAMP" : "DPP16/CLAMP") : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::integer_mad_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::integer_mad_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::integer_mad_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_arithmetic(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 3u, 9u, 17u})
    for (unsigned descriptor : {0u, 5u})
      for (bool modified : {false, true}) {
        uint32_t low = modified ? GOC_ALU_NEG_A | GOC_ALU_OMOD_2 : 0;
        uint64_t mode = goc_test::dpp_modes[descriptor] | low;
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = bits(float(int(lane % 11) - 5));
          r.data[4][lane] = bits(float(int(lane % 7) - 3));
          r.data[8][lane] = bits(float(int(lane % 9) - 4));
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          r.expected[128 * (lane / 16) + lane % 16] = goc_test::dpp_arithmetic_reference(
              op, src < 0 ? 0 : r.data[0][src], r.data[4][lane], r.data[8][lane], low);
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
          return goc_test::dpp_arithmetic_call(op, flags, mask, mode, d, a, b, c);
        };
        const char *label = descriptor == 0 ? (modified ? "DPP8/NEG/mul2" : "DPP8")
                                            : (modified ? "DPP16/NEG/mul2" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::dpp_arithmetic_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::dpp_arithmetic_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::dpp_arithmetic_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_bitfield(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {1u, 3u, 7u})
    for (unsigned descriptor : {0u, 5u}) {
      uint64_t mode = goc_test::dpp_modes[descriptor];
      Registers r;
      r.output_regs = 1;
      for (unsigned lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0x9e3779b9u * (lane + 1);
        r.data[4][lane] = lane * 17;
        r.data[8][lane] = 0x7fffffffu - lane;
      }
      for (unsigned lane = 0; lane < 32; ++lane) {
        int src;
        goc_test::dpp_source(mode, UINT32_MAX, lane, src);
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::bitfield_reference(
            op, src < 0 ? 0 : r.data[0][src], r.data[4][lane], r.data[8][lane]);
      }
      auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
        return goc_test::bitfield_functions[op](flags, mask, mode, d, a, b, c);
      };
      const char *label = descriptor == 0 ? "DPP8" : "DPP16";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(goc_test::bitfield_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(goc_test::bitfield_names[op], "loose", label, "x86-64-v3", simd,
                     scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(goc_test::bitfield_names[op], "loose", label, "x86-64-v4", simd,
                     scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_dpp_integer_ternary(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 3u, 7u})
    for (unsigned descriptor : {0u, 5u}) {
      uint64_t mode = goc_test::dpp_modes[descriptor];
      Registers r;
      r.output_regs = 1;
      for (unsigned lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0x9e3779b9u * (lane + 1);
        r.data[4][lane] = lane * 17;
        r.data[8][lane] = 0x7fffffffu - lane;
      }
      for (unsigned lane = 0; lane < 32; ++lane) {
        int src;
        goc_test::dpp_source(mode, UINT32_MAX, lane, src);
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::integer_ternary_reference(
            op, src < 0 ? 0 : r.data[0][src], r.data[4][lane], r.data[8][lane]);
      }
      auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
        return goc_test::integer_ternary_functions[op](flags, mask, mode, d, a, b, c);
      };
      const char *label = descriptor == 0 ? "DPP8" : "DPP16";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(goc_test::integer_ternary_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(goc_test::integer_ternary_names[op], "loose", label, "x86-64-v3", simd,
                     scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(goc_test::integer_ternary_names[op], "loose", label, "x86-64-v4", simd,
                     scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_dpp_integer_minmax(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 6u, 7u, 13u})
    for (unsigned descriptor : {0u, 5u}) {
      uint64_t mode = goc_test::dpp_modes[descriptor];
      Registers r;
      r.output_regs = 1;
      for (unsigned lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0x9e3779b9u * (lane + 1);
        r.data[4][lane] = lane * 17;
        r.data[8][lane] = 0x7fffffffu - lane;
      }
      for (unsigned lane = 0; lane < 32; ++lane) {
        int src;
        goc_test::dpp_source(mode, UINT32_MAX, lane, src);
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::integer_minmax_reference(
            op, src < 0 ? 0 : r.data[0][src], r.data[4][lane], r.data[8][lane]);
      }
      auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
        return goc_test::integer_minmax_functions[op](flags, mask, mode, d, a, b, c);
      };
      const char *label = descriptor == 0 ? "DPP8" : "DPP16";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(goc_test::integer_minmax_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(goc_test::integer_minmax_names[op], "loose", label, "x86-64-v3", simd,
                     scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(goc_test::integer_minmax_names[op], "loose", label, "x86-64-v4", simd,
                     scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_dpp_half_fma(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 1u})
    for (unsigned descriptor : {0u, 5u})
      for (unsigned modified = 0; modified < 2; ++modified) {
        uint64_t mode =
            goc_test::dpp_modes[descriptor] | (modified ? (op == 0 ? 8191u : 6107u) : 0);
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = 0x3c00bc00u + lane;
          r.data[4][lane] = 0x4000b800u + lane;
          r.data[8][lane] = 0x38004200u + lane;
          r.data[16][lane] = op == 1 ? r.data[8][lane] : 0x12345678;
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          r.expected[128 * (lane / 16) + lane % 16] =
              goc_test::half_fma_result(op, src < 0 ? 0 : r.data[0][src], r.data[4][lane],
                                        r.data[8][lane], uint32_t(mode), r.data[16][lane]);
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
          if (op == 1)
            std::memcpy(d[0], c[0], 32 * sizeof(uint32_t));
          return goc_test::half_fma_functions[op](flags, mask, mode, d, a, b, c);
        };
        const char *label = descriptor == 0 ? (modified ? "DPP8/all modifiers" : "DPP8")
                                            : (modified ? "DPP16/all modifiers" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::half_fma_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::half_fma_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::half_fma_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_half_minmax(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 2u, 8u})
    for (unsigned descriptor : {0u, 5u})
      for (unsigned modified = 0; modified < 2; ++modified) {
        uint64_t mode = goc_test::dpp_modes[descriptor] | (modified ? 8191u : 0);
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = 0x3c00bc00u + lane;
          r.data[4][lane] = 0x4000b800u + lane;
          r.data[8][lane] = 0x38004200u + lane;
          r.data[16][lane] = 0x12345678;
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          uint32_t value =
              goc_test::half_minmax_reference(op, src < 0 ? 0 : r.data[0][src], r.data[4][lane],
                                              r.data[8][lane], uint32_t(mode), false);
          unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
          r.expected[128 * (lane / 16) + lane % 16] =
              (r.data[16][lane] & ~(UINT32_C(65535) << shift)) | (value << shift);
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
          return goc_test::half_minmax_functions[op](flags, mask, mode, d, a, b, c);
        };
        const char *label = descriptor == 0 ? (modified ? "DPP8/all modifiers" : "DPP8")
                                            : (modified ? "DPP16/all modifiers" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::half_minmax_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::half_minmax_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::half_minmax_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_integer16_ternary(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 1u, 6u, 7u})
    for (unsigned descriptor : {0u, 5u})
      for (unsigned modified = 0; modified < 2; ++modified) {
        uint64_t mode = goc_test::dpp_modes[descriptor] |
                        (modified ? goc_test::integer16_ternary_mode_bits(op < 2 ? 31 : 15) : 0);
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = 0x9e3779b9u * (lane + 1);
          r.data[4][lane] = 0x01010101u * (lane * 17 + 1);
          r.data[8][lane] = 0x87654321u - lane;
          r.data[16][lane] = 0x12345678;
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          r.expected[128 * (lane / 16) + lane % 16] = goc_test::integer16_ternary_reference(
              op, src < 0 ? 0 : r.data[0][src], r.data[4][lane], r.data[8][lane], uint32_t(mode),
              r.data[16][lane]);
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
          return goc_test::integer16_ternary_functions[op](flags, mask, mode, d, a, b, c);
        };
        const char *label =
            descriptor == 0 ? (modified ? (op < 2 ? "DPP8/HIGH/CLAMP" : "DPP8/HIGH") : "DPP8")
                            : (modified ? (op < 2 ? "DPP16/HIGH/CLAMP" : "DPP16/HIGH") : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::integer16_ternary_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::integer16_ternary_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::integer16_ternary_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_half_unary(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {2u, 4u, 5u, 7u, 8u, 10u})
    for (unsigned descriptor : {0u, 5u})
      for (unsigned modified = 0; modified < 2; ++modified) {
        uint64_t mode =
            goc_test::dpp_modes[descriptor] | (modified ? goc_test::half_unary_modifiers(127) : 0);
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = 0x3c004000u + lane;
          r.data[16][lane] = 0x12345678;
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          uint32_t value = goc_test::half_unary_reference(op, src < 0 ? 0 : r.data[0][src],
                                                          uint32_t(mode), false);
          unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
          r.expected[128 * (lane / 16) + lane % 16] =
              (r.data[16][lane] & ~(UINT32_C(65535) << shift)) | (value << shift);
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *, const uint32_t *const *) {
          return goc_test::half_unary_functions[op](flags, mask, mode, d, a);
        };
        const char *label = descriptor == 0 ? (modified ? "DPP8/all modifiers" : "DPP8")
                                            : (modified ? "DPP16/all modifiers" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::half_unary_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::half_unary_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::half_unary_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_half_binary(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 2u, 3u, 6u})
    for (unsigned descriptor : {0u, 5u})
      for (unsigned modified = 0; modified < 2; ++modified) {
        uint64_t mode = goc_test::dpp_modes[descriptor] |
                        (modified ? goc_test::half_binary_modifiers(1023) : 0);
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = 0x3c00bc00u + lane;
          r.data[4][lane] = 0x4000b800u + lane;
          r.data[16][lane] = 0x12345678;
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          uint32_t value = goc_test::half_binary_reference(op, src < 0 ? 0 : r.data[0][src],
                                                           r.data[4][lane], uint32_t(mode), false);
          unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
          r.expected[128 * (lane / 16) + lane % 16] =
              (r.data[16][lane] & ~(UINT32_C(65535) << shift)) | (value << shift);
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *) {
          return goc_test::half_binary_functions[op](flags, mask, mode, d, a, b);
        };
        const char *label = descriptor == 0 ? (modified ? "DPP8/all modifiers" : "DPP8")
                                            : (modified ? "DPP16/all modifiers" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::half_binary_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::half_binary_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::half_binary_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_integer16(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 4u, 8u, 9u})
    for (unsigned descriptor : {0u, 5u})
      for (unsigned modified = 0; modified < 2; ++modified) {
        uint64_t mode = goc_test::dpp_modes[descriptor] |
                        (modified ? goc_test::integer16_mode_bits(op < 4 ? 15 : 7) : 0);
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = 0x9e3779b9u * (lane + 1);
          r.data[4][lane] = 0x01010101u * (lane * 17 + 1);
          r.data[16][lane] = 0x12345678;
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          r.expected[128 * (lane / 16) + lane % 16] = goc_test::integer16_reference(
              op, src < 0 ? 0 : r.data[0][src], r.data[4][lane], r.data[16][lane], uint32_t(mode));
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *) {
          return goc_test::integer16_functions[op](flags, mask, mode, d, a, b);
        };
        const char *label =
            descriptor == 0 ? (modified ? (op < 4 ? "DPP8/HIGH/CLAMP" : "DPP8/HIGH") : "DPP8")
                            : (modified ? (op < 4 ? "DPP16/HIGH/CLAMP" : "DPP16/HIGH") : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::integer16_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::integer16_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::integer16_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_boolean16(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {4u, 7u})
    for (unsigned descriptor : {0u, 5u})
      for (unsigned modified = 0; modified < 2; ++modified) {
        uint64_t mode =
            goc_test::dpp_modes[descriptor] | (modified ? goc_test::boolean_mode(op, 7) : 0);
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = 0x9e3779b9u * (lane + 1);
          r.data[4][lane] = lane * 17;
          r.data[16][lane] = 0x12345678;
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          r.expected[128 * (lane / 16) + lane % 16] = goc_test::boolean_reference(
              op, src < 0 ? 0 : r.data[0][src], r.data[4][lane], r.data[16][lane], uint32_t(mode));
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *) {
          return goc_test::boolean_functions[op](flags, mask, mode, d, a, b);
        };
        const char *label = descriptor == 0 ? (modified ? "DPP8/HIGH" : "DPP8")
                                            : (modified ? "DPP16/HIGH" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::dpp_boolean16_names[op - 4], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::dpp_boolean16_names[op - 4], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::dpp_boolean16_names[op - 4], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_integer_mul(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {3u, 4u, 5u, 6u})
    for (unsigned descriptor : {0u, 5u})
      for (unsigned clamp = 0; clamp <= unsigned(goc_test::integer_mul_can_clamp(op)); ++clamp) {
        uint64_t mode = goc_test::dpp_modes[descriptor] | (clamp ? GOC_ALU_CLAMP : 0);
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = 0x9e3779b9u * (lane + 1);
          r.data[4][lane] = lane * 17;
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          r.expected[128 * (lane / 16) + lane % 16] = goc_test::integer_mul_reference(
              op, src < 0 ? 0 : r.data[0][src], r.data[4][lane], clamp);
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *) {
          return goc_test::integer_mul_functions[op](flags, mask, mode, d, a, b);
        };
        const char *label =
            descriptor == 0 ? (clamp ? "DPP8/CLAMP" : "DPP8") : (clamp ? "DPP16/CLAMP" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::dpp_integer_mul_names[op - 3], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::dpp_integer_mul_names[op - 3], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::dpp_integer_mul_names[op - 3], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp_integer(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {0u, 3u, 4u, 6u, 10u, 13u})
    for (unsigned descriptor : {0u, 5u}) {
      uint64_t mode = goc_test::dpp_modes[descriptor];
      Registers r;
      r.output_regs = 1;
      for (unsigned lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0x9e3779b9u * (lane + 1);
        r.data[4][lane] = lane * 17;
      }
      for (unsigned lane = 0; lane < 32; ++lane) {
        int src;
        goc_test::dpp_source(mode, UINT32_MAX, lane, src);
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::dpp_integer_reference(
            op, src < 0 ? 0 : r.data[0][src], r.data[4][lane], lane);
      }
      auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
        return goc_test::dpp_integer_functions[op](flags, mask, mode, d, a, b);
      };
      const char *label = descriptor == 0 ? "DPP8" : "DPP16";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(goc_test::dpp_integer_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(goc_test::dpp_integer_names[op], "loose", label, "x86-64-v3", simd,
                     scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(goc_test::dpp_integer_names[op], "loose", label, "x86-64-v4", simd,
                     scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_dpp_unary(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned op : {3u, 4u, 5u, 9u})
    for (unsigned descriptor : {0u, 5u})
      for (bool modified : {false, true}) {
        uint32_t low = modified ? GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP : 0;
        uint64_t mode = goc_test::dpp_modes[descriptor] | low;
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          const float values[] = {0.25f, 1, 4, 16};
          r.data[0][lane] = bits(values[lane % 4] * (modified ? -1 : 1));
        }
        for (unsigned lane = 0; lane < 32; ++lane) {
          int src;
          goc_test::dpp_source(mode, UINT32_MAX, lane, src);
          r.expected[128 * (lane / 16) + lane % 16] =
              bits(goc_test::unary_reference(op, src < 0 ? 0 : goc::as_float(r.data[0][src]), low));
        }
        auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *, const uint32_t *const *) {
          return goc_test::unary_functions[op](flags, mask, mode, d, a);
        };
        const char *label = descriptor == 0 ? (modified ? "DPP8/ABS/half" : "DPP8")
                                            : (modified ? "DPP16/ABS/half" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::unary_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::unary_names[op], "loose", label, "x86-64-v3", simd, scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::unary_names[op], "loose", label, "x86-64-v4", simd, scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_dpp16(uint64_t cpu, int iterations, int min_ms) {
  for (bool modified : {false, true}) {
    uint64_t mode = goc_test::dpp16_mode(0x10f, modified, 1, 15, 15) |
                    (modified ? GOC_ALU_NEG_A | GOC_ALU_OMOD_2 : 0);
    Registers r;
    r.output_regs = 1;
    for (unsigned lane = 0; lane < 32; ++lane) {
      r.data[0][lane] = bits(float(lane + 1));
      r.data[4][lane] = bits(2);
      r.data[8][lane] = bits(float(100 + lane));
    }
    for (unsigned lane = 0; lane < 32; ++lane) {
      int src;
      goc_test::dpp16_reference(0x10f, modified, 1, 15, 15, UINT32_MAX, lane, src);
      int a = src < 0 ? 0 : src + 1;
      float value = modified ? float((100 + int(lane) - 2 * a) * 2) : float(100 + lane + 2 * a);
      r.expected[128 * (lane / 16) + lane % 16] = bits(value);
    }
    const char *label = modified ? "DPP16/FI/NEG/mul2" : "DPP16";
    double scalar = measure(goc_rdna4_v_fma_f32, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
    if (scalar < 0)
      return false;
    print_result("v_fma_f32", "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3) {
      double simd = measure(goc_rdna4_v_fma_f32, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_fma_f32", "loose", label, "x86-64-v3", simd, scalar / simd);
    }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
    if (cpu >= GOC_CPU_X86_64_V4) {
      double simd = measure(goc_rdna4_v_fma_f32, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_fma_f32", "loose", label, "x86-64-v4", simd, scalar / simd);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_dpp8(uint64_t cpu, int iterations, int min_ms) {
  for (bool modified : {false, true}) {
    uint32_t selectors = 0;
    for (unsigned i = 0; i < 8; ++i)
      selectors |= (7 - i) << (3 * i);
    uint64_t mode = GOC_DPP8 | (uint64_t(selectors) << GOC_DPP8_SELECT_SHIFT) |
                    (modified ? GOC_DPP_FI | GOC_ALU_NEG_A | GOC_ALU_OMOD_2 : 0);
    Registers r;
    r.output_regs = 1;
    for (unsigned lane = 0; lane < 32; ++lane) {
      r.data[0][lane] = bits(float(lane + 1));
      r.data[4][lane] = bits(2);
      r.data[8][lane] = bits(float(100 + lane));
    }
    for (unsigned lane = 0; lane < 32; ++lane) {
      unsigned src = lane ^ 7;
      float value = modified ? float((100 + int(lane) - 2 * int(src + 1)) * 2)
                             : float(100 + lane + 2 * (src + 1));
      r.expected[128 * (lane / 16) + lane % 16] = bits(value);
    }
    const char *label = modified ? "DPP8/FI/NEG/mul2" : "DPP8";
    double scalar = measure(goc_rdna4_v_fma_f32, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
    if (scalar < 0)
      return false;
    print_result("v_fma_f32", "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3) {
      double simd = measure(goc_rdna4_v_fma_f32, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_fma_f32", "loose", label, "x86-64-v3", simd, scalar / simd);
    }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
    if (cpu >= GOC_CPU_X86_64_V4) {
      double simd = measure(goc_rdna4_v_fma_f32, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_fma_f32", "loose", label, "x86-64-v4", simd, scalar / simd);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_permlane(uint64_t cpu, int iterations, int min_ms) {
  const char *names[] = {"v_permlane16_b32", "v_permlanex16_b32", "v_permlane16_var_b32",
                         "v_permlanex16_var_b32"};
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned mode : {0u, 3u}) {
      Registers r;
      r.output_regs = 1;
      for (unsigned lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = 0xabc00000 + lane;
        r.data[4][lane] = lane * 7 + 3;
      }
      for (unsigned lane = 0; lane < 32; ++lane)
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::permlane_reference(
            op, UINT32_MAX, mode, lane, r.data[0], r.data[4], 0, 0x12345678, 0x9abcdef0);
      auto fn = [op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
        return goc_test::permlane_call(op, flags, mask, mode, d, a, b, 0x12345678, 0x9abcdef0);
      };
      double scalar = measure(fn, GOC_SEMANTICS_EXACT_EMPIRICAL, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      const char *label = mode ? "FI/bound_ctrl" : "none";
      print_result(names[op], "exact", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_CPU_X86_64_V3, r, iterations,
                              min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_CPU_X86_64_V4, r, iterations,
                              min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_cndmask(uint64_t cpu, int iterations, int min_ms) {
  for (bool half : {false, true})
    for (bool modified : {false, true}) {
      unsigned compact = modified ? (half ? 119 : 7) : 0;
      uint32_t mode = goc_test::cndmask_mode(compact);
      Registers r;
      r.output_regs = 1;
      const uint32_t condition = 0x96969696;
      for (unsigned lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = (lane * 0x7395a831u) ^ 0xa7925163u;
        r.data[4][lane] = (lane * 0x83a1459du) ^ 0x5389d241u;
        r.expected[128 * (lane / 16) + lane % 16] =
            goc_test::cndmask_reference(half, r.data[0][lane], r.data[4][lane], r.data[16][lane],
                                        compact, (condition >> lane) & 1);
      }
      auto fn = [half](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *) {
        return (half ? goc_rdna4_v_cndmask_b16 : goc_rdna4_v_cndmask_b32)(flags, mask, mode, d, a,
                                                                          b, condition);
      };
      const char *name = half ? "v_cndmask_b16" : "v_cndmask_b32";
      const char *label = modified ? (half ? "ABS/NEG/high" : "ABS/NEG") : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(name, "loose", label, "scalar", scalar, 1);

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

bool benchmark_interp16(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_interp_p10_f16_f32, goc_rdna4_v_interp_p2_f16_f32,
                            goc_rdna4_v_interp_p10_rtz_f16_f32, goc_rdna4_v_interp_p2_rtz_f16_f32};
  const char *names[] = {"v_interp_p10_f16_f32", "v_interp_p2_f16_f32", "v_interp_p10_rtz_f16_f32",
                         "v_interp_p2_rtz_f16_f32"};
  for (unsigned op = 0; op < 4; ++op)
    for (bool modified : {false, true}) {
      unsigned compact = modified ? 61 : 0;
      uint32_t mode = goc_test::interp16_mode(op, compact, modified ? 7 : 0);
      Registers r;
      r.output_regs = 1;
      for (unsigned lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = uint32_t(0x3800 + lane * 13) | (uint32_t(0x4000 + lane * 7) << 16);
        r.data[4][lane] = bits(float(int(lane) - 16) * 0.0625f);
        r.data[8][lane] = op & 1
                              ? bits(float(int(lane) - 8) * 0.03125f)
                              : uint32_t(0x3000 + lane * 17) | (uint32_t(0x3400 + lane * 11) << 16);
      }
      for (unsigned lane = 0; lane < 32; ++lane)
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::interp16_reference(
            op, r.data[0], r.data[4], r.data[8], r.data[16][lane], lane, compact, false);
      const char *label = modified ? "NEG/clamp/high/wait7" : "none";
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

bool benchmark_interp32(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_interp_p10_f32, goc_rdna4_v_interp_p2_f32};
  const char *names[] = {"v_interp_p10_f32", "v_interp_p2_f32"};
  for (unsigned op = 0; op < 2; ++op)
    for (bool modified : {false, true}) {
      unsigned compact = modified ? 13 : 0;
      uint32_t mode = goc_test::interp32_mode(compact, modified ? 7 : 0);
      Registers r;
      r.output_regs = 1;
      uint32_t inputs[4][32];
      goc_test::interp32_capture_inputs(inputs);
      for (unsigned lane = 0; lane < 32; ++lane) {
        for (unsigned reg = 0; reg < 3; ++reg)
          r.data[4 * reg][lane] = inputs[reg][lane];
        r.expected[128 * (lane / 16) + lane % 16] = goc_test::interp32_bits(
            goc_test::interp32_reference(op, inputs[0], inputs[1], inputs[2], lane, compact));
      }
      auto fn = functions[op];
      const char *label = modified ? "NEG/clamp/wait7" : "none";
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

bool benchmark_rcp_iflag(uint64_t cpu, int iterations, int min_ms) {
  struct IflagRegisters : Registers {
    uint32_t status = 0, want = 0;
    unsigned mode = 0;

    bool correct() const {
      if (status != want)
        return false;
      for (unsigned lane = 0; lane < 32; ++lane)
        if (!goc_test::rcp_iflag_close(data[16][lane],
                                       goc_test::rcp_iflag_reference(data[0][lane], mode)))
          return false;
      return true;
    }
  };

  for (unsigned m : {0u, 15u, 31u}) {
    IflagRegisters r;
    r.output_regs = 1;
    r.mode = m;
    for (unsigned lane = 0; lane < 32; ++lane)
      r.data[0][lane] = goc_test::rcp_iflag_input(lane * 32);
    r.want = goc_test::rcp_iflag_status(r.data[0], UINT32_MAX, m, 0x15);
    auto fn = [&r](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *, const uint32_t *const *) {
      return goc_rdna4_v_rcp_iflag_f32(flags, mask, mode, d, a, &r.status, 0x15);
    };
    const char *label = m == 0 ? "none" : m == 15 ? "ABS/NEG/half" : "ABS/NEG/half/clamp";
    uint32_t mode = goc_test::rcp_iflag_mode(m);
    double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
    if (scalar < 0)
      return false;
    print_result("v_rcp_iflag_f32", "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3) {
      double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_rcp_iflag_f32", "loose", label, "x86-64-v3", simd, scalar / simd);
    }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
    if (cpu >= GOC_CPU_X86_64_V4) {
      double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_rcp_iflag_f32", "loose", label, "x86-64-v4", simd, scalar / simd);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_scalar_round(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, want = 0;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const { return result == want; }
  };

  // GFX1201 capture: FP32 sample 10 / FP16 sample 0x3e01.
  const uint32_t gold[8][2] = {{0x40000000u, 0x40000000u}, {0x00004000u, 0x00004000u},
                               {0x3f800000u, 0x3f800000u}, {0x00003c00u, 0x00003c00u},
                               {0x3f800000u, 0x3f800000u}, {0x00003c00u, 0x00003c00u},
                               {0x40000000u, 0x40000000u}, {0x00004000u, 0x00004000u}};
  for (unsigned op = 0; op < 8; ++op)
    for (unsigned state = 0; state < 2; ++state) {
      ScalarRegisters r;
      r.want = gold[op][state];
      uint32_t w[2];
      goc_test::scalar_round_inputs(op & 1 ? 0x3e01 : 10, op & 1, w);
      auto fn = [&r, op, &w](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *) {
        return goc_test::scalar_round_functions[op](flags, mask, mode, &r.result, w[0]);
      };
      double scalar =
          measure(fn, GOC_SEMANTICS_EXACT_EMPIRICAL | (state ? GOC_FP_FLUSH_INPUT_DENORMALS : 0), r,
                  iterations, min_ms, 0);
      if (scalar < 0)
        return false;
      print_result(goc_test::scalar_round_names[op], "exact", "none", "scalar", scalar, 1, 32,
                   state ? "flush-input" : "none");
    }
  return true;
}

bool benchmark_scalar_fma(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, want = 0;
    bool half = false;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const {
      return goc_test::scalar_fp_canonical(result, half) ==
             goc_test::scalar_fp_canonical(want, half);
    }
  };

  // GFX1201 capture: input pair 3001, preserve / flush both + OVFL / flush output.
  const uint32_t gold[4][3] = {{0x53af5ff5u, 0x53af5ff5u, 0x53af5ff5u},
                               {0x00007fd8u, 0x00007fd8u, 0x00007fd8u},
                               {0x3f800001u, 0x3f800001u, 0x3f800001u},
                               {0x52f2f70cu, 0x52f2f70cu, 0x52f2f70cu}};
  const unsigned states[] = {3, 4, 1};
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant = 0; variant < 3; ++variant) {
      ScalarRegisters r;
      r.half = op == 1;
      r.want = gold[op][variant];
      uint32_t w[3];
      goc_test::scalar_fma_inputs(3001, op == 1, w);
      auto fn = [&r, op, &w](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *) {
        r.result = w[2];
        unsigned k = op < 2 ? op : op == 2 ? 8 : 20;
        return goc_test::scalar_fma_call(k, flags, mask, mode, &r.result, w[0], w[1]);
      };
      double scalar =
          measure(fn, goc_test::scalar_fp_flags(states[variant]), r, iterations, min_ms, 0);
      if (scalar < 0)
        return false;
      print_result(goc_test::scalar_fma_names[op], "loose", "none", "scalar", scalar, 1, 32,
                   variant == 0   ? "none"
                   : variant == 1 ? "flush-io-ovfl"
                                  : "flush-output");
    }
  return true;
}

bool benchmark_scalar_compare(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, want = 0;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const { return result == want; }
  };

  // GFX1201 capture: integer pair 333 / floating pair 3, preserve / flush input.
  const uint32_t gold[46][2] = {
      {0u, 0u}, {1u, 1u}, {0u, 0u}, {0u, 0u}, {1u, 1u}, {1u, 1u}, {0u, 0u}, {1u, 1u},
      {0u, 0u}, {0u, 0u}, {1u, 1u}, {1u, 1u}, {0u, 0u}, {1u, 1u}, {0u, 0u}, {1u, 1u},
      {0u, 0u}, {1u, 1u}, {1u, 0u}, {1u, 0u}, {0u, 1u}, {0u, 1u}, {1u, 1u}, {1u, 1u},
      {0u, 0u}, {0u, 0u}, {1u, 0u}, {1u, 0u}, {0u, 1u}, {0u, 1u}, {1u, 1u}, {1u, 1u},
      {0u, 0u}, {0u, 0u}, {1u, 0u}, {1u, 0u}, {0u, 1u}, {0u, 1u}, {1u, 1u}, {1u, 1u},
      {0u, 0u}, {0u, 0u}, {1u, 0u}, {1u, 0u}, {0u, 1u}, {0u, 1u}};
  for (unsigned op = 0; op < 46; ++op)
    for (unsigned state = 0; state < (op < 18 ? 1u : 2u); ++state) {
      ScalarRegisters r;
      r.want = gold[op][state];
      uint32_t w[4];
      goc_test::scalar_compare_inputs(op < 18 ? 333 : 3, op, w);
      uint64_t a = (uint64_t(w[1]) << 32) | w[0], b = (uint64_t(w[3]) << 32) | w[2];
      auto fn = [&r, op, a, b](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                               const uint32_t *const *, const uint32_t *const *,
                               const uint32_t *const *) {
        return goc_test::scalar_compare_call(op, flags, mask, mode, &r.result, a, b);
      };
      double scalar =
          measure(fn, GOC_SEMANTICS_EXACT_EMPIRICAL | (state ? GOC_FP_FLUSH_INPUT_DENORMALS : 0), r,
                  iterations, min_ms, 0);
      if (scalar < 0)
        return false;
      print_result(goc_test::scalar_compare_names[op], "exact", "none", "scalar", scalar, 1, 32,
                   state ? "flush-input" : "none");
    }
  return true;
}

bool benchmark_scalar_convert(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, want = 0;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const { return result == want; }
  };

  // GFX1201 capture: input pair 3001, preserve / flush both + OVFL / flush output.
  const uint32_t gold[8][3] = {
      {0x4ea5e5eeu, 0x4ea5e5eeu, 0x4ea5e5eeu}, {0x4ea5e5eeu, 0x4ea5e5eeu, 0x4ea5e5eeu},
      {0x7fffffffu, 0x7fffffffu, 0x7fffffffu}, {0xffffffffu, 0xffffffffu, 0xffffffffu},
      {0x00007c00u, 0x00007bffu, 0x00007c00u}, {0x39772000u, 0x39772000u, 0x39772000u},
      {0x39772000u, 0x39772000u, 0x39772000u}, {0x00007bffu, 0x00007bffu, 0x00007bffu}};
  const unsigned states[] = {3, 4, 1};
  for (unsigned op = 0; op < 8; ++op)
    for (unsigned variant = 0; variant < 3; ++variant) {
      ScalarRegisters r;

      r.want = gold[op][variant];
      uint32_t w[2];
      goc_test::scalar_convert_inputs(3001, op, w);
      auto fn = [&r, op, &w](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *) {
        return goc_test::scalar_convert_call(op, flags, mask, mode, &r.result, w[0], w[1]);
      };
      double scalar =
          measure(fn, goc_test::scalar_fp_flags(states[variant]), r, iterations, min_ms, 0);
      if (scalar < 0)
        return false;
      print_result(goc_test::scalar_convert_names[op], "loose", "none", "scalar", scalar, 1, 32,
                   variant == 0   ? "none"
                   : variant == 1 ? "flush-io-ovfl"
                                  : "flush-output");
    }
  return true;
}

bool benchmark_scalar_fp(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, want = 0;
    bool half = false;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const {
      return goc_test::scalar_fp_canonical(result, half) ==
             goc_test::scalar_fp_canonical(want, half);
    }
  };

  // GFX1201 capture: input pair 3001, preserve / flush both + OVFL / flush output.
  const uint32_t gold[14][3] = {
      {0x52f2f70au, 0x52f2f70au, 0x52f2f70au}, {0x00007fd8u, 0x00007fd8u, 0x00007fd8u},
      {0x52f2f70au, 0x52f2f70au, 0x52f2f70au}, {0x0000ffd8u, 0x0000ffd8u, 0x0000ffd8u},
      {0x31680ccbu, 0x31680ccbu, 0x31680ccbu}, {0x00007fd8u, 0x00007fd8u, 0x00007fd8u},
      {0x1df47fd8u, 0x1df47fd8u, 0x1df47fd8u}, {0x0000f70au, 0x0000f70au, 0x0000f70au},
      {0x52f2f70au, 0x52f2f70au, 0x52f2f70au}, {0x0000f70au, 0x0000f70au, 0x0000f70au},
      {0x1df47fd8u, 0x1df47fd8u, 0x1df47fd8u}, {0x00007fd8u, 0x00007fd8u, 0x00007fd8u},
      {0x52f2f70au, 0x52f2f70au, 0x52f2f70au}, {0x00007fd8u, 0x00007fd8u, 0x00007fd8u}};
  const unsigned states[] = {3, 4, 1};
  for (unsigned op = 0; op < 14; ++op)
    for (unsigned variant = 0; variant < 3; ++variant) {
      ScalarRegisters r;
      r.half = op & 1;
      r.want = gold[op][variant];
      uint32_t w[2];
      goc_test::scalar_fp_inputs(3001, op & 1, w);
      auto fn = [&r, op, &w](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *) {
        return goc_test::scalar_fp_functions[op](flags, mask, mode, &r.result, w[0], w[1]);
      };
      double scalar =
          measure(fn, goc_test::scalar_fp_flags(states[variant]), r, iterations, min_ms, 0);
      if (scalar < 0)
        return false;
      print_result(goc_test::scalar_fp_names[op], "loose", "none", "scalar", scalar, 1, 32,
                   variant == 0   ? "none"
                   : variant == 1 ? "flush-io-ovfl"
                                  : "flush-output");
    }
  return true;
}

bool benchmark_scalar_field(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, cc = 1;
    uint64_t wide = 0, want = 0;
    uint32_t want_cc = 0;
    bool is_wide = false;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const { return (is_wide ? wide : result) == want && cc == want_cc; }
  };

  // GFX1201 capture: input pair 9999, incoming SCC 1.
  const uint64_t gold[20][2] = {{UINT64_C(0x0), 0u},        {UINT64_C(0x0), 0u},
                                {UINT64_C(0x0), 0u},        {UINT64_C(0x0), 0u},
                                {UINT64_C(0xc0000000), 1u}, {UINT64_C(0xc000000000000000), 1u},
                                {UINT64_C(0x12), 1u},       {UINT64_C(0x23), 1u},
                                {UINT64_C(0xe), 1u},        {UINT64_C(0x1d), 1u},
                                {UINT64_C(0x1), 1u},        {UINT64_C(0x1), 1u},
                                {UINT64_C(0x3), 1u},        {UINT64_C(0x2), 1u},
                                {UINT64_C(0x3), 1u},        {UINT64_C(0x2), 1u},
                                {UINT64_C(0x1b842d72), 1u}, {UINT64_C(0x35f600bc1b842d72), 1u},
                                {UINT64_C(0x5b842d72), 1u}, {UINT64_C(0x75f600bc1b842d72), 1u}};
  uint32_t words[4];
  goc_test::scalar_field_inputs(9999, words);
  const uint64_t a = (uint64_t(words[1]) << 32) | words[0];
  const uint64_t b = (uint64_t(words[3]) << 32) | words[2];
  for (unsigned op = 0; op < 20; ++op) {
    ScalarRegisters r;
    r.want = gold[op][0];
    r.want_cc = uint32_t(gold[op][1]);
    r.is_wide = op == 2 || op == 3 || op == 5 || op == 17 || op == 19;
    auto fn = [&r, op, a, b](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *) {
      r.result = uint32_t(a);
      r.wide = a;
      r.cc = 1;
      return goc_test::scalar_field_call(op, flags, mask, mode, &r.result, &r.wide, a, b, &r.cc);
    };
    double scalar = measure(fn, GOC_SEMANTICS_EXACT_EMPIRICAL, r, iterations, min_ms, 0);
    if (scalar < 0)
      return false;
    print_result(goc_test::scalar_field_names[op], "exact", "none", "scalar", scalar, 1);
  }
  return true;
}

bool benchmark_scalar_sign_extend(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, want = 0;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const { return result == want; }
  };

  const auto functions = {goc_rdna4_s_sext_i32_i8, goc_rdna4_s_sext_i32_i16};
  const char *names[] = {"s_sext_i32_i8", "s_sext_i32_i16"};
  unsigned op = 0;
  for (auto function : functions) {
    ScalarRegisters r;
    r.want = op ? 0xffff8081u : 0xffffff81u;
    auto fn = [&r, function](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *) {
      return function(flags, mask, mode, &r.result, 0xabcd8081);
    };
    double scalar = measure(fn, GOC_SEMANTICS_EXACT_EMPIRICAL, r, iterations, min_ms, 0);
    if (scalar < 0)
      return false;
    print_result(names[op++], "exact", "none", "scalar", scalar, 1);
  }
  return true;
}

bool benchmark_scalar_pack(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, cc = 1;
    uint64_t wide = 0, want = 0;
    uint32_t want_cc = 0;
    bool is_wide = false;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const { return (is_wide ? wide : result) == want && cc == want_cc; }
  };

  // GFX1201 capture: input pair 333, incoming SCC 1.
  const uint64_t gold[11][2] = {{UINT64_C(0xc3cc5f78), 1u},
                                {UINT64_C(0x6b8c5f78), 1u},
                                {UINT64_C(0xc3cc6b4e), 1u},
                                {UINT64_C(0x6b8c6b4e), 1u},
                                {UINT64_C(0x3ccf30fc33ff3fc0), 1u},
                                {UINT64_C(0x6b4e5f78), 1u},
                                {UINT64_C(0xfe3996de6b4e5f78), 1u},
                                {UINT64_C(0xff), 1u},
                                {UINT64_C(0xffff), 1u},
                                {UINT64_C(0xffffffff), 1u},
                                {UINT64_C(0xffffffffffffffff), 1u}};
  uint32_t words[4];
  goc_test::scalar_integer_inputs(333, words);
  const uint64_t a = (uint64_t(words[1]) << 32) | words[0];
  const uint64_t b = (uint64_t(words[3]) << 32) | words[2];
  for (unsigned op = 0; op < 11; ++op) {
    ScalarRegisters r;
    r.want = gold[op][0];
    r.want_cc = uint32_t(gold[op][1]);
    r.is_wide = op == 4 || op == 6 || op == 8 || op == 10;
    auto fn = [&r, op, a, b](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *) {
      r.result = uint32_t(a);
      r.cc = 1;
      return goc_test::scalar_pack_call(op, flags, mask, mode, &r.result, &r.wide, a, b, &r.cc, 1);
    };
    double scalar = measure(fn, GOC_SEMANTICS_EXACT_EMPIRICAL, r, iterations, min_ms, 0);
    if (scalar < 0)
      return false;
    print_result(goc_test::scalar_pack_names[op], "exact", "none", "scalar", scalar, 1);
  }
  return true;
}

bool benchmark_scalar_bits(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, cc = 1;
    uint64_t wide = 0, want = 0;
    uint32_t want_cc = 0;
    bool is_wide = false;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const { return (is_wide ? wide : result) == want && cc == want_cc; }
  };

  // GFX1201 capture: input pair 333, incoming SCC 1.
  const uint64_t gold[30][2] = {{UINT64_C(0x6b0c4348), 1u}, {UINT64_C(0x2a3880da6b0c4348), 1u},
                                {UINT64_C(0x6bcedffc), 1u}, {UINT64_C(0xfefbffde6bcedffc), 1u},
                                {UINT64_C(0xc29cb4), 1u},   {UINT64_C(0xd4c37f0400c29cb4), 1u},
                                {UINT64_C(0x94f3bcb7), 1u}, {UINT64_C(0xd5c77f2594f3bcb7), 1u},
                                {UINT64_C(0x94312003), 1u}, {UINT64_C(0x104002194312003), 1u},
                                {UINT64_C(0xff3d634b), 1u}, {UINT64_C(0x2b3c80fbff3d634b), 1u},
                                {UINT64_C(0x421c30), 1u},   {UINT64_C(0xd401160400421c30), 1u},
                                {UINT64_C(0xff7f7f7b), 1u}, {UINT64_C(0xff3d96ffff7f7f7b), 1u},
                                {UINT64_C(0x94b1a087), 1u}, {UINT64_C(0x1c6692194b1a087), 1u},
                                {UINT64_C(0x1efa72d6), 1u}, {UINT64_C(0x1efa72d67b699c7f), 1u},
                                {UINT64_C(0xe5f78000), 1u}, {UINT64_C(0x996de6b4e5f78000), 1u},
                                {UINT64_C(0x6b4e5), 1u},    {UINT64_C(0xfe3996de6b4e5), 1u},
                                {UINT64_C(0x6b4e5), 1u},    {UINT64_C(0xffffe3996de6b4e5), 1u},
                                {UINT64_C(0x422982bc), 1u}, {UINT64_C(0x18c641ac), 1u},
                                {UINT64_C(0xc5ffbf8c), 1u}, {UINT64_C(0x2072bb4c), 1u}};
  uint32_t words[4];
  goc_test::scalar_integer_inputs(333, words);
  const uint64_t a = (uint64_t(words[1]) << 32) | words[0];
  const uint64_t b = (uint64_t(words[3]) << 32) | words[2];
  for (unsigned op = 0; op < 30; ++op) {
    ScalarRegisters r;
    r.want = gold[op][0];
    r.want_cc = uint32_t(gold[op][1]);
    r.is_wide = op < 26 && (op & 1);
    auto fn = [&r, op, a, b](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *) {
      r.result = uint32_t(a);
      r.cc = 1;
      return goc_test::scalar_bits_call(op, flags, mask, mode, &r.result, &r.wide, a, b, &r.cc);
    };
    double scalar = measure(fn, GOC_SEMANTICS_EXACT_EMPIRICAL, r, iterations, min_ms, 0);
    if (scalar < 0)
      return false;
    print_result(goc_test::scalar_bits_names[op], "exact", "none", "scalar", scalar, 1);
  }
  return true;
}

bool benchmark_scalar_integer(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, cc = 1;
    uint64_t wide = 0, want = 0;
    uint32_t want_cc = 0;
    bool is_wide = false;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const { return (is_wide ? wide : result) == want && cc == want_cc; }
  };

  // GFX1201 capture: input pair 42, incoming SCC 1, literal -1.
  const uint64_t gold[20][2] = {{UINT64_C(0x0), 1u},
                                {UINT64_C(0xfffffffc), 0u},
                                {UINT64_C(0x0), 0u},
                                {UINT64_C(0xfffffffc), 0u},
                                {UINT64_C(0x1), 1u},
                                {UINT64_C(0xfffffffb), 0u},
                                {UINT64_C(0x2), 1u},
                                {UINT64_C(0x4), 1u},
                                {UINT64_C(0xfffffffe), 1u},
                                {UINT64_C(0x2), 0u},
                                {UINT64_C(0x2), 0u},
                                {UINT64_C(0xfffffffe), 1u},
                                {UINT64_C(0xfffffffc), 1u},
                                {UINT64_C(0x1), 1u},
                                {UINT64_C(0xffffffff), 1u},
                                {UINT64_C(0x8000000000000000), 1u},
                                {UINT64_C(0x7ffffffffffffffc), 1u},
                                {UINT64_C(0xfffffffffffffffc), 1u},
                                {UINT64_C(0xfffffffd), 0u},
                                {UINT64_C(0x2), 1u}};
  uint32_t words[4];
  goc_test::scalar_integer_inputs(42, words);
  const uint64_t a = (uint64_t(words[1]) << 32) | words[0];
  const uint64_t b = (uint64_t(words[3]) << 32) | words[2];
  for (unsigned op = 0; op < 20; ++op) {
    ScalarRegisters r;
    r.want = gold[op][0];
    r.want_cc = uint32_t(gold[op][1]);
    r.is_wide = op >= 15 && op <= 17;
    auto fn = [&r, op, a, b](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *) {
      r.result = uint32_t(a);
      r.cc = 1;
      return goc_test::scalar_integer_call(op, flags, mask, mode, &r.result, &r.wide, a, b, &r.cc,
                                           1, 0xffff);
    };
    double scalar = measure(fn, GOC_SEMANTICS_EXACT_EMPIRICAL, r, iterations, min_ms, 0);
    if (scalar < 0)
      return false;
    print_result(goc_test::scalar_integer_names[op], "exact", "none", "scalar", scalar, 1);
  }
  return true;
}

bool benchmark_pseudo_scalar(int iterations, int min_ms) {
  struct ScalarRegisters : Registers {
    uint32_t result = 0, want = 0;
    bool half = false;

    ScalarRegisters() { output_regs = 0; }

    bool correct() const { return goc_test::pseudo_scalar_close(result, want, half); }
  };

  for (unsigned op = 0; op < 10; ++op)
    for (unsigned variant = 0; variant < 3; ++variant) {
      unsigned state = variant == 0   ? 3
                       : variant == 1 ? 7
                                      : 4,
               m = variant == 0   ? 0
                   : variant == 1 ? 7
                                  : 31,
               sample = 42;
      ScalarRegisters r;
      r.half = op & 1;
      r.data[0][0] = goc_test::pseudo_scalar_input(op & 1, goc_test::pseudo_scalar_samples[sample]);
      r.want =
          goc_test::pseudo_scalar_outputs[goc_test::pseudo_scalar_blocks[(state * 10 + op) * 32 +
                                                                         m]][sample];
      auto fn = [&r, op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                         const uint32_t *const *a, const uint32_t *const *,
                         const uint32_t *const *) {
        return goc_test::pseudo_scalar_functions[op](flags, mask, mode, &r.result, a[0][0]);
      };
      double scalar = measure(fn, goc_test::pseudo_scalar_flags(state), r, iterations, min_ms,
                              goc_test::pseudo_scalar_mode(m));
      if (scalar < 0)
        return false;
      print_result(goc_test::pseudo_scalar_names[op], "loose",
                   variant == 0   ? "none"
                   : variant == 1 ? "ABS/NEG/mul2"
                                  : "ABS/NEG/half/clamp",
                   "scalar", scalar, 1, 32,
                   variant == 0   ? "none"
                   : variant == 1 ? "fp16-ovfl"
                                  : "flush-io-ovfl");
    }
  return true;
}

bool benchmark_float_compare(uint64_t cpu, int iterations, int min_ms) {
  struct CompareRegisters : Registers {
    uint32_t result = 0, want = 0;

    CompareRegisters() { output_regs = 0; }

    bool correct() const { return result == want; }
  };

  for (unsigned op = 0; op < 84; ++op)
    for (unsigned config = 0; config < 3; ++config) {
      unsigned m = config ? (op < 28 ? 63 : 15) : 0;
      bool flush = config == 2;
      CompareRegisters r;
      for (unsigned lane = 0; lane < 32; ++lane) {
        uint32_t w[4];
        goc_test::float_compare_inputs(op / 28, 128 + lane, w);
        r.data[0][lane] = w[0];
        r.data[1][lane] = w[1];
        r.data[4][lane] = w[2];
        r.data[5][lane] = w[3];
        r.want |= uint32_t(goc_test::float_compare_reference(op, m, flush, w)) << lane;
      }
      auto fn = [&r, op, flush](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *) {
        return goc_test::float_compare_functions[op](flags | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                                         GOC_SEMANTICS_STRICT |
                                                         (flush ? GOC_FP_FLUSH_INPUT_DENORMALS : 0),
                                                     mask, mode, &r.result, a, b);
      };
      auto name = goc_test::float_compare_names[op];
      const char *label = m ? (op < 28 ? "ABS/NEG/high" : "ABS/NEG") : "none",
                 *state = flush ? "flush-input" : "none";
      uint32_t mode = goc_test::float_compare_mode(m);
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(name, "exact", label, "scalar", scalar, 1, 32, state);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(name, "exact", label, "x86-64-v3", simd, scalar / simd, 32, state);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(name, "exact", label, "x86-64-v4", simd, scalar / simd, 32, state);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_integer_compare(uint64_t cpu, int iterations, int min_ms) {
  struct CompareRegisters : Registers {
    uint32_t result = 0, want = 0;

    CompareRegisters() { output_regs = 0; }

    bool correct() const { return result == want; }
  };

  for (unsigned op = 0; op < 72; ++op)
    for (unsigned m : {0u, 3u}) {
      if (op >= 24 && m)
        continue;
      CompareRegisters r;
      for (unsigned lane = 0; lane < 32; ++lane) {
        uint32_t w[4];
        goc_test::integer_compare_inputs(op / 24, 128 + lane, w);
        r.data[0][lane] = w[0];
        r.data[1][lane] = w[1];
        r.data[4][lane] = w[2];
        r.data[5][lane] = w[3];
        r.want |= uint32_t(goc_test::integer_compare_reference(op, m, w)) << lane;
      }
      auto fn = [&r, op](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *) {
        return goc_test::integer_compare_functions[op](flags | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                                           GOC_SEMANTICS_STRICT,
                                                       mask, mode, &r.result, a, b);
      };
      auto name = goc_test::integer_compare_names[op];
      const char *label = m ? "high" : "none";
      uint32_t mode = goc_test::integer_compare_mode(m);
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(name, "exact", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(name, "exact", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(name, "exact", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_class(uint64_t cpu, int iterations, int min_ms) {
  using Fn = decltype(&goc_rdna4_v_cmp_class_f16);
  const Fn functions[] = {goc_rdna4_v_cmp_class_f16, goc_rdna4_v_cmpx_class_f16,
                          goc_rdna4_v_cmp_class_f32, goc_rdna4_v_cmpx_class_f32,
                          goc_rdna4_v_cmp_class_f64, goc_rdna4_v_cmpx_class_f64};
  const char *names[] = {"v_cmp_class_f16",  "v_cmpx_class_f16", "v_cmp_class_f32",
                         "v_cmpx_class_f32", "v_cmp_class_f64",  "v_cmpx_class_f64"};

  struct ClassRegisters : Registers {
    uint32_t result = 0, want = 0;

    ClassRegisters() { output_regs = 0; }

    bool correct() const { return result == want; }
  };

  for (unsigned op = 0; op < 6; ++op)
    for (bool modified : {false, true}) {
      unsigned compact = modified ? (op < 2 ? 15 : 3) : 0;
      uint32_t mode = goc_test::class_mode(compact);
      ClassRegisters r;
      for (unsigned lane = 0; lane < 32; ++lane) {
        uint64_t value = op < 2 ? uint64_t(goc_test::class_edges16[lane % 16]) |
                                      (uint64_t(goc_test::class_edges16[(lane + 7) % 16]) << 16)
                         : op < 4 ? goc_test::class_edges32[lane % 16]
                                  : goc_test::class_edges64[lane % 16];
        r.data[0][lane] = uint32_t(value);
        r.data[1][lane] = uint32_t(value >> 32);
        r.data[4][lane] = (lane * 0x9e3779b9u) ^ 0xa5a59669u;
        r.want |= uint32_t(goc_test::class_reference(op / 2, r.data[0][lane], r.data[1][lane],
                                                     r.data[4][lane], compact))
                  << lane;
      }
      auto fn = [&r, op, &functions](uint64_t flags, uint64_t mask, uint64_t mode,
                                     uint32_t *const *, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *) {
        return functions[op](flags | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask,
                             mode, &r.result, a, b);
      };
      const char *label = modified ? (op < 2 ? "ABS/NEG/high" : "ABS/NEG") : "none";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "exact", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_trig_preop(uint64_t cpu, int iterations, int min_ms) {
  for (bool modified : {false, true}) {
    unsigned compact = modified ? 31 : 0;
    uint32_t mode = goc_test::trig_preop_mode(compact);
    Registers r;
    r.output_regs = 2;
    for (unsigned lane = 0; lane < 32; ++lane) {
      r.data[0][lane] = 0xabcdef;
      r.data[1][lane] = goc_test::trig_preop_exponent(lane) << 20;
      r.data[4][lane] = goc_test::trig_preop_selector(lane);
      for (unsigned reg = 0; reg < 2; ++reg)
        r.expected[128 * (lane / 16) + 16 * reg + lane % 16] =
            uint32_t(goc_test::trig_preop_samples[compact][lane] >> (32 * reg));
    }
    auto fn = [](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
      return goc_rdna4_v_trig_preop_f64(
          flags | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, mode, d, a, b);
    };
    const char *label = modified ? "ABS/NEG/half/clamp" : "none";
    double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
    if (scalar < 0)
      return false;
    print_result("v_trig_preop_f64", "exact", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3) {
      double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_trig_preop_f64", "exact", label, "x86-64-v3", simd, scalar / simd);
    }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
    if (cpu >= GOC_CPU_X86_64_V4) {
      double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_trig_preop_f64", "exact", label, "x86-64-v4", simd, scalar / simd);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_mullit(uint64_t cpu, int iterations, int min_ms) {
  for (bool modified : {false, true}) {
    uint32_t mode =
        modified ? GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_ABS_C | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP
                 : 0;
    Registers r;
    r.output_regs = 1;
    for (int lane = 0; lane < 32; ++lane) {
      r.data[0][lane] = bits(float(lane - 16) * 0.25f);
      r.data[4][lane] = bits(float(lane % 7 - 3) * 0.5f);
      r.data[8][lane] = bits(float(lane % 5 - 2));
      r.expected[128 * (lane / 16) + lane % 16] =
          goc_test::mullit_reference(r.data[0][lane], r.data[4][lane], r.data[8][lane], mode);
    }
    const char *label = modified ? "ABS/NEG/half/clamp" : "none";
    double scalar = measure(goc_rdna4_v_mullit_f32, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
    if (scalar < 0)
      return false;
    print_result("v_mullit_f32", "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3) {
      double simd = measure(goc_rdna4_v_mullit_f32, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_mullit_f32", "loose", label, "x86-64-v3", simd, scalar / simd);
    }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
    if (cpu >= GOC_CPU_X86_64_V4) {
      double simd = measure(goc_rdna4_v_mullit_f32, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_mullit_f32", "loose", label, "x86-64-v4", simd, scalar / simd);
    }
#endif
  }
  (void)cpu;
  return true;
}

bool benchmark_pack(uint64_t cpu, int iterations, int min_ms) {
  for (unsigned variant : {0u, 1u, 2u, 65u}) {
    Registers r;
    r.output_regs = 1;
    for (unsigned lane = 0; lane < 32; ++lane) {
      r.data[0][lane] = lane * 0x397fa113u;
      r.data[4][lane] = lane * 0x159bc385u + 0xfc017c01u;
      r.data[16][lane] = 0xcafebeef;
      r.expected[128 * (lane / 16) + lane % 16] =
          goc_test::pack_reference(variant, r.data[0][lane], r.data[4][lane], r.data[16][lane]);
    }
    const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *) {
      return goc_test::pack_call(variant, flags, mask, mode, d, a, b);
    };
    uint32_t mode = goc_test::pack_mode(variant);
    const char *name = variant < 2 ? "v_sat_pk_u8_i16" : "v_pack_b32_f16";
    const char *label = variant == 1 ? "high" : variant == 65 ? "ABS/NEG/high" : "none";
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

bool benchmark_swmmac_integer(uint64_t cpu, int iterations, int min_ms) {
  using Fn = decltype(&goc_rdna4_v_swmmac_i32_16x16x32_iu8);
  const Fn functions[] = {goc_rdna4_v_swmmac_i32_16x16x32_iu8, goc_rdna4_v_swmmac_i32_16x16x32_iu4,
                          goc_rdna4_v_swmmac_i32_16x16x64_iu4};
  const char *names[] = {"v_swmmac_i32_16x16x32_iu8", "v_swmmac_i32_16x16x32_iu4",
                         "v_swmmac_i32_16x16x64_iu4"};

  struct SparseRegisters {
    uint32_t data[24][32] = {}, *v[24];
    unsigned regs;
    uint64_t expected;

    bool correct() const {
      uint64_t digest = UINT64_C(14695981039346656037);
      for (unsigned reg = 0; reg < regs; ++reg)
        for (unsigned lane = 0; lane < 32; ++lane)
          for (unsigned byte = 0; byte < 4; ++byte) {
            digest ^= (data[16 + reg][lane] >> (8 * byte)) & 255;
            digest *= UINT64_C(1099511628211);
          }
      return digest == expected;
    }
  };

  for (unsigned op = 0; op < 3; ++op)
    for (bool modified : {false, true}) {
      unsigned variant = modified ? (op == 2 ? 7u : 15u) : 0;
      uint32_t initial[15][32];
      goc_test::swmmac_integer_capture_inputs(0, initial);
      SparseRegisters r;
      r.regs = 8;
      r.expected = goc_test::swmmac_integer_capture_digests[op][0][variant];
      for (unsigned reg = 0; reg < 24; ++reg)
        r.v[reg] = r.data[reg];
      std::memcpy(r.data, initial, 2 * 32 * sizeof(uint32_t));
      std::memcpy(r.data + 4, initial + 2, 4 * 32 * sizeof(uint32_t));
      const uint32_t *index[] = {initial[14]};
      const uint32_t mode = (variant & 3) | ((variant & 4) << 4) | ((variant & 8) << 4);
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        // Reset the in/out accumulator on every call; all paths include this cost.
        for (unsigned reg = 0; reg < r.regs; ++reg)
          std::memcpy(d[reg], initial[6 + reg], 32 * sizeof(uint32_t));
        return functions[op](flags, mask, modifiers, d, a, b, index);
      };
      const char *label = modified ? (op == 2 ? "signed/clamp" : "signed/clamp/key1") : "none";
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

bool benchmark_swmmac8(uint64_t cpu, int iterations, int min_ms) {
  using Fn = decltype(&goc_rdna4_v_swmmac_f32_16x16x32_fp8_fp8);
  const Fn functions[] = {
      goc_rdna4_v_swmmac_f32_16x16x32_fp8_fp8, goc_rdna4_v_swmmac_f32_16x16x32_fp8_bf8,
      goc_rdna4_v_swmmac_f32_16x16x32_bf8_fp8, goc_rdna4_v_swmmac_f32_16x16x32_bf8_bf8};
  const char *names[] = {"v_swmmac_f32_16x16x32_fp8_fp8", "v_swmmac_f32_16x16x32_fp8_bf8",
                         "v_swmmac_f32_16x16x32_bf8_fp8", "v_swmmac_f32_16x16x32_bf8_bf8"};

  struct SparseRegisters {
    uint32_t data[24][32] = {}, *v[24];
    unsigned regs;
    uint64_t expected;

    bool correct() const {
      uint64_t digest = UINT64_C(14695981039346656037);
      for (unsigned reg = 0; reg < regs; ++reg)
        for (unsigned lane = 0; lane < 32; ++lane)
          for (unsigned byte = 0; byte < 4; ++byte) {
            digest ^= (data[16 + reg][lane] >> (8 * byte)) & 255;
            digest *= UINT64_C(1099511628211);
          }
      return digest == expected;
    }
  };

  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant : {0u, 1u}) {
      uint32_t initial[15][32];
      goc_test::swmmac8_capture_inputs(op, 0, initial);
      SparseRegisters r;
      r.regs = 8;
      r.expected = goc_test::swmmac8_capture_digests[op][0][variant];
      for (unsigned reg = 0; reg < 24; ++reg)
        r.v[reg] = r.data[reg];
      std::memcpy(r.data, initial, 2 * 32 * sizeof(uint32_t));
      std::memcpy(r.data + 4, initial + 2, 4 * 32 * sizeof(uint32_t));
      const uint32_t *index[] = {initial[14]};
      const uint32_t mode = variant ? GOC_SWMMAC_INDEX_KEY_1 : 0;
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        // Reset the in/out accumulator on every call; all paths include this cost.
        for (unsigned reg = 0; reg < r.regs; ++reg)
          std::memcpy(d[reg], initial[6 + reg], 32 * sizeof(uint32_t));
        return functions[op](flags, mask, modifiers, d, a, b, index);
      };
      const char *label = variant ? "key1" : "none";
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

bool benchmark_swmmac16(uint64_t cpu, int iterations, int min_ms) {
  using Fn = decltype(&goc_rdna4_v_swmmac_f32_16x16x32_f16);
  const Fn functions[] = {goc_rdna4_v_swmmac_f32_16x16x32_f16, goc_rdna4_v_swmmac_f32_16x16x32_bf16,
                          goc_rdna4_v_swmmac_f16_16x16x32_f16,
                          goc_rdna4_v_swmmac_bf16_16x16x32_bf16};
  const char *names[] = {"v_swmmac_f32_16x16x32_f16", "v_swmmac_f32_16x16x32_bf16",
                         "v_swmmac_f16_16x16x32_f16", "v_swmmac_bf16_16x16x32_bf16"};

  struct SparseRegisters {
    uint32_t data[24][32] = {}, *v[24];
    unsigned regs;
    uint64_t expected;

    bool correct() const {
      uint64_t digest = UINT64_C(14695981039346656037);
      for (unsigned reg = 0; reg < regs; ++reg)
        for (unsigned lane = 0; lane < 32; ++lane)
          for (unsigned byte = 0; byte < 4; ++byte) {
            digest ^= (data[16 + reg][lane] >> (8 * byte)) & 255;
            digest *= UINT64_C(1099511628211);
          }
      return digest == expected;
    }
  };

  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant : {0u, 25u}) {
      uint32_t initial[21][32];
      goc_test::swmmac16_capture_inputs(op, initial);
      SparseRegisters r;
      r.regs = op >= 2 ? 4 : 8;
      r.expected = goc_test::swmmac16_capture_digests[op][variant];
      for (unsigned reg = 0; reg < 24; ++reg)
        r.v[reg] = r.data[reg];
      std::memcpy(r.data, initial, 12 * 32 * sizeof(uint32_t));
      const uint32_t *index[] = {initial[20]};
      const uint32_t mode = (variant & 3) | ((variant & 12) << 1) | ((variant & 16) << 3);
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        // Reset the in/out accumulator on every call; all paths include this cost.
        for (unsigned reg = 0; reg < r.regs; ++reg)
          std::memcpy(d[reg], initial[12 + reg], 32 * sizeof(uint32_t));
        return functions[op](flags, mask, modifiers, d, a, b, index);
      };
      const char *label = variant ? "NEG/key1" : "none";
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

bool benchmark_mad64(uint64_t cpu, int iterations, int min_ms) {
  using Mad = decltype(&goc_rdna4_v_mad_co_u64_u32);
  const Mad functions[] = {goc_rdna4_v_mad_co_u64_u32, goc_rdna4_v_mad_co_i64_i32};
  const char *names[] = {"v_mad_co_u64_u32", "v_mad_co_i64_i32"};

  struct MadRegisters : Registers {
    uint32_t carry = 0, expected_carry = 0;

    bool correct() const { return Registers::correct() && carry == expected_carry; }
  };

  const uint32_t factors[] = {0, 1, 0x7fffffff, 0x80000000, 0xfffffffe, UINT32_MAX};
  const uint64_t addends[] = {0, 1, UINT64_MAX >> 1, UINT64_C(1) << 63, UINT64_MAX - 1, UINT64_MAX};
  const uint64_t semantics = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  for (unsigned op = 0; op < 2; ++op)
    for (bool clamp : {false, true}) {
      MadRegisters r;
      r.output_regs = 2;
      for (unsigned lane = 0; lane < 32; ++lane) {
        uint32_t a = factors[lane % 6], b = factors[(lane / 6 + lane) % 6];
        uint64_t c = addends[(lane * 5 + 2) % 6];
        r.data[0][lane] = a;
        r.data[4][lane] = b;
        r.data[8][lane] = uint32_t(c);
        r.data[9][lane] = uint32_t(c >> 32);
        auto gold = goc_test::mad64_reference(op, a, b, c, clamp);
        r.expected[128 * (lane / 16) + lane % 16] = uint32_t(gold.value);
        r.expected[128 * (lane / 16) + 16 + lane % 16] = uint32_t(gold.value >> 32);
        r.expected_carry |= uint32_t(gold.carry) << lane;
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
        return functions[op](flags, mask, modifiers, d, &r.carry, a, b, c);
      };
      uint32_t mode = clamp ? GOC_ALU_CLAMP : 0;
      const char *label = clamp ? "clamp" : "none";
      double scalar = measure(fn, semantics, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "exact", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, semantics | GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, semantics | GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_carry(uint64_t cpu, int iterations, int min_ms) {
  const char *names[] = {"v_add_co_u32",    "v_sub_co_u32",    "v_subrev_co_u32",
                         "v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32"};

  struct CarryRegisters : Registers {
    uint32_t carry = 0, expected_carry = 0;

    bool correct() const { return Registers::correct() && carry == expected_carry; }
  };

  const uint32_t values[] = {0, 1, 0x7fffffff, 0x80000000, 0xfffffffe, UINT32_MAX};
  const uint64_t semantics = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  for (unsigned op = 0; op < 6; ++op)
    for (bool clamp : {false, true}) {
      CarryRegisters r;
      r.output_regs = 1;
      for (unsigned lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = values[lane % 6];
        r.data[4][lane] = values[(lane / 6 + lane) % 6];
        auto gold = goc_test::carry_reference(op, r.data[0][lane], r.data[4][lane],
                                              (0xa5a5a5a5 >> lane) & 1, clamp);
        r.expected[128 * (lane / 16) + lane % 16] = gold.value;
        r.expected_carry |= uint32_t(gold.carry) << lane;
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
        return goc_test::carry_functions[op](flags, mask, modifiers, d, &r.carry, a, b, 0xa5a5a5a5);
      };
      uint32_t mode = clamp ? GOC_ALU_CLAMP : 0;
      const char *label = clamp ? "clamp" : "none";
      double scalar = measure(fn, semantics, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "exact", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, semantics | GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, semantics | GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_div_fmas(uint64_t cpu, int iterations, int min_ms) {
  using Fmas = decltype(&goc_rdna4_v_div_fmas_f32);
  const Fmas functions[] = {goc_rdna4_v_div_fmas_f32, goc_rdna4_v_div_fmas_f64};
  const char *names[] = {"v_div_fmas_f32", "v_div_fmas_f64"};
  const uint64_t semantics = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  for (unsigned op = 0; op < 2; ++op)
    for (bool modified : {false, true}) {
      Registers r;
      r.output_regs = op ? 2 : 1;
      unsigned fraction = op ? 52 : 23, bias = op ? 1023 : 127;
      uint64_t sign = UINT64_C(1) << (op ? 63 : 31);
      uint32_t mode = modified ? GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_ABS_B | GOC_ALU_NEG_C |
                                     GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP
                               : 0;
      for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned operand = 0; operand < 3; ++operand) {
          int exponent = int(bias) + int((lane * 3 + operand * 5) % 19) - 9;
          uint64_t raw =
              (uint64_t(exponent) << fraction) |
              ((UINT64_C(0x123456789ab) * (lane + operand + 1)) & ((UINT64_C(1) << fraction) - 1));
          if ((lane + operand) & 1)
            raw |= sign;
          r.data[4 * operand][lane] = uint32_t(raw);
          r.data[4 * operand + 1][lane] = uint32_t(raw >> 32);
        }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
        return functions[op](flags, mask, modifiers, d, a, b, c, 0xa5a5a5a5);
      };
      if (fn(semantics, UINT32_MAX, mode, r.v + 16, r.v, r.v + 4, r.v + 8) != GOC_SUCCESS)
        return false;
      for (int reg = 0; reg < r.output_regs; ++reg)
        for (unsigned lane = 0; lane < 32; ++lane)
          r.expected[128 * (lane / 16) + 16 * reg + lane % 16] = r.data[16 + reg][lane];
      const char *label = modified ? "mixed/half/clamp" : "none";
      double scalar = measure(fn, semantics, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "exact", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, semantics | GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, semantics | GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_div_scale(uint64_t cpu, int iterations, int min_ms) {
  using Scale = decltype(&goc_rdna4_v_div_scale_f32);
  const Scale functions[] = {goc_rdna4_v_div_scale_f32, goc_rdna4_v_div_scale_f64};
  const char *names[] = {"v_div_scale_f32", "v_div_scale_f64"};
  const uint64_t semantics = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

  struct ScaleRegisters : Registers {
    uint32_t condition = 0, expected_condition = 0;

    bool correct() const { return Registers::correct() && condition == expected_condition; }
  };

  for (unsigned op = 0; op < 2; ++op)
    for (bool modified : {false, true}) {
      ScaleRegisters r;
      r.output_regs = op ? 2 : 1;
      unsigned fraction = op ? 52 : 23, bias = op ? 1023 : 127, threshold = op ? 768 : 96;
      uint64_t sign = UINT64_C(1) << (op ? 63 : 31);
      unsigned exponents[] = {0,           1,        fraction,         fraction + 1, bias - 1,
                              bias,        bias + 1, bias + threshold, 2 * bias - 1, 2 * bias,
                              2 * bias + 1};
      uint32_t mode = modified ? GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_NEG_C | GOC_ALU_OMOD_HALF |
                                     GOC_ALU_CLAMP
                               : 0;
      for (unsigned lane = 0; lane < 32; ++lane) {
        uint64_t b = (uint64_t(exponents[lane % 11]) << fraction) | (lane + 1);
        uint64_t c = (uint64_t(exponents[(lane * 7 + 3) % 11]) << fraction) | (lane * 5 + 1);
        if (lane & 4)
          b |= sign;
        if (lane & 8)
          c |= sign;
        uint64_t raw[] = {lane & 1 ? c : b, b, c};
        for (unsigned operand = 0; operand < 3; ++operand) {
          r.data[4 * operand][lane] = uint32_t(raw[operand]);
          r.data[4 * operand + 1][lane] = uint32_t(raw[operand] >> 32);
        }
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
        return functions[op](flags, mask, modifiers, d, &r.condition, a, b, c);
      };
      if (fn(semantics, UINT32_MAX, mode, r.v + 16, r.v, r.v + 4, r.v + 8) != GOC_SUCCESS)
        return false;
      r.expected_condition = r.condition;
      for (int reg = 0; reg < r.output_regs; ++reg)
        for (unsigned lane = 0; lane < 32; ++lane)
          r.expected[128 * (lane / 16) + 16 * reg + lane % 16] = r.data[16 + reg][lane];
      const char *label = modified ? "neg/half/clamp" : "none";
      double scalar = measure(fn, semantics, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "exact", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd = measure(fn, semantics | GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd = measure(fn, semantics | GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_div_fixup(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_div_fixup_f16, goc_rdna4_v_div_fixup_f32,
                            goc_rdna4_v_div_fixup_f64};
  const char *names[] = {"v_div_fixup_f16", "v_div_fixup_f32", "v_div_fixup_f64"};
  const unsigned widths[] = {16, 32, 64}, fractions[] = {10, 23, 52}, biases[] = {15, 127, 1023};
  const uint64_t semantics = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  for (unsigned op = 0; op < 3; ++op)
    for (bool modified : {false, true}) {
      uint32_t mode = modified ? GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_ABS_B | GOC_ALU_NEG_C |
                                     GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP
                               : 0;
      if (op == 0 && modified)
        mode |= GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D;
      uint64_t fp = modified ? GOC_FP16_OVFL : 0, sign = UINT64_C(1) << (widths[op] - 1),
               one = uint64_t(biases[op]) << fractions[op],
               inf = uint64_t(2 * biases[op] + 1) << fractions[op];
      uint64_t values[] = {0,
                           sign,
                           one,
                           one | sign,
                           one + (UINT64_C(1) << fractions[op]),
                           inf - 1,
                           inf,
                           sign | inf,
                           1,
                           inf | 1,
                           one + 1,
                           one - 1,
                           one + (UINT64_C(1) << (fractions[op] - 1)),
                           sign | 1,
                           inf - (UINT64_C(1) << fractions[op]),
                           inf | sign | 3};
      Registers r;
      r.output_regs = op == 2 ? 2 : 1;
      for (unsigned lane = 0; lane < 32; ++lane) {
        uint64_t raw[3];
        for (unsigned operand = 0; operand < 3; ++operand) {
          raw[operand] = values[(lane * (2 * operand + 1) + 5 * operand) % 16];
          r.data[4 * operand][lane] = uint32_t(raw[operand]);
          r.data[4 * operand + 1][lane] = uint32_t(raw[operand] >> 32);
          if (op == 0) {
            unsigned shift = mode & (GOC_ALU_HIGH_A << operand) ? 16 : 0;
            r.data[4 * operand][lane] =
                uint32_t(raw[operand]) << shift | (uint32_t(raw[operand] ^ 0x8000) << (16 - shift));
          }
        }
        r.data[16][lane] = 0x12345678;
        r.data[17][lane] = 0xabcdef01;
        uint64_t output =
            goc_test::fixup_reference(widths[op], raw[0], raw[1], raw[2], mode, modified);
        if (op == 0) {
          unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
          output = (r.data[16][lane] & ~(65535u << shift)) | (uint32_t(output) << shift);
        }
        r.expected[128 * (lane / 16) + lane % 16] = uint32_t(output);
        if (op == 2)
          r.expected[128 * (lane / 16) + 16 + lane % 16] = uint32_t(output >> 32);
      }
      const char *label = modified ? (op == 0 ? "mixed/high" : "mixed/half/clamp") : "none";
      double scalar =
          measure(functions[op], GOC_CPU_BASELINE | semantics | fp, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "exact", label, "scalar", scalar, 1, 32, fp ? "fp16-ovfl" : "none");
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd =
            measure(functions[op], GOC_CPU_X86_64_V3 | semantics | fp, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v3", simd, scalar / simd, 32,
                     fp ? "fp16-ovfl" : "none");
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd =
            measure(functions[op], GOC_CPU_X86_64_V4 | semantics | fp, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v4", simd, scalar / simd, 32,
                     fp ? "fp16-ovfl" : "none");
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_cube(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {goc_rdna4_v_cubeid_f32, goc_rdna4_v_cubesc_f32, goc_rdna4_v_cubetc_f32,
                            goc_rdna4_v_cubema_f32};
  const char *names[] = {"v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32"};
  const uint32_t values[] = {0x3f800000, 0xc0000000, 0x40800000, 0x3f000000, 0xc0a00000, 0,
                             0x3e800000, 0xc0400000, 0x3f800001, 0xbf800000, 0x40400000, 0x40000000,
                             0xbe800000, 0x80000000, 0x3fc00000, 0xc0200000};
  const uint64_t semantics = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  for (unsigned op = 0; op < 4; ++op)
    for (bool modified : {false, true}) {
      uint32_t mode = modified ? GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_ABS_B | GOC_ALU_NEG_C |
                                     GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP
                               : 0;
      Registers r;
      r.output_regs = 1;
      for (unsigned lane = 0; lane < 32; ++lane) {
        r.data[0][lane] = values[lane % 16];
        r.data[4][lane] = values[(lane * 3 + 5) % 16];
        r.data[8][lane] = values[(lane * 7 + 9) % 16];
        r.expected[128 * (lane / 16) + lane % 16] =
            goc_test::cube_reference(op, r.data[0][lane], r.data[4][lane], r.data[8][lane], mode);
      }
      const char *label = modified ? "mixed/half/clamp" : "none";
      double scalar =
          measure(functions[op], GOC_CPU_BASELINE | semantics, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "exact", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (cpu >= GOC_CPU_X86_64_V3) {
        double simd =
            measure(functions[op], GOC_CPU_X86_64_V3 | semantics, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v3", simd, scalar / simd);
      }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
      if (cpu >= GOC_CPU_X86_64_V4) {
        double simd =
            measure(functions[op], GOC_CPU_X86_64_V4 | semantics, r, iterations, min_ms, mode);
        if (simd < 0)
          return false;
        print_result(names[op], "exact", label, "x86-64-v4", simd, scalar / simd);
      }
#endif
    }
  (void)cpu;
  return true;
}

bool benchmark_fp8_narrow(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_cvt_pk_fp8_f32);
  const Binary functions[] = {goc_rdna4_v_cvt_pk_fp8_f32, goc_rdna4_v_cvt_pk_bf8_f32,
                              goc_rdna4_v_cvt_sr_fp8_f32, goc_rdna4_v_cvt_sr_bf8_f32};
  const char *names[] = {"v_cvt_pk_fp8_f32", "v_cvt_pk_bf8_f32", "v_cvt_sr_fp8_f32",
                         "v_cvt_sr_bf8_f32"};
  const uint32_t values[] = {0,          0x3f800001, 0x3f880000, 0xbf980000, 0x43e80000, 0x47700000,
                             0x31000000, 0x3a800001, 0x7f800000, 0xff800000, 0x7fc12345, 0x80000000,
                             0x3dabcdef, 0xbe923456, 0x41212345, 0x37800001};
  for (unsigned op = 0; op < 4; ++op)
    for (int descriptor : {-1, 0, 5})
      for (bool modified : {false, true}) {
        uint64_t mode = modified ? goc_test::fp8_narrow_mode(op, op >= 2 ? 15 : 23) : 0;
        uint64_t fp = modified ? GOC_FP16_OVFL : 0;
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = values[lane % 16];
          r.data[4][lane] = op >= 2 ? 0x98765431u * lane : values[(lane + 5) % 16];
          r.data[16][lane] = 0xdecafbad;
          int source = int(lane);
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          uint32_t selected = source < 0 ? 0 : values[source % 16];
          r.expected[128 * (lane / 16) + lane % 16] = goc_test::fp8_narrow_result(
              op, selected, r.data[4][lane], r.data[16][lane], uint32_t(mode), modified);
        }
        auto call = [&](uint64_t f, uint64_t m, uint64_t i, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *) { return functions[op](f, m, i, d, a, b); };
        const char *label = modified ? (op >= 2 ? "NEG/ABS/byte3" : "mixed/high") : "none";
        if (descriptor >= 0)
          label = descriptor == 0 ? (modified ? "DPP8/modified" : "DPP8")
                                  : (modified ? "DPP16/modified" : "DPP16");
        double scalar = measure(call, GOC_CPU_BASELINE | fp, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(names[op], "loose", label, "scalar", scalar, 1, 32, fp ? "fp16-ovfl" : "none");
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(call, GOC_CPU_X86_64_V3 | fp, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(names[op], "loose", label, "x86-64-v3", simd, scalar / simd, 32,
                       fp ? "fp16-ovfl" : "none");
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(call, GOC_CPU_X86_64_V4 | fp, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(names[op], "loose", label, "x86-64-v4", simd, scalar / simd, 32,
                       fp ? "fp16-ovfl" : "none");
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_byte_pack(uint64_t cpu, int iterations, int min_ms) {
  const Wmma functions[] = {
      [](uint64_t f, uint64_t m, uint64_t i, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *,
         const uint32_t *const *) { return goc_rdna4_v_cvt_off_f32_i4(f, m, i, d, a); },
      goc_rdna4_v_cvt_pk_u8_f32};
  const char *names[] = {"v_cvt_off_f32_i4", "v_cvt_pk_u8_f32"};
  const uint32_t values[] = {0,          0x3f000000, 0x3fc00000, 0x40200000,
                             0x437f8000, 0xbf800000, 0x42ff0000, 0x3fffffff};
  for (unsigned op = 0; op < 2; ++op)
    for (int descriptor : {-1, 0, 5})
      for (bool modified : {false, true}) {
        if (!op && descriptor >= 0)
          continue;
        uint64_t mode = !modified ? 0
                        : op      ? GOC_ALU_NEG_A | GOC_ALU_CLAMP
                                  : GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint32_t raw = op ? values[lane % 8] : 0xdecaf000 | lane;
          r.data[0][lane] = raw;
          r.data[4][lane] = lane;
          r.data[8][lane] = 0x12345678u * lane;
          int source = int(lane);
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          uint32_t selected = descriptor < 0 ? raw : source < 0 ? 0 : values[source % 8];
          r.expected[128 * (lane / 16) + lane % 16] =
              op ? goc_test::byte_pack_reference(selected, r.data[4][lane], r.data[8][lane], mode)
                 : goc_test::nibble_offset_reference(raw, mode);
        }
        const char *label = descriptor < 0    ? (!modified ? "none"
                                                 : op      ? "NEG_A/clamp"
                                                           : "half/clamp")
                            : descriptor == 0 ? (modified ? "DPP8/NEG/clamp" : "DPP8")
                                              : (modified ? "DPP16/NEG/clamp" : "DPP16");
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

bool benchmark_integer_conversion(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_cvt_pk_i16_i32);
  const Binary functions[] = {
      [](uint64_t f, uint64_t m, uint64_t i, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *) { return goc_rdna4_v_cvt_i32_i16(f, m, i, d, a); },
      [](uint64_t f, uint64_t m, uint64_t i, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *) { return goc_rdna4_v_cvt_u32_u16(f, m, i, d, a); },
      goc_rdna4_v_cvt_pk_i16_i32, goc_rdna4_v_cvt_pk_u16_u32};
  const char *names[] = {"v_cvt_i32_i16", "v_cvt_u32_u16", "v_cvt_pk_i16_i32", "v_cvt_pk_u16_u32"};
  const uint32_t values[] = {0, 1, 0xffffffff, 0x7fff, 0x8000, 0x10000, 0xffff7fff, 0x80000000};
  for (unsigned op = 0; op < 4; ++op)
    for (int descriptor : {-1, 0, 5})
      for (unsigned select = 0; select < (op < 2 ? 2u : 1u); ++select) {
        uint64_t mode = select ? GOC_ALU_HIGH_A : 0;
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = values[lane % 8];
          r.data[4][lane] = values[(lane + 3) % 8];
          int source = int(lane);
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          uint32_t selected = source < 0 ? 0 : values[source % 8];
          r.expected[128 * (lane / 16) + lane % 16] =
              goc_test::integer_conversion_reference(op, selected, r.data[4][lane], uint32_t(mode));
        }
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *) {
          return functions[op](flags, mask, modifiers, d, a, b);
        };
        const char *label = descriptor < 0    ? (select ? "HIGH_A" : "none")
                            : descriptor == 0 ? (select ? "DPP8/HIGH_A" : "DPP8")
                                              : (select ? "DPP16/HIGH_A" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (op >= 2 && cpu >= GOC_CPU_X86_64_V3) {
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

bool benchmark_normalized(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_cvt_pk_norm_i16_f32);
  const Binary functions[] = {
      goc_rdna4_v_cvt_pk_norm_i16_f32,
      goc_rdna4_v_cvt_pk_norm_u16_f32,
      goc_rdna4_v_cvt_pk_norm_i16_f16,
      goc_rdna4_v_cvt_pk_norm_u16_f16,
      [](uint64_t f, uint64_t m, uint64_t i, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *) { return goc_rdna4_v_cvt_norm_i16_f16(f, m, i, d, a); },
      [](uint64_t f, uint64_t m, uint64_t i, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *) { return goc_rdna4_v_cvt_norm_u16_f16(f, m, i, d, a); }};
  const char *names[] = {"v_cvt_pk_norm_i16_f32", "v_cvt_pk_norm_u16_f32", "v_cvt_pk_norm_i16_f16",
                         "v_cvt_pk_norm_u16_f16", "v_cvt_norm_i16_f16",    "v_cvt_norm_u16_f16"};
  const uint32_t floats[] = {0,          0x3f000000, 0x3f7fffff, 0x3f800000,
                             0xbe800000, 0xbf800000, 0x3727c5ac, 0x3dcccccd};
  const uint32_t halves[] = {0, 0x3800, 0x3bff, 0x3c00, 0xb400, 0xbc00, 1, 0x2e66};
  for (unsigned op = 0; op < 6; ++op)
    for (int descriptor : {-1, 0, 5})
      for (bool modified : {false, true}) {
        uint64_t mode =
            modified ? goc_test::normalized_mode(op, goc_test::normalized_modes(op) - 1) : 0;
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] =
              op < 2 ? floats[lane % 8] : halves[lane % 8] | (halves[(lane + 3) % 8] << 16);
          r.data[4][lane] = op < 2 ? floats[(lane + 5) % 8]
                                   : halves[(lane + 5) % 8] | (halves[(lane + 1) % 8] << 16);
          r.data[16][lane] = 0xa5a5a5a5;
          int source = int(lane);
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          uint32_t selected = source < 0 ? 0
                              : op < 2   ? floats[source % 8]
                                         : halves[source % 8] | (halves[(source + 3) % 8] << 16);
          r.expected[128 * (lane / 16) + lane % 16] = goc_test::normalized_reference(
              op, selected, r.data[4][lane], 0xa5a5a5a5, uint32_t(mode));
        }
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *) {
          return functions[op](flags, mask, modifiers, d, a, b);
        };
        const char *label = !modified ? "none"
                            : op < 2  ? "ABS/NEG/clamp"
                            : op < 4  ? "ABS/NEG/hi/clamp"
                                      : "ABS/NEG/hi/OMOD/cl";
        if (descriptor >= 0)
          label = descriptor == 0 ? (modified ? "DPP8/modified" : "DPP8")
                                  : (modified ? "DPP16/modified" : "DPP16");
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

bool benchmark_packed_conversion(uint64_t cpu, int iterations, int min_ms) {
  using Binary = decltype(&goc_rdna4_v_cvt_pk_rtz_f16_f32);
  const Binary functions[] = {goc_rdna4_v_cvt_pk_rtz_f16_f32, goc_rdna4_v_cvt_pk_i16_f32,
                              goc_rdna4_v_cvt_pk_u16_f32};
  const char *names[] = {"v_cvt_pk_rtz_f16_f32", "v_cvt_pk_i16_f32", "v_cvt_pk_u16_f32"};
  const uint32_t values[] = {0x3fc00000, 0xc7000100, 0x477fff00, 0x7f7fffff,
                             0x387fffff, 0x33800000, 0xbf801fff, 0};
  for (unsigned op = 0; op < 3; ++op)
    for (int descriptor : {-1, 0, 5})
      for (bool modified : {false, true}) {
        uint64_t mode =
            modified
                ? goc_test::packed_conversion_mode(op, goc_test::packed_conversion_modes(op) - 1)
                : 0;
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          r.data[0][lane] = values[lane % 8];
          r.data[4][lane] = values[(lane + 3) % 8];
          int source = int(lane);
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          uint32_t selected = source < 0 ? 0 : values[source % 8];
          r.expected[128 * (lane / 16) + lane % 16] =
              goc_test::packed_conversion_reference(op, selected, r.data[4][lane], uint32_t(mode));
        }
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *) {
          return functions[op](flags, mask, modifiers, d, a, b);
        };
        const char *label = modified ? (op == 0 ? "ABS/NEG/cl/OMOD" : "ABS/NEG/clamp") : "none";
        if (descriptor >= 0)
          label = descriptor == 0 ? (modified ? "DPP8/modified" : "DPP8")
                                  : (modified ? "DPP16/modified" : "DPP16");
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

bool benchmark_fp8_conversion(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_cvt_f32_fp8);
  const Unary functions[] = {goc_rdna4_v_cvt_f32_fp8, goc_rdna4_v_cvt_f32_bf8,
                             goc_rdna4_v_cvt_pk_f32_fp8, goc_rdna4_v_cvt_pk_f32_bf8};
  const char *names[] = {"v_cvt_f32_fp8", "v_cvt_f32_bf8", "v_cvt_pk_f32_fp8", "v_cvt_pk_f32_bf8"};
  for (unsigned op = 0; op < 4; ++op)
    for (int descriptor : {-1, 0, 5})
      for (bool upper : {false, true}) {
        bool packed = op >= 2;
        if (packed && descriptor >= 0)
          continue;
        unsigned selection = upper ? (packed ? 1 : 3) : 0;
        uint64_t mode = goc_test::fp8_conversion_mode(packed, selection);
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
        Registers r;
        r.output_regs = packed ? 2 : 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint32_t raw = lane * 0x072f0311u;
          r.data[0][lane] = raw;
          int source = int(lane);
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          uint32_t selected = source < 0 ? 0 : uint32_t(source) * 0x072f0311u;
          for (int reg = 0; reg < r.output_regs; ++reg)
            r.expected[128 * (lane / 16) + 16 * reg + lane % 16] =
                goc_test::fp8_conversion_reference(
                    op & 1, uint8_t(selected >> (selection * (packed ? 16 : 8) + 8 * reg)));
        }
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *,
                            const uint32_t *const *) {
          return functions[op](flags, mask, modifiers, d, a);
        };
        const char *label = upper ? (packed ? "HIGH_A" : "byte:3") : "none";
        if (descriptor >= 0)
          label = descriptor == 0 ? (upper ? "DPP8/byte:3" : "DPP8")
                                  : (upper ? "DPP16/byte:3" : "DPP16");
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

bool benchmark_byte_conversion(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_cvt_f32_ubyte0);
  const Unary functions[] = {goc_rdna4_v_cvt_f32_ubyte0, goc_rdna4_v_cvt_f32_ubyte1,
                             goc_rdna4_v_cvt_f32_ubyte2, goc_rdna4_v_cvt_f32_ubyte3,
                             goc_rdna4_v_cvt_off_f32_i4};
  const char *names[] = {"v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2",
                         "v_cvt_f32_ubyte3", "v_cvt_off_f32_i4"};
  for (unsigned byte = 0; byte < 5; ++byte)
    for (int descriptor : {-1, 0, 5})
      for (unsigned variant : {0u, 7u}) {
        // The ordinary nibble conversion is covered by benchmark_byte_pack.
        if (byte == 4 && descriptor < 0)
          continue;
        uint64_t mode = goc_test::byte_conversion_mode(variant);
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
        Registers r;
        r.output_regs = 1;
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint32_t raw = lane * 0x09070301u;
          r.data[0][lane] = raw;
          int source = int(lane);
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          uint32_t selected = source < 0 ? 0 : uint32_t(source) * 0x09070301u;
          r.expected[128 * (lane / 16) + lane % 16] =
              byte < 4 ? goc_test::byte_conversion_reference(byte, selected, uint32_t(mode))
                       : goc_test::nibble_offset_reference(selected, uint32_t(mode));
        }
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *,
                            const uint32_t *const *) {
          return functions[byte](flags, mask, modifiers, d, a);
        };
        const char *label = descriptor < 0    ? (variant ? "half/clamp" : "none")
                            : descriptor == 0 ? (variant ? "DPP8/half/clamp" : "DPP8")
                                              : (variant ? "DPP16/half/clamp" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(names[byte], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(names[byte], "loose", label, "x86-64-v3", simd, scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(names[byte], "loose", label, "x86-64-v4", simd, scalar / simd);
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_conversion16(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_cvt_f16_i16);
  const Unary functions[] = {goc_rdna4_v_cvt_f16_i16, goc_rdna4_v_cvt_f16_u16,
                             goc_rdna4_v_cvt_i16_f16, goc_rdna4_v_cvt_u16_f16,
                             goc_rdna4_v_cvt_f16_f32, goc_rdna4_v_cvt_f32_f16};
  const uint32_t words[] = {0x3e00be00, 0x7bfffbff, 0x04000001, 0x7c00fc00,
                            0x3555b555, 0x00008000, 0x3c004000, 0xf8007800};
  const uint32_t float_inputs[] = {0x3fc00000, 0xbfc00000, 0x477ff000, 0xc7800000,
                                   0x387fffff, 0xb8800000, 0x3f000000, 0xbf000000};
  for (int op = 0; op < 6; ++op)
    for (int descriptor : {-1, 0, 5})
      for (bool modified : {false, true}) {
        uint64_t mode =
            modified ? goc_test::conversion16_mode(op, goc_test::conversion16_modes(op) - 1) : 0;
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
        Registers r;
        r.output_regs = 1;
        for (int lane = 0; lane < 32; ++lane) {
          uint32_t raw = op < 2    ? uint32_t(lane) * 134217757u
                         : op == 4 ? float_inputs[lane % 8]
                                   : words[lane % 8];
          r.data[0][lane] = raw;
          r.data[16][lane] = 0xa5a55a5a;
          int source = lane;
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          uint32_t input = source < 0 ? 0
                           : op < 2   ? uint32_t(source) * 134217757u
                           : op == 4  ? float_inputs[source % 8]
                                      : words[source % 8];
          r.expected[128 * (lane / 16) + lane % 16] =
              goc_test::conversion16_reference(op, input, 0xa5a55a5a, uint32_t(mode), true);
        }
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *,
                            const uint32_t *const *) {
          return functions[op](flags, mask, modifiers, d, a);
        };
        const char *label = !modified ? "none"
                            : op < 2  ? "hi/half/clamp"
                            : op < 4  ? "ABS/NEG/hi/OMOD/cl"
                                      : "ABS/NEG/hi/half/cl";
        if (descriptor >= 0)
          label = descriptor == 0 ? (modified ? "DPP8/modifiers" : "DPP8")
                                  : (modified ? "DPP16/modifiers" : "DPP16");
        // Exercise the finite-overflow path for all FP16 conversion timings.
        double scalar = measure(fn, GOC_CPU_BASELINE | GOC_FP16_OVFL, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::conversion16_names[op], "loose", label, "scalar", scalar, 1, 32,
                     "fp16-ovfl");
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3 | GOC_FP16_OVFL, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::conversion16_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd, 32, "fp16-ovfl");
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4 | GOC_FP16_OVFL, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::conversion16_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd, 32, "fp16-ovfl");
        }
#endif
      }
  (void)cpu;
  return true;
}

bool benchmark_conversion64(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_cvt_f64_i32);
  const Unary functions[] = {goc_rdna4_v_cvt_f64_i32, goc_rdna4_v_cvt_f64_u32,
                             goc_rdna4_v_cvt_i32_f64, goc_rdna4_v_cvt_u32_f64,
                             goc_rdna4_v_cvt_f64_f32, goc_rdna4_v_cvt_f32_f64};
  const char *names[] = {"v_cvt_f64_i32", "v_cvt_f64_u32", "v_cvt_i32_f64",
                         "v_cvt_u32_f64", "v_cvt_f64_f32", "v_cvt_f32_f64"};
  const uint64_t double_inputs[] = {0x3ff8000000000000, 0xbff8000000000000, 0x41dfffffffffffff,
                                    0xc1e0000000100000, 0x41efffffffffffff, 0x41f0000000000000,
                                    0x3fe0000000000000, 0xbfe0000000000000};
  const uint32_t float_inputs[] = {0x3fc00000, 0xbfc00000, 0x4effffff, 0xcf000001,
                                   0x4f7fffff, 0x4f800000, 0x3f000000, 0xbf000000};
  for (int op = 0; op < 6; ++op)
    for (bool modified : {false, true}) {
      uint32_t mode =
          modified ? goc_test::conversion64_mode(op, goc_test::conversion64_modes(op) - 1) : 0;
      Registers r;
      r.output_regs = goc_test::conversion64_wide_output(op) ? 2 : 1;
      for (int lane = 0; lane < 32; ++lane) {
        uint64_t raw = op < 2    ? uint32_t(lane) * 134217757u
                       : op == 4 ? float_inputs[lane % 8]
                                 : double_inputs[lane % 8];
        r.data[0][lane] = uint32_t(raw);
        r.data[1][lane] = uint32_t(raw >> 32);
        uint64_t expected = goc_test::conversion64_reference(op, raw, mode);
        for (int reg = 0; reg < r.output_regs; ++reg)
          r.expected[128 * (lane / 16) + 16 * reg + lane % 16] = uint32_t(expected >> (32 * reg));
      }
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *,
                          const uint32_t *const *) {
        return functions[op](flags, mask, modifiers, d, a);
      };
      const char *label = !modified ? "none"
                          : op < 2  ? "half/clamp"
                          : op < 4  ? "ABS/NEG/OMOD/clamp"
                                    : "ABS/NEG/half/clamp";
      double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
      if (scalar < 0)
        return false;
      print_result(names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
      if (op != 2 && cpu >= GOC_CPU_X86_64_V3) {
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

bool benchmark_conversion32(uint64_t cpu, int iterations, int min_ms) {
  using Unary = decltype(&goc_rdna4_v_cvt_f32_i32);
  const Unary functions[] = {goc_rdna4_v_cvt_f32_i32,         goc_rdna4_v_cvt_f32_u32,
                             goc_rdna4_v_cvt_i32_f32,         goc_rdna4_v_cvt_u32_f32,
                             goc_rdna4_v_cvt_nearest_i32_f32, goc_rdna4_v_cvt_floor_i32_f32};
  const uint32_t float_inputs[] = {0x3fc00000, 0xbfc00000, 0x4effffff, 0xcf000001,
                                   0x4f7fffff, 0x4f800000, 0x7fc12345, 0xff800000};
  for (int op = 0; op < 6; ++op)
    for (int descriptor : {-1, 0, 5})
      for (bool modified : {false, true}) {
        uint64_t mode =
            modified ? goc_test::conversion32_mode(op, goc_test::conversion32_modes(op) - 1) : 0;
        if (descriptor >= 0)
          mode |= goc_test::dpp_modes[descriptor];
        Registers r;
        r.output_regs = 1;
        for (int lane = 0; lane < 32; ++lane) {
          uint32_t raw = op < 2 ? uint32_t(lane) * 134217757u : float_inputs[lane % 8];
          r.data[0][lane] = raw;
          int source = lane;
          if (descriptor >= 0)
            goc_test::dpp_source(mode, UINT32_MAX, lane, source);
          uint32_t input = source < 0 ? 0
                           : op < 2   ? uint32_t(source) * 134217757u
                                      : float_inputs[source % 8];
          r.expected[128 * (lane / 16) + lane % 16] =
              goc_test::conversion32_reference(op, input, uint32_t(mode));
        }
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *,
                            const uint32_t *const *) {
          return functions[op](flags, mask, modifiers, d, a);
        };
        const char *label = !modified ? "none"
                            : op < 2  ? "half/clamp"
                            : op < 4  ? "ABS/NEG/OMOD/clamp"
                                      : "ABS/NEG/clamp";
        if (descriptor >= 0)
          label = descriptor == 0 ? (modified ? "DPP8/modifiers" : "DPP8")
                                  : (modified ? "DPP16/modifiers" : "DPP16");
        double scalar = measure(fn, GOC_CPU_BASELINE, r, iterations, min_ms, mode);
        if (scalar < 0)
          return false;
        print_result(goc_test::conversion32_names[op], "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
        if (op != 2 && op != 3 && cpu >= GOC_CPU_X86_64_V3) {
          double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::conversion32_names[op], "loose", label, "x86-64-v3", simd,
                       scalar / simd);
        }
#endif
#if defined(GOC_BENCH_HAVE_X86_64_V4)
        if (cpu >= GOC_CPU_X86_64_V4) {
          double simd = measure(fn, GOC_CPU_X86_64_V4, r, iterations, min_ms, mode);
          if (simd < 0)
            return false;
          print_result(goc_test::conversion32_names[op], "loose", label, "x86-64-v4", simd,
                       scalar / simd);
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
  const char *names[] = {"v_trunc_f32", "v_ceil_f32",  "v_rndne_f32",     "v_floor_f32",
                         "v_sqrt_f32",  "v_rcp_f32",   "v_rsq_f32",       "v_exp_f32",
                         "v_log_f32",   "v_fract_f32", "v_frexp_mant_f32"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
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
  const char *names[] = {"v_add_f32",     "v_sub_f32",     "v_subrev_f32",
                         "v_mul_f32",     "v_min_num_f32", "v_max_num_f32",
                         "v_minimum_f32", "v_maximum_f32", "v_mul_dx9_zero_f32"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
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
  const char *names[] = {"v_mad_u32_u16", "v_mad_i32_i16", "v_mad_u32_u24", "v_mad_i32_i24"};
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
  const char *names[] = {"v_mad_u16",  "v_mad_i16",  "v_min3_u16", "v_min3_i16",
                         "v_max3_u16", "v_max3_i16", "v_med3_u16", "v_med3_i16"};
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
      const char *name = sign ? "v_pk_mad_i16" : "v_pk_mad_u16";
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
  const char *names[] = {"v_add_nc_i16", "v_sub_nc_i16",  "v_add_nc_u16",  "v_sub_nc_u16",
                         "v_min_i16",    "v_max_i16",     "v_min_u16",     "v_max_u16",
                         "v_mul_lo_u16", "v_lshlrev_b16", "v_lshrrev_b16", "v_ashrrev_i16"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
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
  const char *names[] = {"v_pk_add_i16",     "v_pk_sub_i16",     "v_pk_add_u16",
                         "v_pk_sub_u16",     "v_pk_min_i16",     "v_pk_max_i16",
                         "v_pk_min_u16",     "v_pk_max_u16",     "v_pk_mul_lo_u16",
                         "v_pk_lshlrev_b16", "v_pk_lshrrev_b16", "v_pk_ashrrev_i16"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
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
  const char *names[] = {"v_pk_add_f16",     "v_pk_mul_f16",     "v_pk_min_num_f16",
                         "v_pk_max_num_f16", "v_pk_minimum_f16", "v_pk_maximum_f16"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
        if (accumulate) {
          // Include the fixed-input accumulator reset in all FMAC timings.
          std::memcpy(d[0], c[0], 32 * sizeof(uint32_t));
          return goc_rdna4_v_pk_fmac_f16(flags, mask, modifiers, d, a, b);
        }
        return goc_rdna4_v_pk_fma_f16(flags, mask, modifiers, d, a, b, c);
      };
      const char *name = accumulate ? "v_pk_fmac_f16" : "v_pk_fma_f16";
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
        const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *) {
          if (half)
            return multiply
                       ? goc_rdna4_v_fmamk_f16(flags, mask, modifiers, d, a, uint16_t(literal), b)
                       : goc_rdna4_v_fmaak_f16(flags, mask, modifiers, d, a, b, uint16_t(literal));
          return multiply ? goc_rdna4_v_fmamk_f32(flags, mask, modifiers, d, a, literal, b)
                          : goc_rdna4_v_fmaak_f32(flags, mask, modifiers, d, a, b, literal);
        };
        const char *name = half ? (multiply ? "v_fmamk_f16" : "v_fmaak_f16")
                                : (multiply ? "v_fmamk_f32" : "v_fmaak_f32");
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
        // Keep the accumulator input fixed, avoiding drift during repeated FMAC.
        // This reset is included in timings for every CPU path.
        std::memcpy(d[0], c[0], 32 * sizeof(uint32_t));
        return operation(flags, mask, modifiers, d, a, b);
      };
      const char *name = half ? "v_fmac_f16" : "v_fmac_f32";
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
    print_result("v_fma_f16", "loose", label, "scalar", scalar, 1);
#if defined(GOC_BENCH_HAVE_X86_64_V3)
    if (cpu >= GOC_CPU_X86_64_V3) {
      double simd = measure(fn, GOC_CPU_X86_64_V3, r, iterations, min_ms, mode);
      if (simd < 0)
        return false;
      print_result("v_fma_f16", "loose", label, "x86-64-v3", simd, scalar / simd);
    }
#endif
    double exact =
        measure(fn, GOC_CPU_BASELINE | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, r,
                iterations, min_ms, mode);
    if (exact < 0)
      return false;
    print_result("v_fma_f16", "exact", label, "scalar", exact, 1);
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
  const char *names[] = {"v_min3_num_f16",       "v_max3_num_f16",       "v_minmax_num_f16",
                         "v_maxmin_num_f16",     "v_minimum3_f16",       "v_maximum3_f16",
                         "v_minimummaximum_f16", "v_maximumminimum_f16", "v_med3_num_f16"};
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
  const char *names[] = {"v_min3_num_f32",       "v_max3_num_f32",       "v_minmax_num_f32",
                         "v_maxmin_num_f32",     "v_minimum3_f32",       "v_maximum3_f32",
                         "v_minimummaximum_f32", "v_maximumminimum_f32", "v_med3_num_f32"};
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
  const char *names[] = {"v_add_f64",     "v_mul_f64",     "v_fma_f64",    "v_min_num_f64",
                         "v_max_num_f64", "v_minimum_f64", "v_maximum_f64"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
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
  const char *names[] = {"v_trunc_f64", "v_ceil_f64",  "v_rndne_f64",
                         "v_floor_f64", "v_fract_f64", "v_sqrt_f64",
                         "v_rcp_f64",   "v_rsq_f64",   "v_frexp_mant_f64"};
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
      const auto fn = [&](uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
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
      const char *name = bf16 ? "v_dot2_bf16_bf16" : "v_dot2_f16_f16";
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
  const char *names[] = {"v_wmma_f32_16x16x16_fp8_fp8", "v_wmma_f32_16x16x16_fp8_bf8",
                         "v_wmma_f32_16x16x16_bf8_fp8", "v_wmma_f32_16x16x16_bf8_bf8"};
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
  const char *names[] = {"v_dot4_f32_fp8_fp8", "v_dot4_f32_fp8_bf8", "v_dot4_f32_bf8_fp8",
                         "v_dot4_f32_bf8_bf8"};
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
      const char *name = bf16 ? "v_dot2_f32_bf16" : "v_dot2_f32_f16";
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
  const char *names[] = {"v_dot4_i32_iu8", "v_dot4_u32_u8", "v_dot8_i32_iu4", "v_dot8_u32_u4"};
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
  bool have_iterations = false;
  bool have_min_ms = false;
  for (int arg = 1; arg < argc; ++arg) {
    if (std::strcmp(argv[arg], "--csv") == 0 && !csv_output) {
      csv_output = true;
    } else if (std::strcmp(argv[arg], "--min-ms") == 0 && !have_min_ms && arg + 1 < argc &&
               nonnegative_integer(argv[arg + 1], min_ms)) {
      have_min_ms = true;
      ++arg;
    } else if (!have_iterations && nonnegative_integer(argv[arg], iterations) && iterations > 0) {
      have_iterations = true;
    } else {
      std::fprintf(
          stderr,
          "Usage: %s [--csv] [--min-ms nonnegative milliseconds] [positive initial iterations]\n",
          argv[0]);
      return 2;
    }
  }

  uint64_t cpu = goc_init_cpu_flags();
  FILE *messages = csv_output ? stderr : stdout;
  std::fprintf(messages, "RDNA4 instruction benchmarks; CPU flags 0x%llx\n",
               static_cast<unsigned long long>(cpu));
  std::fprintf(messages, "Median of 7 samples, each at least %d ms, after warmup.\n", min_ms);
  std::fprintf(messages, "Start at %d calls; double until the minimum duration is reached.\n",
               iterations);
  std::fprintf(messages,
               "Fixed inputs, full EXEC, separate C/D, hot buffers; all outputs checked.\n");
  std::fprintf(messages,
               "Timings include public API dispatch, input conversions and output stores.\n");
  std::fprintf(
      messages,
      "FP rows: loose speedups, exact scalar separately. Integer WMMA rows: exact. Speedups "
      "compare "
      "matching instruction-flags settings.\n");
  std::fprintf(messages, "mixed = NEG_HI_A | NEG_LO_B | ABS_C | NEG_C.\n");
  print_columns("Instruction", "Semantics", "Instruction flags", "CPU path", "ns/wave", "Speedup",
                "Wave", "FP state");
  if (!benchmark_rcp_iflag(cpu, iterations, min_ms) ||
      !benchmark_scalar_sign_extend(iterations, min_ms) ||
      !benchmark_scalar_pack(iterations, min_ms) || !benchmark_scalar_compare(iterations, min_ms) ||
      !benchmark_scalar_convert(iterations, min_ms) ||
      !benchmark_scalar_round(iterations, min_ms) || !benchmark_scalar_fma(iterations, min_ms) ||
      !benchmark_scalar_fp(iterations, min_ms) || !benchmark_scalar_field(iterations, min_ms) ||
      !benchmark_scalar_bits(iterations, min_ms) || !benchmark_scalar_integer(iterations, min_ms) ||
      !benchmark_pseudo_scalar(iterations, min_ms) ||
      !benchmark_float_compare(cpu, iterations, min_ms) ||
      !benchmark_integer_compare(cpu, iterations, min_ms) ||
      !benchmark_class(cpu, iterations, min_ms) || !benchmark_interp16(cpu, iterations, min_ms) ||
      !benchmark_interp32(cpu, iterations, min_ms) ||
      !benchmark_dpp_arithmetic(cpu, iterations, min_ms) ||
      !benchmark_dpp_unary(cpu, iterations, min_ms) ||
      !benchmark_dpp_integer(cpu, iterations, min_ms) ||
      !benchmark_dpp_integer_minmax(cpu, iterations, min_ms) ||
      !benchmark_dpp_integer_ternary(cpu, iterations, min_ms) ||
      !benchmark_dpp_bitfield(cpu, iterations, min_ms) ||
      !benchmark_dpp_integer_add(cpu, iterations, min_ms) ||
      !benchmark_dpp_integer_mul(cpu, iterations, min_ms) ||
      !benchmark_dpp_integer_mad(cpu, iterations, min_ms) ||
      !benchmark_dpp_boolean16(cpu, iterations, min_ms) ||
      !benchmark_dpp_integer16(cpu, iterations, min_ms) ||
      !benchmark_dpp_half_binary(cpu, iterations, min_ms) ||
      !benchmark_dpp_half_unary(cpu, iterations, min_ms) ||
      !benchmark_dpp_half_minmax(cpu, iterations, min_ms) ||
      !benchmark_dpp_half_fma(cpu, iterations, min_ms) ||
      !benchmark_dpp_integer16_ternary(cpu, iterations, min_ms) ||
      !benchmark_dpp16(cpu, iterations, min_ms) || !benchmark_dpp8(cpu, iterations, min_ms) ||
      !benchmark_permlane(cpu, iterations, min_ms) || !benchmark_cndmask(cpu, iterations, min_ms) ||
      !benchmark_trig_preop(cpu, iterations, min_ms) ||
      !benchmark_mullit(cpu, iterations, min_ms) || !benchmark_pack(cpu, iterations, min_ms) ||
      !benchmark_swmmac_integer(cpu, iterations, min_ms) ||
      !benchmark_swmmac8(cpu, iterations, min_ms) || !benchmark_swmmac16(cpu, iterations, min_ms) ||
      !benchmark_mad64(cpu, iterations, min_ms) || !benchmark_carry(cpu, iterations, min_ms) ||
      !benchmark_div_fmas(cpu, iterations, min_ms) ||
      !benchmark_div_scale(cpu, iterations, min_ms) ||
      !benchmark_div_fixup(cpu, iterations, min_ms) || !benchmark_cube(cpu, iterations, min_ms) ||
      !benchmark_fp8_narrow(cpu, iterations, min_ms) ||
      !benchmark_byte_pack(cpu, iterations, min_ms) ||
      !benchmark_integer_conversion(cpu, iterations, min_ms) ||
      !benchmark_normalized(cpu, iterations, min_ms) ||
      !benchmark_packed_conversion(cpu, iterations, min_ms) ||
      !benchmark_fp8_conversion(cpu, iterations, min_ms) ||
      !benchmark_byte_conversion(cpu, iterations, min_ms) ||
      !benchmark_conversion16(cpu, iterations, min_ms) ||
      !benchmark_conversion64(cpu, iterations, min_ms) ||
      !benchmark_conversion32(cpu, iterations, min_ms) ||
      !benchmark_integer_mad(cpu, iterations, min_ms) ||
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

# GoC: GPU arithmetic on a CPU

GoC is a synchronous C library for GPU emulators. Each call executes one GPU
instruction for an entire wave on the calling CPU thread. It implements RDNA3
and RDNA4 arithmetic, conversions, comparisons, matrix operations, and register
routing, with portable code and x86 SIMD paths.

Choose fast, approximate **loose** semantics or supported **empirically bit-exact**
semantics. Coverage varies by instruction and mode; GoC is not a GPU runtime or
instruction decoder. See [PLAN.md](https://github.com/bjacob/goc/blob/main/PLAN.md)
for design, contracts, limitations, and testing policy.

## Build and run

Requires CMake 3.20+, a C++17 compiler, and Ninja. Public headers support C99
and C++17. Tests use a system GTest or fetch it automatically.

```sh
cmake -S . -B ../goc-build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ../goc-build --parallel
ctest --test-dir ../goc-build --parallel --output-on-failure
../goc-build/tests/cpuinfo
../goc-build/tests/goc_benchmark --min-ms 10
```

Pass `--csv` to the benchmark for CSV on stdout. Build directories must be
outside the source tree. To embed GoC with CMake, use `add_subdirectory` and link
the `goc` target; it supplies the include path and a static, position-independent
library with hidden symbols. Set `BUILD_TESTING=OFF` to omit tests.

## API example

Include [`goc/goc.h`](https://github.com/bjacob/goc/blob/main/include/goc/goc.h).
This C/C++ example invokes RDNA4 Wave32 FP16 WMMA with FP32 accumulation on
zero-initialized operands:

```c
#include "goc/goc.h"

int main(void) {
  uint32_t a_words[4][32] = {0}, b_words[4][32] = {0};
  uint32_t c_words[8][32] = {0}, d_words[8][32] = {0};
  const uint32_t *a[4], *b[4], *c[8];
  uint32_t *d[8];
  for (int i = 0; i < 4; ++i) {
    a[i] = a_words[i];
    b[i] = b_words[i];
  }
  for (int i = 0; i < 8; ++i) {
    c[i] = c_words[i];
    d[i] = d_words[i];
  }
  return goc_v_wmma_f32_16x16x16_f16_rdna4(
      goc_init_cpu_flags() | GOC_SEMANTICS_LOOSE, 0, d, a, b, c);
}
```

Each pointer represents one VGPR containing 32 lane words in the GPU's register
layout. This instruction ignores EXEC and therefore takes no execution mask.
Request `GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT` instead of
`GOC_SEMANTICS_LOOSE` to require its exact path. Calls return zero on success
and leave outputs unchanged on error.

See the [instruction declarations](https://github.com/bjacob/goc/blob/main/include/goc/detail/goc_instructions.h)
and [WMMA tests](https://github.com/bjacob/goc/blob/main/tests/wmma_test.cpp)
for layouts, modifiers, and nonzero examples.

Licensed under the [MIT license](https://github.com/bjacob/goc/blob/main/LICENSE).

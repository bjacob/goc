# GoC: GPU-on-CPU: CPU implementations of GPU arithmetic instructions.

## The API

### General shape

This library shall expose a C API where most entry points map 1:1 to GPU arithmetic
instructions. The primary intended use case is to implement a GPU emulator running
on a CPU. Each entry point performs on CPU the same arithmetic computation
as the corresponding GPU instruction does.

Each entry point performs the work of an entire GPU wave, not just a single GPU
thread. The threading model is that an entire GPU wave (not a single GPU thread) is
mapped to one CPU thread.

The library is not threaded at all: it just does its work
on the CPU thread that it's called on; it doesn't use any synchronization primitive, atomics,
barriers --- nothing. Each function just does its work, synchronously.

That means that any "worker thread" or "asynchronous" or "deferred" computation
ideas are things for the caller to layer on top of this library. They are not
in scope for this library.

### C vs C++

The API is C, but both the internal implementation and the users are allowed to
be either C or C++. On both sides, C++ is expected to be common.

This means that the API header file(s) must compile as both C99 and C++17.
The usual `#ifdef __cplusplus`, `extern "C"` idiom will be necessary. As a result,
the compile will generate a C ABI, not a C++ ABI, regardless of whether it is
compiling as C or C++.

In the API, try to avoid C-type contortions to represent GPU data types.
For example, representing a `bf16` value might be possible using a non-standard
compiler extension type such as `__bf16`. Don't bother with that in the API.
If we had to pass a `bf16` uniform value, it could be passed as a `uint16_t`.
Generally, fall back on sized unsigned integer types to pass raw bits if there
isn't a standard C type / `stdint.h` type doing exactly what is needed. A `float32`
value may be passed as C `float`. Generally, we are willing to entrench the
assumption that `float` is IEEE-754 single precision, and `double` is IEEE-754
double precision. For integer types, see [Data types](#data-types) below.

Public named integer constants shall use `static const` with an explicit integer
type, rather than preprocessor macros. C99 constant-expression support is not
required for these names. If that becomes necessary, consider moving to C23.

### Anatomy of an entry point

GoC entry point names follow the pattern

`goc_<mnemonic>`

Where:

* Shared instruction semantics use the same entry point across architectures.
  If a newer architecture changes the semantics of an existing mnemonic, its
  variant gets an architecture suffix, such as `_rdna4`. No current entry point
  needs an architecture suffix. Wave32 is the default; dedicated Wave64 variants
  append `_wave64` after the mnemonic (and after any architecture suffix).
* `<mnemonic>` is the instruction mnemonic like `v_wmma_f32_16x16x16_f16`. We follow
  instruction mnemonics, not intrinsic names, because the API model here really
  is much closer to the instructions than it is to the intrinsics: see how
  instruction VGPR operands map 1:1 to function parameters.

Example:

`goc_v_wmma_f32_16x16x16_f16`

The API shall provide an umbrella C header, [`goc/goc.h`](https://github.com/bjacob/goc/blob/main/include/goc/goc.h), directly including
[`goc/detail/goc_common.h`](https://github.com/bjacob/goc/blob/main/include/goc/detail/goc_common.h) for common flags, error codes and CPU initialization, and
[`goc/detail/goc_instructions.h`](https://github.com/bjacob/goc/blob/main/include/goc/detail/goc_instructions.h) for instruction declarations and modifiers.

API users, including implementations and tests, shall include [`goc/goc.h`](https://github.com/bjacob/goc/blob/main/include/goc/goc.h). It is the
only header directly under [`include/goc/`](https://github.com/bjacob/goc/tree/main/include/goc); component headers live under
[`include/goc/detail/`](https://github.com/bjacob/goc/tree/main/include/goc/detail). Within [`include/`](https://github.com/bjacob/goc/tree/main/include), quoted includes are relative to the
containing header's own directory. GoC's CMake targets expose [`include/`](https://github.com/bjacob/goc/tree/main/include), but the
headers must also work when consumers reach them through other include paths.
Internal and standard-library dependencies shall be included directly. Use minimal
standard `#include` directives (`stdint.h`, and maybe a few more as needed).

Most entry points shall correspond 1:1 to a supported GPU instruction
following the above pattern. Each such function shall take the following function parameters:

* `uint64_t flags`, a bit-field described below.
* `uint32_t exec_mask` for Wave32 and `uint64_t exec_mask` for Wave64, with the
  same semantics as in the GPU architectures, each bit enabling a lane in the destination VGPRs.
  Omit this parameter for instructions that ignore EXEC: scalar, pseudo-scalar,
  WMMA and SWMMAC. Matrix instructions read and write all lanes, even when
  architectural EXEC is zero.
* Scalars/literals can be passed by value as function parameters of suitable C type.
  - Default to unsigned integer C types, meaning "raw bits", unless a standard C
    type exists with exactly the right semantics, e.g. a signed integer or
    standard floating-point type as appropriate.
  - Operands shall be enumerated in the same order as in the assembly syntax.
  - Instructions that have mode/flag bits, can pass them here, typically as a
    `instruction_flags` parameter before other instruction operands, a `uint64_t`.
    For example, MFMA instructions pass `CBSZ`, `ABID`, and related modes this way.
    Do combine all such flags into a single unsigned integer, rather than passing
    multiple short integers.
* Implicit scalar inputs are passed by value after the explicit instruction
  operands. For example, `v_div_fmas_f32` and `v_div_fmas_f64` take a `uint32_t`
  condition mask representing wave32 `VCC` after their three VGPR inputs.
* Scalar register outputs use pointers to their raw unsigned integer values, in
  assembly operand order. Wave32 condition outputs use `uint32_t *`. Their EXEC
  behavior follows the instruction: `v_div_scale_f32` and `v_div_scale_f64` clear
  inactive condition bits, including writing zero for zero EXEC. Carry/borrow
  instructions follow the same rule for their scalar output masks. Their scalar
  output pointer must therefore remain writable even when VGPR pointers may be null.
* For each VGPR operand of the GPU instruction, a pointer to the array of pointers
  representing the VGPRs backing that operand: `const uint32_t *const *` for inputs,
  and `uint32_t *const *` for outputs. Input pointers permit reads only; writes use
  output pointers. This `const` qualification does not prohibit input/output aliasing.
  - Operands shall be enumerated in the same order as in the assembly syntax.
  - Each VGPR is expected to be backed by a contiguous array of `uint32_t` words,
    one per lane. Thus, each VGPR is represented by one `const uint32_t *` input
    pointer or `uint32_t *` output pointer. An array of such pointers can represent
    an arbitrary multi-VGPR operand, without
    requiring the VGPRs to be adjacent to one another in memory, only requiring
    each of them to be internally contiguous (no stride between lanes).
  - These pointers are not `restrict`. Aliasing is explicitly supported, corresponding
    to overlapping operands. results behave as though all required source values were read before any destination writes; masked-off destination lanes remain unchanged. It is legal for multiple VGPR-data pointer to point to the same VGPR-backing address. However, overlap must be exact/full. If two VGPR-data pointers are different then they
    are assumed to not overlap. That means that at VGPR granularity, partial-overlap is not supported.
    Multi-VGPR operands may have partial overlap in the sense of sharing some of their VGPRs.

Each entry point shall have return type `int`. A return value of 0 means success.
Any nonzero return value is an error code, see the "Error codes" section below.
On error, the destination register values are left unchanged. That means that
validation must happen before computation.

Example:

```c
int goc_v_wmma_f32_16x16x16_f16(
  uint64_t flags,
  uint64_t instruction_flags,  // NEG and NEG_HI bits go here.
  uint32_t *const * vgpr_d,
  const uint32_t *const * vgpr_a,
  const uint32_t *const * vgpr_b,
  const uint32_t *const * vgpr_c
);
```

Notes:

* What about Wave64 variants on Wave32-native architecture?
  - Typical lane-wise instructions can just use two calls to the wave32 function.
  - For instructions like WMMA or lane-index-dependent `MBCNT`, we may have a
    separate dedicated wave64 entry point with a `_wave64` suffix, e.g.
    `goc_v_wmma_f32_16x16x16_f16_wave64`.
* For instructions that obey EXEC, why make `exec_mask` part of GoC instead of letting the caller handle it?
  - For the caller to handle it correctly w.r.t. input-output aliasing, they
    would need to save destination registers before calling GoC.
  - x86-64 masked stores are exactly the CPU ISA feature making this simpler
    and more efficient to handle inside GoC.

* Improving performance on sparse `exec_mask` values is an explicit non-goal.
  Performance should be independent of the mask: compute the full result and
  use masked destination stores. Do not add mask-density checks, active-lane
  iteration, or specialized sparse-mask paths; their code size and runtime
  overhead are unwanted. This is a performance design goal, not a constant-time
  guarantee. An early return for a zero effective mask is explicitly allowed;
  it is not a reason to introduce further mask-dependent optimizations.

### The `flags` bit-field

This `uint64_t` bit-field shall combine:

* CPU flags, indicating a CPU feature level to rely on. Most users will pass the
  return value of the `goc_init_cpu_flags()` function described below.
* Semantics flags. This controls the accuracy requirements on the output.
  This allows switching between a default loose mode (no particular accuracy requirement,
  any algebraically-sound implementation is accepted), and other bit-exact modes,
  plural because bit-exact could still mean different things.
* GPU floating-point state/mode to emulate. Things like rounding mode,
  denormals handling etc.

One special entry point, `goc_init_cpu_flags`, shall have the prototype

`uint64_t goc_init_cpu_flags(void);`

and it shall perform CPU feature detection, returning a `uint64_t` whose bits shall
encode the availability of certain CPU features that GoC functions may rely on.

Naturally, the 3 kinds of flags occupy disjoint bit ranges, and `goc_init_cpu_flags`
leaves of the others zero. Callers OR-combine that with their own values of the
other flags.

### Error codes

Entry points return a nonzero integer on errors. We will define some symbolic
constants for such error codes. The implementation work is free to decide, as it goes, what error codes to create, how to name them symbolically and how to assign them nonzero integer values.

## The implementation

The implementation may use C or C++. We expect that C++ could be useful to bring
flavors of generic programming to deal with repetitive implementation, such as
when multiple instructions belong naturally to a type- or size-parametrized
family.

### Data types

For integer types, we follow the philosophy in
https://google.github.io/styleguide/cppguide.html#Integer_Types

Try to stick to standard C with the data types provided by `stdint.h` and `stddef.h`
unless a specific need arises, then discuss.

Prefer sized integer types from `stdint.h` (e.g. `int32_t`, `uint64_t`). Of the
builtin-keyword standard integer types (`short`, `int`, `long`, `long long`), only use
`int`, and only use it to mean "must be at least 32-bit, don't care about the
exact size".

For sizes and indices, use signed integers, not `unsigned`.

Use `unsigned` to convey either that we are looking at raw bits, or that wrapping
modulo 2^N is explicitly intended.

### Source tree

| Path | Purpose |
| --- | --- |
| `cmake/` | Any shared CMake files (planned; directory not yet present). |
| [`CMakeLists.txt`](https://github.com/bjacob/goc/blob/main/CMakeLists.txt) | Root CMake configuration. |
| [`include/goc/goc.h`](https://github.com/bjacob/goc/blob/main/include/goc/goc.h) | Umbrella API header. |
| [`include/goc/detail/goc_common.h`](https://github.com/bjacob/goc/blob/main/include/goc/detail/goc_common.h) | Common API definitions. |
| [`include/goc/detail/goc_instructions.h`](https://github.com/bjacob/goc/blob/main/include/goc/detail/goc_instructions.h) | Shared instruction API. |
| [`src/`](https://github.com/bjacob/goc/tree/main/src) | Implementation; architecture-agnostic files directly here. |
| [`src/CMakeLists.txt`](https://github.com/bjacob/goc/blob/main/src/CMakeLists.txt) | Library build configuration. |
| [`src/x86_64/`](https://github.com/bjacob/goc/tree/main/src/x86_64) | x86-64-specific code paths (AVX etc). No further subdirectories for now. |
| [`src/x86_64/CMakeLists.txt`](https://github.com/bjacob/goc/blob/main/src/x86_64/CMakeLists.txt) | x86-64 code-path build configuration. |
| [`tests/`](https://github.com/bjacob/goc/tree/main/tests) | Validation test suite. |
| [`tests/CMakeLists.txt`](https://github.com/bjacob/goc/blob/main/tests/CMakeLists.txt) | Build and register the tests. |

### ISA-extension-specific code paths

We don't know outright what specific ISA features (e.g. AVX-512 features) we will need.

When implementation code needs to use a feature:

* The code path actually using the feature needs to be in an source file with the feature name part of its leaf file name,
  e.g. [`wmma_avx512bf16.cpp`](https://github.com/bjacob/goc/blob/main/src/x86_64/wmma_avx512bf16.cpp) (use the lowercase CPU feature string names from GCC/Clang feature enablement strings, as in the GCC/Clang `-march=` flag, just without the prefix + sign).
* These code paths must be located in an architecture-specific subdir, e.g. [`src/x86_64/`](https://github.com/bjacob/goc/tree/main/src/x86_64).
* The functions actually using the feature also must have the same lowercase GCC-Clang-compatible feature name as a suffix
  in their identifier names.
* The local [`src/x86_64/CMakeLists.txt`](https://github.com/bjacob/goc/blob/main/src/x86_64/CMakeLists.txt) shall handle the build, such that each source file
  gets the appropriate GCC/Clang compile flags. Only GCC and Clang are concerned: any other compiler will get only the
  default, standard-C/C++, architecture-agnostic code paths.
* The local [`src/x86_64/CMakeLists.txt`](https://github.com/bjacob/goc/blob/main/src/x86_64/CMakeLists.txt) will ensure that features are only enabled when the compiler supports the corresponding flag. This can use the standard CMake mechanism to try compiler flags. If the compiler does not support the corresponding flag, the feature will be disabled.
* A reasonable balance will be achieved in granularity of CPU feature testing, w.r.t. our current audience and the fact that we are now in late 2026 and working at AMD. Here are our tiers of usage scenarios. Tier 1: modern (>= Zen4) AMD CPUs must get the best possible code paths. That means that an important baseline feature set for us is what Zen4 supports, and we will as useful add additional code paths for newer Zen architectures, e.g. is Zen6 adds `avx512fp16` we will want to use that. Tier 2: AMD and Intel CPUs from the past 10 years should all support the `x86-64-v3` feature set (which includes AVX2, FMA, F16C, BMI and BMI2). We want to support that decently: make good use of these features as useful, but without necessarily trying to support all the intermediate feature sets between that and Zen4, meaning that for a CPU like Zen3 that supports `x86-64-v3` plus several features, it is not a high priority to utilize these features, even if they would be substantial speedups. It might be reasonable to have one middle step being `x86-64-v4`: that would help some Intel CPUs. I'd be open to that if not too much effort. Tier 3: anything not supporting `x86-64-v3`, we don't care to optimize for, we just want to not crash. That means, use plain scalar code, no SIMD.

### Details on CPU flags and `goc_init_cpu_flags`.

Out of the 64 flag bits, we need to reserve bits for CPU flags, but that needs to include lots of room for future
growth. So we won't want to waste dozens of bits on fine-grained CPU identification that we don't really care for (see previous paragraph).

For x86-64, that means that we are going to lean on folding features into totally-ordered feature sets.
We will reserve 4 bits to encode the baseline feature set:

* `0` = Baseline x86-64
* `1` = `x86-64-v3`
* `2` = `x86-64-v4`
* `3` = The Zen4 feature set.  That means `x86-64-v4` plus these additional AVX-512 features: VPOPCNTDQ, IFMA, VBMI, VBMI2, VNNI, BF16, BITALG, VPCLMULQDQ, GFNI, VAES).
* `4..15` = reserved.

We can then reserve the next 12 bits for additional CPU features.
For example, if we ported to Intel Sapphire Rapids, we would use the above baseline Zen4 feature set
and then allocate bits to encode support for AMX and AVX-512-FP16.

These CPU flags shall mean "the feature may actually be used, at runtime, on this machine". This is a strictly more
restrictive condition than merely "the host CPU supports this feature", since this also requires the operating system
to support the corresponding registers in context switches. Thus, proper detection code needs to combine the CPU
hardware query (which can use `__builtin_cpuid` on x86-64) with either a control-word access or an OS system call to confirm that the OS is allowing userspace code to rely on the feature. Fortunately, we won't have to invent this code as it is already written in

[`~/workspace/hrx-system/runtime/src/iree/base/internal/cpu_x86_64.c`](https://github.com/ROCm/hrx-system/blob/main/runtime/src/iree/base/internal/cpu_x86_64.c)

Take a close look at this code and borrow from it what is relevant to us. The fine-grained CPU feature check will need
to be borrowed into the implementation of `goc_init_cpu_flags`, but unlike the hrx-system code, we won't directly expose these fine-grained feature, we will only expose coalesced coarser feature flags as described above. When borrowing from hrx-system, do not bother reflecting their file structure 1:1, liberally adapt it to minimize our resulting codebase. Disclaimer: I was the main author of that hrx-system code, so I know it's OK to borrow without much crediting, but it will help to have a short comment stating where this is coming from.

### Details on the semantics flags

For now, this can be a 2-bit field encoding an enumeration:

* `0` = Default, loose semantics. This means that it's OK to naively implement a GPU FMA as a CPU FMA or even a separate mul and add, it's OK to have more roundings or fewer roundings, and it's OK to implement a GPU math function like `log2()` as the corresponding C standard library math function, whatever the numerical discrepancies. It would NOT be OK to generate a nonsensical result, so any common bug such as mixing up the lane mapping, or omitting an arithmetic operation, should be expected to produce a test failure outside of a minority of accidental cases where the wrong value just happens to be close to the correct value. TLDR: "Anything that is mathematically sound, neglecting reasonable approximation discrepancies". Testing will rely on fuzzy comparisons with empirically-adjusted tolerances.

* `1` = Empirically bit-exact. This means that the implementation must produce results that are in practice, empirically, observed to be bit-exact. Such implementations might be created by agents observing the values coming out of a program running on the actual hardware, and then empirically constructing a model, or just lookup tables, to match these results.

* `2..3` = Reserved for possible future bit-exact semantics going beyond the above "empirically bit-exact" mode.

Additionally, one separate flag bit will encode the semantics fallback policy:

* `0` = loose fallback policy = The semantics flags is "best effort". If an implementation is not able to honor it, it may silently fall back to the closest supported semantics.
* `1` = strict fallback policy = The semantics flags must be honored precisely. If an implementation is not able to honot it, it must return an error code (see "error codes" above) conveying something like "Unsupported semantics for this instruction". Just a single error code for this whole class of errors, no need to go fine-grained here, this likely will only ever be consumed by tests.

### Details on the floating-point environment flags

Users must specify an explicit floating-point environment in the dedicated flag bits.
We can start with 0 bits: that whole topic may wait until a later phase of execution.
When we do create floating-point environment bits, each of them should have value 0 meaning the common/default case
and nonzero bit values meaning the non-default/less-common cases. Thus, users who don't really care can continue leaving these bits as 0.

All of this floating-point environment concept is about the GPU being emulated, not about the host CPU floating-point environment. The latter is treated as an implementation detail. It is up to the implementation to decide, based on the instruction to be emulated and its semantics flags, what to do about the CPU environment. One specific directive though:
The implementation may assume that the CPU environment is in its default/standard mode: default round-to-nearest, break-ties-to-nearest-even mode, full denormals support. That is part of the contract with the user: if the user calls GoC on a thread with a non-default floating-point environment, they are breaking the contract and thus GoC is relieved from all accuracy promises. So with that out of the way, it is generally expected that at least the loose-semantics implementations will never need to bother about the floating-point environment. Bit-exact semantics may consider mutating the floating-point environment as necessary, but:

* They must restore it (push-pop, do not assume that it was default, do restore it as you found it) before they return.
* They should not automatically resort to floating-point environment manipulation without exploring alternatives (sometimes a simple bit manipulation can achieve the same result without the high overhead of changing FP state).

### GPU floating-point types with no CPU support

For FP8, FP6, FP4 arithmetic with no CPU support, the implementation will need to implement the correct arithmetic locally.
Before embarking on big adventures, it will tour existing rocjitsu code, as well as the conversion helpers in this hrx-systems file: [`hrx-system/runtime/src/iree/base/internal/math.h`](https://github.com/ROCm/hrx-system/blob/main/runtime/src/iree/base/internal/math.h) for a good generic implementation of conversions to/from these types (which may be borrowed as needed). The implementation will make a reasonable effort to mutualize this support code rather than have it be completely duplicated between all instruction implementations, while striking a trade-off with performance.

### Testing.

The test suite in [`tests/`](https://github.com/bjacob/goc/tree/main/tests) will be more of a "validation" than a "unit" test suite: our implementation won't have
much testable finer-grained functions than the API entry points, and testing on these API entry points will basically mean
comparing their numerical outputs to a reference output, using either fuzzy or exact comparisons.

Fuzzy comparisons should broadly follow numpy conventions, as already imported into C in hrx-system: See their
`iree_math_fuzzy_compare_f64` function and the comment accompanying it.

Testing will use pseudorandom numbers. It is OK to use the C++ standard header `<random>` as long as only the
fully defined (not implementation-defined), deterministic subset is used. This is OK: `std::mt19937`, `std::minstd_rand`, any other `std::` specific engine.  This is NOT OK:  `std::default_random_engine` (not OK because implementation-defined which engine is selected as the default), `std::*_distribution` (the generated values are implementation-defined).  Tests should use random engines in a way that guarantees that the values remain the same regardless of the order in which testcases are run, or filtered. This could be achieved by letting each test use its own random engine object.

Testing should use the GTest framework.  The CMake build should look for it as a system-installed dependency, and if not found, fall back to the CMake `FetchContent` feature.

Testing should be structured such that all implementation code paths are testable without GPU access.
This means that we completely decouple the discovery (what are GPUs doing exactly, how do we isolate the contract that bit-exactness actually means) from the testing (ensure that GoC implementation doesn't regress wrt the contract). Tests should be self-contained in covering the contract. This means golden output values encoded in tests.

There is no tension between using `<random>` for test inputs and golden values for test outputs, once it is well understood that we use `<random>` in a completely deterministic, defined (implementation-independent) way.


## Execution plan

### Prioritization

Multiple dimensions here, and I am not asking for lexicographic order: do not try to exhaust one dimension first. Grow intelligently gradually along multiple dimensions.

#### Dimension 1: GPU architectures

First architecture to start with: RDNA4. Then extend to this set: RDNA3/4, CDNA3/4/5.

#### Dimension 2: instructions

Start with a few key instructions of each class: a few WMMA/MFMA, a few elementwise basic arithmetic (vector FMA), and maybe one math function (say `log2`). Then grow to more WMMA/MFMA and dot-products, remaining element-wise arithmetic, remaining math functions.

#### Dimension 3: semantics modes

Start with semantic mode 0 (default loose), then extend to mode 1 (empirical bit-exact) but only to the extent that you have existing bit-exact rocjitsu code to learn from (e.g. I believe WMMA on RDNA4). Do not try to reverse-engineer for now.
Do not bother about semantics modes 2-3 for now, treat them as just reserved for future usage.

#### Dimension 4: instruction flags

For instructions that have flags (e.g. `NEG` on RDNA4 WMMA), focus at first on the default common case, then later grow to include the support for these flags. Note: the implementation and testing of instruction flags shall be grown simultaneously, so that there is never a question of "what should tests do if an instruction flag hasn't been implemented yet". Before it exists at all, the question itself doesn't exit. Once it exists, it exists jointly in the implementation and test. This leaves open the question of what if the instruction flag exists at an early stage where only loose semantics are implemented, and then bit-exact semantics are implemented but only for some instruction flags values. In that case, an instruction with instruction flags not supported by the given semantics flags, and having the "strict" no-fallback bit set in its semantics flag, is treated as the error case of unsupported semantics. That is, the error-generation won't try to distinguish the fact that only that particular instruction flag is not supported by these semantics.

#### Dimension 5: floating-point environment

The default loose testing mode shouldn't be too sensitive to this anyway. When you get to bit-exact mode, that will presumably be very sensitive to the floating-point environment, so testing will need to scale accordingly.


## Implicit architectural register outputs

An instruction API appends one nullable pointer per implicit architectural
register it may update, named after that register in lowercase. Registers already
represented by an operand/output parameter are not duplicated. Instructions
without such side effects gain no parameter. Register updates follow the ISA;
temporary internal changes restored by the instruction are not extra outputs.
`GOC_ERROR_UNSUPPORTED_GLOBAL_STATE` indicates that faithful updates to a
requested implicit architectural register are unavailable. This mechanism applies
to any such global state, not only exception flags.

`uint32_t *excp_flag_user` represents RDNA4's wave-wide sticky exception register.
A non-null pointer accumulates newly generated bits with bitwise OR; existing bits
are never cleared. Only participating lanes contribute, following each
instruction's execution rules. `NULL` opts out. Storage must not overlap operands, except where explicitly
permitted by an instruction contract (currently `V_RCP_IFLAG_F32`).
Every API error preserves every output, including this register. Guest exception
reporting must never rely on changing or sampling the host FP environment.

Faithful optional global-state output is required **only when bit-exact semantics
are requested and the corresponding output pointer is non-null**, independently
of `GOC_SEMANTICS_STRICT`. A null pointer opts out in every semantics mode; it
creates no obligation to compute that register's update. This rule concerns
optional global-state outputs, not ordinary instruction results such as comparison
masks or `SCC`, which remain required.

In loose semantics there is **no requirement to produce any optional global-state
output**, even with a non-null pointer. SIMD paths are expected to skip this work
and leave the register unchanged. Do not add exception classification, reductions,
scalar fallbacks, or other reporting overhead to loose SIMD paths merely because
an output pointer was supplied. Loose arithmetic does not promise to match GPU
arithmetic, so deriving exception flags from those approximate results cannot
provide a hardware-faithful exception history; spending performance on that goal
would be futile without also implementing the hardware's arithmetic semantics.
Loose mode never returns `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE`. Existing inexpensive
reporting may remain, but callers must not rely on its completeness or accuracy.
Any exception bits that are reported still accumulate with OR.

For bit-exact semantics with a non-null pointer, unavailable faithful updates
return `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE`; they must not silently fall back to
loose reporting. Currently, `V_RCP_IFLAG_F32` retains its implemented
integer-divide-by-zero reporting. Floating comparisons also implement exact
invalid/input-denormal reporting, validated against RX 9070 captures. Their exact
reporting path uses integer classification; loose and null-output paths keep SIMD
dispatch without that work. All eight scalar rounding instructions also report
invalid/input-denormal flags in exact mode; rounding a fractional value does not
raise inexact. Their reporting is checked against 6,291,456 RX 9070 captures.

FP16/32/64 division-fixup reporting covers all six floating exception bits,
including modifier suppression, and is checked against 786,432 RX 9070 captures.

FP16 FMA, FMAC, both literal forms, and packed FMA/FMAC also report exceptions.
Their integer FMA intermediate supplies inexact/underflow/overflow classification;
`CLAMP` suppresses all flags and output scaling suppresses underflow/inexact.
Packed results accumulate exceptions from both selected halves. Loose SIMD
paths skip all of this work.

FP16/32 `SIN`/`COS` report invalid, input-denormal, underflow, and inexact flags.
Their hardware corpora cover every FP16 encoding, FP32 exponent/mantissa patterns,
and FP32 neighborhoods around underflow, quarter turns, zeros, and NaNs.

FP32/64 `DIV_FMAS` report overflow, underflow and inexact after conditional
post-scaling; hardware suppresses invalid/input-denormal flags. Integer guard and
sticky bits supply reporting, with additional FMA16/DIV_FMAS hardware captures
around underflow and overflow rounding boundaries.

FP16 `FMA_MIXLO`/`FMA_MIXHI` accept reporting and leave the register unchanged:
1,048,576 hardware flag reads are zero, including invalid arithmetic, overflow,
source-format/half selection, modifiers, and saturation.
`DOT2_F32_F16`/`DOT2_F32_BF16` likewise leave the register unchanged, validated
with 524,288 hardware flag reads across packed selections, negation, and clamp.

Other affected instructions reject requested reporting in non-loose modes before accessing operands, even for empty
`exec_mask`, with precedence over other validation errors. Null retains existing
validation, numerical behavior, and SIMD dispatch. Implement and hardware-test
exception generation incrementally for bit-exact semantics.

The classification uses the [RDNA4 ISA reference](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)
(§3.4.10, §6.8, §7.7, §7.11, §12.3 and instruction descriptions), cross-checked
against LLVM AMDGPU instruction metadata. LLVM metadata alone is insufficient:
for example, the ISA explicitly supports exceptions for floating min/max and
explicitly excludes FP8 dot products. `V_RCP_IFLAG_F32` reports integer
divide-by-zero through the same register. Matrix instructions, interpolation,
CLASS comparisons, cube operations, FP8 conversions/dots, and other non-reporting
conversion variants retain their existing signatures. The parameter is present
if an instruction can report exceptions in any supported ISA mode, even when a
particular invocation or currently accepted modifier cannot produce them.

Make frequent local git commits using my @bjacob GitHub identity: name = "Benoit Jacob", email = jacob.benoit.1@gmail.com

Make feature work and testing expansing finely intertwined, each git commit must contain tests for the code just added,
and tests must be verified to be all passing before committing.

Borrow the .clang-format from rocjitsu, do clang-format before each commit.

Borrow the license from rocjitsu.

The CMake build directory should be out-of-tree. Use Ninja (generator).
Build and test with full CPU parallelism.

Prioritize simplicity and human understanding when refactoring. A reader should
be able to recognize the algorithm, follow its data flow, and see the important
semantic and hardware differences without mentally expanding layers of templates
or jumping through many files. Familiar, explicit code and some local repetition
are preferable to an abstraction that makes these tasks harder.

Share code when it centralizes a meaningful rule or algorithm and makes its
current callers easier to understand. Do not introduce adapters, policy types,
or generic executors merely to remove similar-looking syntax. Justify each new
abstraction using existing callers, not speculative future reuse. Keep hardware
intrinsics and instruction-specific behavior visible where they aid understanding.

Before accepting a refactoring, explain what a maintainer now needs to understand
and change in fewer places, and account for the added concepts, indirection, and
helper files. Review the resulting code as a whole, not just the deleted lines.
Source and binary size measurements support this judgment; they are not quotas
or substitutes for readability. Preserve correctness and the agreed performance
constraints. Reconsider or remove abstractions that fail this simplicity test,
including ones introduced earlier in the same project.

Separate file banners, header guards, include groups, and declarations with blank lines.

Within include/, quoted #include paths are relative to the containing header's
own directory, so public headers do not depend on the consumer's include paths.
Elsewhere, project-local #include paths are relative to include/, or to src/ for
internal headers. CMake must pass include/ publicly and src/ privately to
implementation and test targets.

Use goc/goc.h for the public API in implementations, tests, and consumers. It is the
only header directly under include/goc/; component headers live in include/goc/detail/.
The umbrella directly includes every component header. Detail headers and their
standalone compile checks may include detail headers directly. Otherwise include
internal and standard-library dependencies directly rather than transitively.

Use architecture-neutral names for shared implementations, headers, tests and
benchmarks. Public instruction APIs use goc_<mnemonic>, with _wave64 appended
for dedicated Wave64 variants. If a newer architecture changes an existing
mnemonic's semantics, append its architecture suffix before _wave64 (if present);
the FP32-output FP16/BF16 WMMA APIs now use this distinction for RDNA3/RDNA4.
Keep hardware provenance in fixture
comments. CPU-specific implementations retain their CPU feature suffix.

Closing braces for namespaces and extern "C" blocks must carry a comment naming
what they close (for example, } // namespace goc or } // extern "C"). This does
not apply to closing braces for classes or functions.

Function comments describe the function contract: inputs, outputs, preconditions,
side effects, and errors. Keep neighboring usage topics in the relevant API or
usage documentation instead.

Use blank lines to separate logical groups of constants, declarations, and code.
A comment must stay attached to its subject, with a blank line separating that
group from neighboring material to which the comment does not apply.

Use C++17 [[...]] syntax for C++ attributes, with a gnu:: namespace where needed,
rather than __attribute__((...)).

Only instructions whose behavior depends on EXEC take exec_mask. Scalar,
pseudo-scalar, WMMA and SWMMAC APIs omit it; matrix instructions read and write
all lanes even when architectural EXEC is zero.

Improving sparse exec_mask performance is a non-goal. Aim for mask-independent
performance by computing full results and masking destination stores. Do not add
mask-density checks, active-lane iteration, or sparse-mask specializations:
their code size and runtime overhead are unwanted. An early return for a zero
effective mask is explicitly allowed; it does not justify further mask-dependent
optimizations.

Production implementations of API entry points must never explicitly mutate the
host CPU floating-point environment, even temporarily. Do not change rounding
modes, exception masks, denormal controls, or exception flags, and do not use
save/modify/restore guards or helpers for that purpose. Ordinary arithmetic may
raise sticky exception flags. Tests may modify the FP environment to verify
behavior under different host settings.

Append implicit architectural register outputs to instruction API parameter
lists, exactly one pointer per register the instruction may write. Use the
lowercase ISA register name and its native-width unsigned integer type. A null
pointer opts out. Apply the register's specified update semantics: in particular,
EXCP_FLAG_USER is a wave-wide uint32_t accumulated with bitwise OR, not cleared
or overwritten. Only participating lanes contribute. Leave all outputs unchanged
on API errors. Faithful optional global-state output is required only for bit-exact
semantics with a non-null output pointer, regardless of GOC_SEMANTICS_STRICT.
Otherwise there is no requirement to compute the update. Ordinary instruction
results, including comparison masks and SCC, remain required.

In loose semantics, SIMD paths are expected to skip optional global-state updates
and leave the register unchanged, even when its pointer is non-null. Do not add
reporting work or sacrifice SIMD performance to produce exception flags from loose
arithmetic: its results need not match hardware, so those flags cannot provide a
faithful hardware exception history. Existing inexpensive reporting may remain,
but loose-mode callers cannot rely on its completeness or accuracy. Never return
GOC_ERROR_UNSUPPORTED_GLOBAL_STATE in loose mode. For bit-exact semantics with a
non-null pointer, return that error if faithful updates are unavailable.

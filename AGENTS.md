Make frequent local git commits using my @bjacob GitHub identity: name = "Benoit Jacob", email = jacob.benoit.1@gmail.com

Make feature work and testing expansing finely intertwined, each git commit must contain tests for the code just added,
and tests must be verified to be all passing before committing.

Borrow the .clang-format from rocjitsu, do clang-format before each commit.

Borrow the license from rocjitsu.

The CMake build directory should be out-of-tree. Use Ninja (generator).
Build and test with full CPU parallelism.

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

GPU-architecture-specific implementation files, internal headers, tests and
fixtures must name the GPU architecture in their filenames (for example rdna4_).
CPU-specific implementations also retain their CPU feature suffix.

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

Make frequent local git commits using my @bjacob GitHub identity: name = "Benoit Jacob", email = jacob.benoit.1@gmail.com

Make feature work and testing expansing finely intertwined, each git commit must contain tests for the code just added,
and tests must be verified to be all passing before committing.

Borrow the .clang-format from rocjitsu, do clang-format before each commit.

Borrow the license from rocjitsu.

The CMake build directory should be out-of-tree. Use Ninja (generator).
Build and test with full CPU parallelism.

Separate file banners, header guards, include groups, and declarations with blank lines.
Project-local #include paths are relative to include/, or to src/ for internal
headers, including includes among files in the same directory. CMake must pass
include/ publicly and src/ privately to implementation and test targets.
Include the headers defining what each file uses directly, including standard
library headers; do not rely on transitive includes. The umbrella goc.h directly
includes every public component header.
Closing braces for namespaces and extern "C" blocks must carry a comment naming
what they close (for example, } // namespace goc or } // extern "C"). This does
not apply to closing braces for classes or functions.
Function comments describe the function contract: inputs, outputs, preconditions,
side effects, and errors. Keep neighboring usage topics in the relevant API or
usage documentation instead.

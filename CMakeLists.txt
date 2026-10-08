cmake_minimum_required(VERSION 3.20)
project(goc VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "GoC requires an out-of-tree build")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Enable x86-64 SIMD implementations when targeting x86-64 with GCC/Clang.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" goc_target_processor)

unset(GOC_ENABLE_X86_64 CACHE)
set(GOC_ENABLE_X86_64 OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$"
   AND goc_target_processor MATCHES "^(x86_64|amd64|x64|x86-64)$"
   AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(GOC_ENABLE_X86_64 ON)
endif()

add_subdirectory(src)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

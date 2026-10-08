cmake_minimum_required(VERSION 3.20)
project(goc VERSION 0.1.0 LANGUAGES C CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "GoC requires an out-of-tree build")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Prefer explicit target selections over CMAKE_SYSTEM_PROCESSOR, which can
# describe the host for native builds (notably with Visual Studio).
set(goc_target_processor "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  # Mixed-architecture Apple builds use the portable paths.
  set(goc_target_processor "${CMAKE_OSX_ARCHITECTURES}")
elseif(CMAKE_VS_PLATFORM_NAME)
  set(goc_target_processor "${CMAKE_VS_PLATFORM_NAME}")
elseif(CMAKE_CXX_COMPILER_TARGET)
  string(REGEX REPLACE "-.*$" "" goc_target_processor "${CMAKE_CXX_COMPILER_TARGET}")
endif()
string(TOLOWER "${goc_target_processor}" goc_target_processor)

# X86 is shorthand for x86-64 only. Remove the former user option from old caches.
unset(GOC_ENABLE_X86 CACHE)
set(GOC_ENABLE_X86 OFF)
if(goc_target_processor MATCHES "^(x86_64|amd64|x64|x86-64)$" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(GOC_ENABLE_X86 ON)
endif()

add_subdirectory(src)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

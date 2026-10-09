find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

function(goc_test_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  endif()
endfunction()

# The private decoder is tested directly without exporting it from either library.
add_executable(goc_cpu_decode_test cpu_decode_test.cpp ${PROJECT_SOURCE_DIR}/src/cpu_decode.cpp)
target_include_directories(goc_cpu_decode_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(goc_cpu_decode_test PRIVATE GTest::gtest_main)
goc_test_warnings(goc_cpu_decode_test)
gtest_discover_tests(goc_cpu_decode_test TEST_PREFIX "internal.")

# Exercise every public RDNA4 entry point, including future additions.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/goc/detail/goc_rdna4.h")
file(STRINGS "${PROJECT_SOURCE_DIR}/include/goc/detail/goc_rdna4.h"
  goc_rdna4_declarations REGEX "^GOC_API int goc_rdna4(w64)?_")
set(goc_rdna4_api_list "")
foreach(declaration IN LISTS goc_rdna4_declarations)
  string(REGEX MATCH "goc_rdna4(w64)?_[a-zA-Z0-9_]+" symbol "${declaration}")
  string(APPEND goc_rdna4_api_list "GOC_RDNA4_API(${symbol})\n")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rdna4_api_symbols.inc" "${goc_rdna4_api_list}")

foreach(library IN LISTS goc_library_targets)
  string(REPLACE "goc_" "" variant "${library}")
  foreach(test IN ITEMS rdna4_dpp16 rdna4_dpp8 rdna4_instruction_flags rdna4_permlane rdna4_scalar_pack rdna4_scalar_compare rdna4_scalar_convert rdna4_scalar_round rdna4_scalar_fma rdna4_scalar_fp rdna4_scalar_field rdna4_scalar_bits rdna4_scalar_integer rdna4_rcp_iflag rdna4_pseudo_scalar rdna4_float_compare rdna4_integer_compare rdna4_class rdna4_interp16 rdna4_interp32 rdna4_cndmask rdna4_trig_preop rdna4_mullit rdna4_pack rdna4_swmmac_integer rdna4_swmmac8 rdna4_swmmac16 rdna4_mad64 rdna4_carry rdna4_div_fmas rdna4_div_scale rdna4_div_fixup rdna4_cube rdna4_fp8_narrow rdna4_byte_pack rdna4_integer_conversion rdna4_normalized rdna4_packed_conversion rdna4_fp8_conversion rdna4_byte_conversion rdna4_conversion16 rdna4_conversion64 rdna4_conversion32 rdna4_mixed_fma rdna4_bit_count rdna4_boolean rdna4_bitfield rdna4_integer_ternary rdna4_sad rdna4_shift rdna4_half_trig rdna4_trig rdna4_integer_mad rdna4_integer16_ternary rdna4_integer16 rdna4_packed_mad rdna4_packed_integer rdna4_packed_binary rdna4_packed_fma rdna4_literal_fma rdna4_fmac rdna4_half_fma rdna4_half_minmax rdna4_half_exponent rdna4_half_unary rdna4_half_binary rdna4_integer_add rdna4_integer_mul rdna4_integer_minmax rdna4_ldexp rdna4_frexp_exp rdna4_fp64_unary rdna4_fp64 rdna4_minmax rdna4_binary rdna4_half_dot rdna4_fp8_dot rdna4_dot2 rdna4_integer_dot rdna4_arithmetic rdna4_unary rdna4_exec_mask cpu rdna4_wmma rdna4_packed_wmma rdna4_subbyte_wmma)
    set(target "goc_${test}_test_${variant}")
    add_executable(${target} "${test}_test.cpp")
    if(test STREQUAL "rdna4_arithmetic")
      target_sources(${target} PRIVATE rdna4_c_api_test.c)
    endif()
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${target} PRIVATE ${library} GTest::gtest_main)
    goc_test_warnings(${target})
    gtest_discover_tests(${target} TEST_PREFIX "${variant}.")
  endforeach()

  set(benchmark "goc_rdna4_benchmark_${variant}")
  add_executable(${benchmark} rdna4_benchmark.cpp)
  target_include_directories(${benchmark} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${benchmark} PRIVATE ${library})
  goc_test_warnings(${benchmark})
  if(GOC_ENABLE_X86_64 AND GOC_COMPILER_X86_64_V3)
    target_compile_definitions(${benchmark} PRIVATE GOC_BENCH_HAVE_X86_64_V3=1)
  endif()
  if(GOC_ENABLE_X86_64 AND GOC_COMPILER_X86_64_V4)
    target_compile_definitions(${benchmark} PRIVATE GOC_BENCH_HAVE_X86_64_V4=1)
  endif()
  if(GOC_ENABLE_X86_64 AND GOC_COMPILER_AVX512BF16)
    target_compile_definitions(${benchmark} PRIVATE GOC_BENCH_HAVE_AVX512BF16=1)
  endif()
  if(GOC_ENABLE_X86_64 AND GOC_COMPILER_AVX512VNNI)
    target_compile_definitions(${benchmark} PRIVATE GOC_BENCH_HAVE_AVX512VNNI=1)
  endif()
  # Smoke-test correctness and execution, with no timing/speedup assertions.
  add_test(NAME ${variant}.rdna4_benchmark COMMAND ${benchmark} --min-ms 0 1)
  add_test(NAME ${variant}.rdna4_benchmark_output
    COMMAND ${CMAKE_COMMAND} -DBENCHMARK=$<TARGET_FILE:${benchmark}>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_rdna4_benchmark_output.cmake)
endforeach()

if(GOC_STATIC)
  add_executable(cpuinfo cpuinfo.c)
  target_link_libraries(cpuinfo PRIVATE goc_static)
  goc_test_warnings(cpuinfo)
  add_test(NAME static.cpuinfo COMMAND cpuinfo)
endif()

# Compile the umbrella and each detail header independently as C and C++.
# Absolute includes and no GoC include path verify header-relative dependencies.
foreach(header IN ITEMS common rdna4 rdna4_umbrella export)
  foreach(extension IN ITEMS c cpp)
    set(source "${CMAKE_CURRENT_BINARY_DIR}/${header}_header_test.${extension}")
    configure_file("headers/${header}.c" "${source}" @ONLY)
    set(target "goc_${header}_header_${extension}_test")
    add_library(${target} OBJECT "${source}")
    goc_test_warnings(${target})
  endforeach()
endforeach()

# Check ELF exports, including static GoC embedded into a consumer shared library.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_NM)
  if(GOC_STATIC)
    add_library(goc_embedding_test SHARED embedding.cpp)
    set_target_properties(goc_embedding_test PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
    )
    target_link_libraries(goc_embedding_test PRIVATE "-Wl,--whole-archive" goc_static "-Wl,--no-whole-archive")
    add_test(NAME static.exports COMMAND ${CMAKE_COMMAND}
      "-DNM=${CMAKE_NM}" "-DLIBRARY=$<TARGET_FILE:goc_embedding_test>" -DVARIANT=static
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_exports.cmake")
  endif()
  if(GOC_SHARED)
    add_test(NAME shared.exports COMMAND ${CMAKE_COMMAND}
      "-DNM=${CMAKE_NM}" "-DLIBRARY=$<TARGET_FILE:goc_shared>" -DVARIANT=shared
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_exports.cmake")
  endif()
endif()

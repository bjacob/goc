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

add_executable(goc_fp_environment_test fp_environment_test.cpp)
target_link_libraries(goc_fp_environment_test PRIVATE GTest::gtest_main)
goc_test_warnings(goc_fp_environment_test)
gtest_discover_tests(goc_fp_environment_test TEST_PREFIX "internal.")

add_executable(goc_capture_hash_test capture_hash_test.cpp)
target_link_libraries(goc_capture_hash_test PRIVATE GTest::gtest_main)
goc_test_warnings(goc_capture_hash_test)
gtest_discover_tests(goc_capture_hash_test TEST_PREFIX "internal.")

# Exercise every public instruction entry point, including future additions.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/goc/detail/goc_instructions.h")
file(STRINGS "${PROJECT_SOURCE_DIR}/include/goc/detail/goc_instructions.h"
  goc_declarations REGEX "^int goc_")
set(goc_api_list "")
foreach(declaration IN LISTS goc_declarations)
  string(REGEX MATCH "goc_[a-zA-Z0-9_]+" symbol "${declaration}")
  string(APPEND goc_api_list "GOC_API(${symbol})\n")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/api_symbols.inc" "${goc_api_list}")

# Keep exception-contract coverage exhaustive as public signatures grow.
file(READ "${PROJECT_SOURCE_DIR}/include/goc/detail/goc_instructions.h" goc_header)
string(REGEX MATCHALL "int goc_[a-zA-Z0-9_]*\\([^;]*excp_flag_user\\)"
  goc_reporting_declarations "${goc_header}")
set(goc_reporting_list "")
foreach(declaration IN LISTS goc_reporting_declarations)
  string(REGEX MATCH "goc_[a-zA-Z0-9_]*" symbol "${declaration}")
  if(symbol STREQUAL "goc_v_rcp_iflag_f32" OR
     symbol MATCHES "^goc_(s_cmp|v_cmpx?)_[a-z]+_f(16|32|64)$" OR
     symbol MATCHES "^goc_v_(fma|fmac|fmamk|fmaak|pk_fma|pk_fmac)_f16$" OR
     symbol MATCHES "^goc_v_(sin|cos)_f(16|32)$" OR
     symbol MATCHES "^goc_v_dot2_f32_(f16|bf16)_rdna4$" OR
     symbol MATCHES "^goc_v_fma_mix(lo|hi)_f16$" OR
     symbol MATCHES "^goc_v_div_fmas_f(32|64)$" OR
     symbol MATCHES "^goc_v_div_fixup_f(16|32|64)$" OR
     symbol MATCHES "^goc_s_(ceil|floor|trunc|rndne)_f(16|32)$")
    continue() # Already implements accumulation; covered by its hardware corpus.
  endif()
  string(APPEND goc_reporting_list "GOC_REPORTING_API(${symbol})\n")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/reporting_symbols.inc" "${goc_reporting_list}")

foreach(test IN ITEMS minmax_spellings guest_mode ds_swizzle ds_permute swap relative_move constant_compare scalar_exec lane_transfer scalar_move scalar_spellings dot2_architecture move wmma_replicated wmma_replicated_exact fma_rounding_exceptions trig_exceptions arithmetic_exceptions compare_exceptions exception_registers dpp_half_unary dpp_half_fma dpp_half_minmax dpp_half_binary dpp_integer16_ternary dpp_integer16 dpp_boolean16 dpp_integer_mad dpp_integer_mul dpp_integer_add dpp_bitfield dpp_integer_ternary dpp_integer_minmax dpp_integer dpp_unary fp64_output output_scaling dpp_arithmetic dpp16 dpp8 instruction_flags permlane scalar_pack scalar_compare scalar_convert scalar_round scalar_fma scalar_fp scalar_field scalar_bits scalar_integer rcp_iflag pseudo_scalar float_compare integer_compare class interp16 interp32 cndmask trig_preop mullit pack swmmac_integer swmmac8 swmmac16 mad64 carry div_fmas div_scale div_fixup cube fp8_narrow byte_pack integer_conversion normalized packed_conversion fp8_conversion byte_conversion conversion16 conversion64 conversion32 mixed_fma bit_count boolean bitfield integer_ternary sad shift half_trig trig integer_mad integer16_ternary integer16 packed_mad packed_integer packed_binary packed_fma literal_fma fmac half_fma half_minmax half_exponent half_unary half_binary integer_add integer_mul integer_minmax ldexp frexp_exp fp64_unary fp64 minmax binary half_dot fp8_dot dot2 integer_dot arithmetic unary exec_mask cpu wmma packed_wmma subbyte_wmma)
  set(target "goc_${test}_test")
  add_executable(${target} "${test}_test.cpp")
  if(test STREQUAL "arithmetic")
    target_sources(${target} PRIVATE c_api_test.c)
  endif()
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${target} PRIVATE goc GTest::gtest_main)
  goc_test_warnings(${target})
  gtest_discover_tests(${target})
endforeach()

set(benchmark "goc_benchmark")
add_executable(${benchmark} benchmark.cpp)
target_include_directories(${benchmark} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${benchmark} PRIVATE goc)
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
add_test(NAME benchmark COMMAND ${benchmark} --min-ms 0 1)
add_test(NAME benchmark_output
  COMMAND ${CMAKE_COMMAND} -DBENCHMARK=$<TARGET_FILE:${benchmark}>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_benchmark_output.cmake)

add_executable(cpuinfo cpuinfo.c)
target_link_libraries(cpuinfo PRIVATE goc)
goc_test_warnings(cpuinfo)
add_test(NAME cpuinfo COMMAND cpuinfo)

# Compile the umbrella and each detail header independently as C and C++.
# Absolute includes and no GoC include path verify header-relative dependencies.
foreach(header IN ITEMS common instructions umbrella)
  foreach(extension IN ITEMS c cpp)
    set(source "${CMAKE_CURRENT_BINARY_DIR}/${header}_header_test.${extension}")
    configure_file("headers/${header}.c" "${source}" @ONLY)
    set(target "goc_${header}_header_${extension}_test")
    add_library(${target} OBJECT "${source}")
    goc_test_warnings(${target})
  endforeach()
endforeach()

# A consumer shared library must export only its own API when embedding GoC.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_NM)
  add_library(goc_embedding_test SHARED embedding.cpp)
  set_target_properties(goc_embedding_test PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  target_link_libraries(goc_embedding_test PRIVATE "-Wl,--whole-archive" goc "-Wl,--no-whole-archive")
  add_test(NAME embedding.exports COMMAND ${CMAKE_COMMAND}
    "-DNM=${CMAKE_NM}" "-DLIBRARY=$<TARGET_FILE:goc_embedding_test>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_exports.cmake")
endif()

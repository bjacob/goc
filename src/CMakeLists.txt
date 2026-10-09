set(goc_library_targets)
if(GOC_STATIC)
  add_library(goc_static STATIC)
  target_compile_definitions(goc_static PUBLIC GOC_STATIC_DEFINE)
  list(APPEND goc_library_targets goc_static)
endif()
if(GOC_SHARED)
  add_library(goc_shared SHARED)
  target_compile_definitions(goc_shared PRIVATE GOC_BUILDING_LIBRARY)
  list(APPEND goc_library_targets goc_shared)
endif()

foreach(target IN LISTS goc_library_targets)
  target_sources(${target} PRIVATE rdna4_cube.cpp rdna4_fp8_narrow.cpp rdna4_integer_conversion.cpp rdna4_normalized.cpp rdna4_packed_conversion.cpp rdna4_fp8_conversion.cpp rdna4_byte_conversion.cpp rdna4_conversion16.cpp rdna4_conversion64.cpp rdna4_conversion32.cpp rdna4_mixed_fma.cpp rdna4_bit_count.cpp rdna4_boolean.cpp rdna4_bitfield.cpp rdna4_integer_ternary.cpp rdna4_sad.cpp rdna4_shift.cpp rdna4_half_trig.cpp rdna4_trig.cpp rdna4_integer_mad.cpp rdna4_integer16_ternary.cpp rdna4_integer16.cpp cpu.cpp cpu_decode.cpp rdna4_arithmetic.cpp rdna4_half_fma.cpp rdna4_packed_fma.cpp rdna4_unary.cpp rdna4_half_unary.cpp rdna4_half_exponent.cpp rdna4_wmma.cpp rdna4_subbyte_wmma.cpp rdna4_dot.cpp rdna4_integer_dot.cpp rdna4_fp8_dot.cpp rdna4_half_dot.cpp rdna4_binary.cpp rdna4_half_binary.cpp rdna4_minmax.cpp rdna4_half_minmax.cpp rdna4_fp64.cpp rdna4_frexp_exp.cpp rdna4_ldexp.cpp rdna4_integer_minmax.cpp rdna4_integer_mul.cpp rdna4_integer_add.cpp)
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME goc
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  target_include_directories(${target}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  )
  if(UNIX)
    target_link_libraries(${target} PRIVATE m)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
  endif()
  if(GOC_ENABLE_X86_64)
    target_compile_definitions(${target} PRIVATE GOC_X86_64_QUERY=1)
  endif()
endforeach()

# Avoid colliding with the shared library's import archive on Windows.
if(WIN32 AND GOC_STATIC)
  set_target_properties(goc_static PROPERTIES OUTPUT_NAME goc_static)
endif()

if(GOC_ENABLE_X86_64)
  add_subdirectory(x86_64)
endif()

set(goc_library_targets ${goc_library_targets} PARENT_SCOPE)

include(CheckCXXCompilerFlag)

set(goc_x86_64_v3_flags "-march=x86-64-v3")
set(goc_x86_64_v4_flags "-march=x86-64-v4")
set(goc_avx512vnni_flags ${goc_x86_64_v4_flags} "-mavx512vnni")
set(goc_avx512bf16_flags ${goc_x86_64_v4_flags} "-mavx512bf16")

# Format the feature flag lists as command-line strings for the compiler checks.
list(JOIN goc_avx512vnni_flags " " goc_avx512vnni_check_flags)
list(JOIN goc_avx512bf16_flags " " goc_avx512bf16_check_flags)

check_cxx_compiler_flag("${goc_x86_64_v3_flags}" GOC_COMPILER_X86_64_V3)
check_cxx_compiler_flag("${goc_x86_64_v4_flags}" GOC_COMPILER_X86_64_V4)
check_cxx_compiler_flag("${goc_avx512bf16_check_flags}" GOC_COMPILER_AVX512BF16)
check_cxx_compiler_flag("${goc_avx512vnni_check_flags}" GOC_COMPILER_AVX512VNNI)

set(goc_x86_64_v3_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_dpp_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_permlane_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_rcp_iflag_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_float_compare_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_compare_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_class_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_interp16_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_interp32_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_trig_preop_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_mullit_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_pack_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_swmmac_integer_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_swmmac_float_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_mad64_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_carry_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_div_fmas_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_div_scale_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_div_fixup_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_cube_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fp8_narrow_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_conversion_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_normalized_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_packed_conversion_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fp8_conversion_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_byte_conversion_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_conversion16_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_conversion64_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_conversion32_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_mixed_fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_bit_count_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_bitfield_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_ternary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_sad_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_shift_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_half_trig_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_trig_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_mad_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer16_ternary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer16_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_half_fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_packed_fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_half_binary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_binary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fp64_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_frexp_exp_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_ldexp_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_minmax_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_mul_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_add_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_half_minmax_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_minmax_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_dot2_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_half_dot_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fp8_dot_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fp8_wmma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_half_exponent_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_half_unary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_unary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_wmma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_wmma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_dot_x86_64_v3.cpp
)

set(goc_x86_64_v4_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_dpp_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_permlane_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_rcp_iflag_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_float_compare_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_compare_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_class_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_interp32_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_cndmask_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_trig_preop_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_mullit_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_pack_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_swmmac_integer_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_swmmac_float_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_mad64_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_carry_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_div_fmas_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_div_scale_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_div_fixup_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_cube_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fp8_narrow_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_conversion_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_normalized_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_packed_conversion_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fp8_conversion_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_byte_conversion_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_conversion16_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_conversion64_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_conversion32_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_bit_count_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_boolean_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_bitfield_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_ternary_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fma_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_ldexp_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_minmax_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_mul_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_add_x86_64_v4.cpp
)

set(goc_avx512bf16_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_wmma_avx512bf16.cpp
)

set(goc_avx512vnni_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_wmma_avx512vnni.cpp
)

# Keep source selection, compile flags and feature definitions together for each ISA.
function(goc_add_x86_64_backend feature)
  if(NOT GOC_COMPILER_${feature})
    return()
  endif()
  string(TOLOWER "${feature}" backend)
  foreach(target IN LISTS goc_library_targets)
    target_sources(${target} PRIVATE ${goc_${backend}_sources})
    set_source_files_properties(${goc_${backend}_sources}
      TARGET_DIRECTORY ${target}
      PROPERTIES COMPILE_OPTIONS "${goc_${backend}_flags}"
    )
    target_compile_definitions(${target} PRIVATE GOC_HAVE_${feature}=1)
  endforeach()
endfunction()

foreach(feature IN ITEMS X86_64_V3 X86_64_V4 AVX512BF16 AVX512VNNI)
  goc_add_x86_64_backend(${feature})
endforeach()

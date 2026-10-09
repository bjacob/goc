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
  ${CMAKE_CURRENT_SOURCE_DIR}/dpp_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/permlane_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rcp_iflag_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/float_compare_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_compare_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/class_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interp16_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interp32_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/trig_preop_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mullit_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pack_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/swmmac_integer_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/swmmac_float_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mad64_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/carry_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/div_fmas_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/div_scale_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/div_fixup_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cube_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp8_narrow_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_conversion_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalized_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/packed_conversion_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp8_conversion_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/byte_conversion_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conversion16_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conversion64_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conversion32_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mixed_fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bit_count_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bitfield_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_ternary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sad_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/shift_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/half_trig_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/trig_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_mad_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer16_ternary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer16_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/half_fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/packed_fma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/half_binary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/binary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp64_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/frexp_exp_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ldexp_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_minmax_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_mul_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_add_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/half_minmax_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/minmax_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dot2_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/half_dot_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp8_dot_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp8_wmma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/half_exponent_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/half_unary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/unary_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_wmma_x86_64_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_dot_x86_64_v3.cpp
)

set(goc_x86_64_v4_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/dpp_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/permlane_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rcp_iflag_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/float_compare_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_compare_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/class_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interp32_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cndmask_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/trig_preop_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mullit_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pack_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/swmmac_integer_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/swmmac_float_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mad64_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/carry_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/div_fmas_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/div_scale_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/div_fixup_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cube_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp8_narrow_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_conversion_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalized_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/packed_conversion_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fp8_conversion_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/byte_conversion_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conversion16_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conversion64_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conversion32_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bit_count_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/boolean_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bitfield_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_ternary_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fma_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ldexp_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_minmax_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_mul_x86_64_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_add_x86_64_v4.cpp
)

set(goc_avx512bf16_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma_avx512bf16.cpp
)

set(goc_avx512vnni_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_wmma_avx512vnni.cpp
)

# Keep source selection, compile flags and feature definitions together for each ISA.
function(goc_add_x86_64_backend feature)
  if(NOT GOC_COMPILER_${feature})
    return()
  endif()
  string(TOLOWER "${feature}" backend)
  target_sources(goc PRIVATE ${goc_${backend}_sources})
  set_source_files_properties(${goc_${backend}_sources}
    TARGET_DIRECTORY goc
    PROPERTIES COMPILE_OPTIONS "${goc_${backend}_flags}"
  )
  target_compile_definitions(goc PRIVATE GOC_HAVE_${feature}=1)
endfunction()

foreach(feature IN ITEMS X86_64_V3 X86_64_V4 AVX512BF16 AVX512VNNI)
  goc_add_x86_64_backend(${feature})
endforeach()

add_library(goc STATIC
  rdna4_dpp.cpp
  rdna4_permlane.cpp
  rdna4_scalar_pack.cpp
  rdna4_scalar_compare.cpp
  rdna4_scalar_convert.cpp
  rdna4_scalar_round.cpp
  rdna4_scalar_fma.cpp
  rdna4_scalar_fp.cpp
  rdna4_scalar_field.cpp
  rdna4_scalar_bits.cpp
  rdna4_scalar_integer.cpp
  rdna4_rcp_iflag.cpp
  rdna4_pseudo_scalar.cpp
  rdna4_float_compare.cpp
  rdna4_integer_compare.cpp
  rdna4_class.cpp
  rdna4_interp16.cpp
  rdna4_interp32.cpp
  rdna4_cndmask.cpp
  rdna4_trig_preop.cpp
  rdna4_mullit.cpp
  rdna4_pack.cpp
  rdna4_swmmac_integer.cpp
  rdna4_swmmac8.cpp
  rdna4_swmmac_float.cpp
  rdna4_swmmac16.cpp
  rdna4_mad64.cpp
  rdna4_carry.cpp
  rdna4_div_fmas.cpp
  rdna4_div_scale.cpp
  rdna4_div_fixup.cpp
  rdna4_cube.cpp
  rdna4_fp8_narrow.cpp
  rdna4_integer_conversion.cpp
  rdna4_normalized.cpp
  rdna4_packed_conversion.cpp
  rdna4_fp8_conversion.cpp
  rdna4_byte_conversion.cpp
  rdna4_conversion16.cpp
  rdna4_conversion64.cpp
  rdna4_conversion32.cpp
  rdna4_mixed_fma.cpp
  rdna4_bit_count.cpp
  rdna4_boolean.cpp
  rdna4_bitfield.cpp
  rdna4_integer_ternary.cpp
  rdna4_sad.cpp
  rdna4_shift.cpp
  rdna4_half_trig.cpp
  rdna4_trig.cpp
  rdna4_integer_mad.cpp
  rdna4_integer16_ternary.cpp
  rdna4_integer16.cpp
  cpu.cpp
  cpu_decode.cpp
  rdna4_arithmetic.cpp
  rdna4_half_fma.cpp
  rdna4_packed_fma.cpp
  rdna4_unary.cpp
  rdna4_half_unary.cpp
  rdna4_half_exponent.cpp
  rdna4_wmma.cpp
  rdna4_subbyte_wmma.cpp
  rdna4_dot.cpp
  rdna4_integer_dot.cpp
  rdna4_fp8_dot.cpp
  rdna4_half_dot.cpp
  rdna4_binary.cpp
  rdna4_half_binary.cpp
  rdna4_minmax.cpp
  rdna4_half_minmax.cpp
  rdna4_fp64.cpp
  rdna4_frexp_exp.cpp
  rdna4_ldexp.cpp
  rdna4_integer_minmax.cpp
  rdna4_integer_mul.cpp
  rdna4_integer_add.cpp
)

set_target_properties(goc PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(goc
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
if(UNIX)
  target_link_libraries(goc PRIVATE m)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(goc PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
endif()
if(GOC_ENABLE_X86_64)
  target_compile_definitions(goc PRIVATE GOC_X86_64_QUERY=1)
  add_subdirectory(x86_64)
endif()

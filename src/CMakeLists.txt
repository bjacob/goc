add_library(goc STATIC
  scalar_move.cpp
  lane_transfer.cpp
  move.cpp
  wmma_replicated.cpp
  dpp.cpp
  permlane.cpp
  scalar_pack.cpp
  scalar_compare.cpp
  scalar_convert.cpp
  scalar_round.cpp
  scalar_fma.cpp
  scalar_fp.cpp
  scalar_field.cpp
  scalar_bits.cpp
  scalar_integer.cpp
  rcp_iflag.cpp
  pseudo_scalar.cpp
  float_compare.cpp
  integer_compare.cpp
  class.cpp
  interp16.cpp
  interp32.cpp
  cndmask.cpp
  trig_preop.cpp
  mullit.cpp
  pack.cpp
  swmmac_integer.cpp
  swmmac8.cpp
  swmmac_float.cpp
  swmmac16.cpp
  mad64.cpp
  carry.cpp
  div_fmas.cpp
  div_scale.cpp
  div_fixup.cpp
  cube.cpp
  fp8_narrow.cpp
  integer_conversion.cpp
  normalized.cpp
  packed_conversion.cpp
  fp8_conversion.cpp
  byte_conversion.cpp
  conversion16.cpp
  conversion64.cpp
  conversion32.cpp
  mixed_fma.cpp
  bit_count.cpp
  boolean.cpp
  bitfield.cpp
  integer_ternary.cpp
  sad.cpp
  shift.cpp
  half_trig.cpp
  trig.cpp
  integer_mad.cpp
  integer16_ternary.cpp
  integer16.cpp
  cpu.cpp
  cpu_decode.cpp
  arithmetic.cpp
  half_fma.cpp
  packed_fma.cpp
  unary.cpp
  half_unary.cpp
  half_exponent.cpp
  wmma.cpp
  subbyte_wmma.cpp
  dot.cpp
  integer_dot.cpp
  fp8_dot.cpp
  half_dot.cpp
  binary.cpp
  half_binary.cpp
  minmax.cpp
  half_minmax.cpp
  fp64.cpp
  frexp_exp.cpp
  ldexp.cpp
  integer_minmax.cpp
  integer_mul.cpp
  integer_add.cpp
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

execute_process(COMMAND "${NM}" -D --defined-only --format=posix "${LIBRARY}"
  RESULT_VARIABLE status OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Cannot inspect exports: ${error}")
endif()

if(VARIANT STREQUAL "static")
  set(expected consumer_cpu_flags)
else()
  set(expected
    goc_init_cpu_flags
    goc_rdna4_v_fma_f32
    goc_rdna4_v_log_f32
    goc_rdna4_v_dot2_f32_f16
    goc_rdna4_v_dot2_f32_bf16
    goc_rdna4_v_wmma_f32_16x16x16_f16
    goc_rdna4_v_wmma_f32_16x16x16_bf16
    goc_rdna4w64_v_wmma_f32_16x16x16_f16
    goc_rdna4w64_v_wmma_f32_16x16x16_bf16
  )
endif()

set(actual)
string(REPLACE "\n" ";" lines "${symbols}")
foreach(line IN LISTS lines)
  if(NOT line STREQUAL "")
    string(REGEX REPLACE " .*" "" symbol "${line}")
    # libstdc++ explicitly exports some inline std:: functions in Debug builds.
    # Those are outside GoC's visibility contract.
    if(NOT symbol MATCHES "^_ZSt")
      list(APPEND actual "${symbol}")
    endif()
  endif()
endforeach()
list(SORT expected)
list(SORT actual)
if(NOT "${actual}" STREQUAL "${expected}")
  message(FATAL_ERROR "Unexpected ${VARIANT} exports.\nExpected: ${expected}\nActual: ${actual}")
endif()

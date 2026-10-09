execute_process(COMMAND "${NM}" -D --defined-only --format=posix "${LIBRARY}"
  RESULT_VARIABLE status OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Cannot inspect exports: ${error}")
endif()

set(expected consumer_cpu_flags)

set(actual)
string(REPLACE "\n" ";" lines "${symbols}")
foreach(line IN LISTS lines)
  if(NOT line STREQUAL "")
    string(REGEX REPLACE " .*" "" symbol "${line}")
    # libstdc++ explicitly exports some inline std:: functions in Debug builds.
    # ASan also contributes linker-defined bounds for its global registry.
    # These runtime symbols are outside GoC's visibility contract.
    if(NOT symbol MATCHES "^_ZSt|^__(start|stop)_asan_globals$")
      list(APPEND actual "${symbol}")
    endif()
  endif()
endforeach()
list(SORT expected)
list(SORT actual)
if(NOT "${actual}" STREQUAL "${expected}")
  message(FATAL_ERROR "Unexpected consumer exports.\nExpected: ${expected}\nActual: ${actual}")
endif()

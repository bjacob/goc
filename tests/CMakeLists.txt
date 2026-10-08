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
add_executable(goc_cpu_decode_tests cpu_decode_test.cpp ${PROJECT_SOURCE_DIR}/src/cpu_decode.cpp)
target_include_directories(goc_cpu_decode_tests PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(goc_cpu_decode_tests PRIVATE GTest::gtest_main)
goc_test_warnings(goc_cpu_decode_tests)
gtest_discover_tests(goc_cpu_decode_tests TEST_PREFIX "internal.")

foreach(library IN LISTS goc_library_targets)
  string(REPLACE "goc_" "" variant "${library}")
  set(target "goc_tests_${variant}")
  add_executable(${target} arithmetic_test.cpp cpu_test.cpp wmma_test.cpp c_api_test.c)
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(${target} PRIVATE ${library} GTest::gtest_main)
  goc_test_warnings(${target})
  gtest_discover_tests(${target} TEST_PREFIX "${variant}.")

  # Retain the cpuinfo executable name, preferring static when both are enabled.
  if(variant STREQUAL "static" OR NOT GOC_STATIC)
    set(cpuinfo_target cpuinfo)
  else()
    set(cpuinfo_target "cpuinfo_${variant}")
  endif()
  add_executable(${cpuinfo_target} cpuinfo.c)
  target_link_libraries(${cpuinfo_target} PRIVATE ${library})
  goc_test_warnings(${cpuinfo_target})
  add_test(NAME "${variant}.cpuinfo" COMMAND ${cpuinfo_target})

  # Compile each public header first and in isolation in both supported languages.
  foreach(header IN ITEMS common rdna4 umbrella export)
    foreach(extension IN ITEMS c cpp)
      set(source "${CMAKE_CURRENT_BINARY_DIR}/${header}_header_test.${extension}")
      configure_file("headers/${header}.c" "${source}" COPYONLY)
      set(target "goc_${header}_header_${extension}_${variant}_test")
      add_executable(${target} "${source}")
      target_link_libraries(${target} PRIVATE ${library})
      goc_test_warnings(${target})
      add_test(NAME "${variant}.${header}_header_${extension}" COMMAND ${target})
    endforeach()
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

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

foreach(library IN LISTS goc_library_targets)
  string(REPLACE "goc_" "" variant "${library}")
  foreach(test IN ITEMS arithmetic cpu wmma)
    set(target "goc_${test}_test_${variant}")
    add_executable(${target} "${test}_test.cpp")
    if(test STREQUAL "arithmetic")
      target_sources(${target} PRIVATE c_api_test.c)
    endif()
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${target} PRIVATE ${library} GTest::gtest_main)
    goc_test_warnings(${target})
    gtest_discover_tests(${target} TEST_PREFIX "${variant}.")
  endforeach()
endforeach()

if(GOC_STATIC)
  add_executable(cpuinfo cpuinfo.c)
  target_link_libraries(cpuinfo PRIVATE goc_static)
  goc_test_warnings(cpuinfo)
  add_test(NAME static.cpuinfo COMMAND cpuinfo)
endif()

# Compile each public header independently as C and C++, without linking.
foreach(header IN ITEMS common rdna4 umbrella export)
  foreach(extension IN ITEMS c cpp)
    set(source "${CMAKE_CURRENT_BINARY_DIR}/${header}_header_test.${extension}")
    configure_file("headers/${header}.c" "${source}" COPYONLY)
    set(target "goc_${header}_header_${extension}_test")
    add_library(${target} OBJECT "${source}")
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
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

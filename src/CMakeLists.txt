add_library(goc cpu.cpp arithmetic.cpp wmma.cpp dot.cpp)

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
  target_compile_definitions(goc PRIVATE GOC_X86_QUERY=1)
  add_subdirectory(x86_64)
endif()

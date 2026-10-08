include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-march=x86-64-v3" GOC_COMPILER_AVX2)
check_cxx_compiler_flag("-march=x86-64-v4" GOC_COMPILER_AVX512F)

if(GOC_COMPILER_AVX2)
  target_sources(goc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fma_avx2.cpp)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/fma_avx2.cpp TARGET_DIRECTORY goc PROPERTIES COMPILE_OPTIONS "-march=x86-64-v3")
  target_compile_definitions(goc PRIVATE GOC_HAVE_AVX2=1)
endif()

if(GOC_COMPILER_AVX512F)
  target_sources(goc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fma_avx512f.cpp)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/fma_avx512f.cpp TARGET_DIRECTORY goc PROPERTIES COMPILE_OPTIONS "-march=x86-64-v4")
  target_compile_definitions(goc PRIVATE GOC_HAVE_AVX512F=1)
endif()

check_cxx_compiler_flag("-march=x86-64-v4 -mavx512bf16" GOC_COMPILER_AVX512BF16)
if(GOC_COMPILER_AVX512BF16)
  target_sources(goc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/wmma_avx512bf16.cpp)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/wmma_avx512bf16.cpp TARGET_DIRECTORY goc PROPERTIES COMPILE_OPTIONS "-march=x86-64-v4;-mavx512bf16")
  target_compile_definitions(goc PRIVATE GOC_HAVE_AVX512BF16=1)
endif()

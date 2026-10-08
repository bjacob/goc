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

foreach(target IN LISTS goc_library_targets)
  if(GOC_COMPILER_X86_64_V3)
    set(goc_x86_64_v3_sources
      ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fma_x86_64_v3.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_wmma_x86_64_v3.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_wmma_x86_64_v3.cpp
    )
    target_sources(${target} PRIVATE ${goc_x86_64_v3_sources})
    set_source_files_properties(${goc_x86_64_v3_sources} TARGET_DIRECTORY ${target} PROPERTIES COMPILE_OPTIONS "${goc_x86_64_v3_flags}")
    target_compile_definitions(${target} PRIVATE GOC_HAVE_X86_64_V3=1)
  endif()

  if(GOC_COMPILER_X86_64_V4)
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fma_x86_64_v4.cpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/rdna4_fma_x86_64_v4.cpp TARGET_DIRECTORY ${target} PROPERTIES COMPILE_OPTIONS "${goc_x86_64_v4_flags}")
    target_compile_definitions(${target} PRIVATE GOC_HAVE_X86_64_V4=1)
  endif()

  if(GOC_COMPILER_AVX512BF16)
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_wmma_avx512bf16.cpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/rdna4_wmma_avx512bf16.cpp TARGET_DIRECTORY ${target} PROPERTIES COMPILE_OPTIONS "${goc_avx512bf16_flags}")
    target_compile_definitions(${target} PRIVATE GOC_HAVE_AVX512BF16=1)
  endif()

  if(GOC_COMPILER_AVX512VNNI)
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_wmma_avx512vnni.cpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/rdna4_integer_wmma_avx512vnni.cpp TARGET_DIRECTORY ${target} PROPERTIES COMPILE_OPTIONS "${goc_avx512vnni_flags}")
    target_compile_definitions(${target} PRIVATE GOC_HAVE_AVX512VNNI=1)
  endif()
endforeach()

set(pp_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(cuda)

list(APPEND pp_SRC 
    ${pp_SRC_DIR}/image/image.cpp 
    ${pp_SRC_DIR}/util/flags.cpp 
    ${pp_SRC_DIR}/util/conf.cpp
)

add_executable(
    pp 
    ${pp_SRC_DIR}/main.cpp 
    ${pp_SRC}
)

list(APPEND pp_IMPL_SRC ${pp_SRC_DIR}/proc/cpu_proc.cpp)

if (pp_USE_CUDA)
    list(APPEND pp_LINKER_LIBS PUBLIC pp_cuda)
    list(APPEND pp_IMPL_SRC ${pp_SRC_DIR}/proc/cuda_proc.cpp)
    list(APPEND pp_DEF PP_USE_CUDA)
else()
    list(APPEND pp_IMPL_SRC ${pp_SRC_DIR}/proc/empty_impl/no_cuda_proc.cpp)
endif()

if (pp_USE_HALIDE)
    list(APPEND pp_IMPL_SRC ${pp_SRC_DIR}/proc/halide_proc.cpp)
    list(APPEND pp_DEF PP_USE_HALIDE)
else()
    list(APPEND pp_IMPL_SRC ${pp_SRC_DIR}/proc/empty_impl/no_halide_proc.cpp)
endif()

target_sources(pp PRIVATE ${pp_IMPL_SRC})
target_link_libraries(pp ${pp_LINKER_LIBS})
target_include_directories(pp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${pp_INCLUDE_DIRS})
target_compile_definitions(pp PUBLIC ${pp_DEF})

add_subdirectory(tests)
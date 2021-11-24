add_subdirectory(cuda)

add_executable(
    pp 
    main.cpp 
    image/image.cpp 
    util/flags.cpp 
    util/conf.cpp
)

list(APPEND pp_IMPL_SRC proc/cpu_proc.cpp)

if (pp_USE_CUDA)
    list(APPEND pp_LINKER_LIBS PUBLIC pp_cuda)
    list(APPEND pp_IMPL_SRC proc/cuda_proc.cpp)
else()
    list(APPEND pp_IMPL_SRC proc/empty_impl/no_cuda_proc.cpp)
endif()

if (pp_USE_HALIDE)
    list(APPEND pp_IMPL_SRC proc/halide_proc.cpp)
else()
    list(APPEND pp_IMPL_SRC proc/empty_impl/no_halide_proc.cpp)
endif()

target_sources(pp PRIVATE ${pp_IMPL_SRC})
target_link_libraries(pp ${pp_LINKER_LIBS})
target_include_directories(pp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${pp_INCLUDE_DIRS})
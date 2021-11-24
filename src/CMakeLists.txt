add_subdirectory(cuda)

add_executable(
    pp 
    main.cpp 
    image/image.cpp 
    util/flags.cpp 
    util/conf.cpp
    proc/cpu_proc.cpp
)
if (pp_USE_CUDA)
    list(APPEND pp_LINKER_LIBS PUBLIC pp_cuda)
endif()
target_link_libraries(pp ${pp_LINKER_LIBS})
target_include_directories(pp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${pp_INCLUDE_DIRS})
add_executable(
    pp 
    main.cpp 
    image/image.cpp 
    util/flags.cpp 
    util/conf.cpp
    proc/cpu_proc.cpp
)
target_link_libraries(pp ${pp_LINKER_LIBS})
target_include_directories(pp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${pp_INCLUDE_DIRS})
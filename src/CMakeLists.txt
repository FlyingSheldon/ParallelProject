add_executable(
    pp 
    main.cpp 
    image/image.cpp 
    util/flags.cpp 
    util/conf.cpp
)
target_link_libraries(pp ${pp_LINKER_LIBS})
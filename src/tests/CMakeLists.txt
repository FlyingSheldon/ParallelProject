add_executable(pp_test cuda_test.cpp halide_test.cpp ${pp_SRC} ${pp_IMPL_SRC})
target_link_libraries(pp_test PUBLIC gtest_main ${pp_LINKER_LIBS})
target_include_directories(pp_test PUBLIC ${pp_SRC_DIR} ${pp_INCLUDE_DIRS})

if (pp_USE_CUDA)
    list(APPEND pp_TEST_DEF PP_USE_CUDA)
endif()

if (pp_USE_HALIDE)
    list(APPEND pp_TEST_DEF PP_USE_HALIDE)
    add_executable(halide_test main.cpp ${pp_SRC} ${pp_IMPL_SRC})
    target_link_libraries(halide_test PUBLIC ${pp_LINKER_LIBS})
    target_include_directories(halide_test PUBLIC ${pp_SRC_DIR} ${pp_INCLUDE_DIRS})
    target_compile_definitions(halide_test PUBLIC ${pp_TEST_DEF})
endif()

target_compile_definitions(pp_test PUBLIC ${pp_TEST_DEF})

gtest_add_tests(TARGET pp_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})


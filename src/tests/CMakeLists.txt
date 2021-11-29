add_executable(pp_test cuda_test.cpp ${pp_SRC} ${pp_IMPL_SRC})
target_link_libraries(pp_test PUBLIC gtest_main ${pp_LINKER_LIBS})
target_include_directories(pp_test PUBLIC ${pp_SRC_DIR} ${pp_INCLUDE_DIRS})

gtest_add_tests(TARGET pp_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
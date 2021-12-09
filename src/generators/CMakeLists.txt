add_executable(pp_generators halide_generators.cpp)
target_link_libraries(pp_generators PRIVATE Halide::Generator)
target_include_directories(pp_generators PUBLIC ${pp_SRC_DIR})
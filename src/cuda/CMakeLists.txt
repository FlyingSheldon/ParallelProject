check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(pp_USE_CUDA true PARENT_SCOPE)

    find_package(CUDA REQUIRED)

    add_library(pp_cuda STATIC cuda_proc.cu proc.cu img_helper.cu)
    set_target_properties(pp_cuda PROPERTIES CUDA_ARCHITECTURES "75")
    set_target_properties(pp_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(pp_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(pp_cuda PROPERTIES CUDA_STANDARD 11)

    target_include_directories(pp_cuda PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(pp_cuda PRIVATE ${CUDA_LIBRARIES} cuda)
else()
    message("CUDA is not supported")
endif()
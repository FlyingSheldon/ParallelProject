check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    # set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CC_COMPILER})
    enable_language(CUDA)
    set(pp_USE_CUDA true PARENT_SCOPE)

    find_package(CUDA REQUIRED)

    add_library(pp_cuda STATIC cuda_proc.cu proc.cu)
    set_target_properties(pp_cuda PROPERTIES CUDA_ARCHITECTURES "75")
    set_target_properties(pp_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(pp_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_include_directories(pp_cuda PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(pp_cuda PRIVATE ${CUDA_LIBRARIES} cuda)
else()
    message("CUDA is not supported")
endif()
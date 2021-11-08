cmake_minimum_required(VERSION 3.0.0)
project(ParallelProject VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)

find_package(gflags REQUIRED)

include(CTest)
enable_testing()

add_executable(imgproc main.cpp)
target_link_libraries(imgproc gflags::gflags)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

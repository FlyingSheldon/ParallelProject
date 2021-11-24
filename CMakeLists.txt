cmake_minimum_required(VERSION 3.13.0)
project(pp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)
include(GNUInstallDirs)
include(CheckLanguage)

include(cmake/Dependencies.cmake)

add_subdirectory(src)

# find_package(gflags REQUIRED)

# include(CTest)
# enable_testing()

# add_executable(imgproc main.cpp)
# target_link_libraries(imgproc gflags)

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)

cmake_minimum_required(VERSION 3.14.0)
project(pp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)
include(GNUInstallDirs)
include(CheckLanguage)

include(cmake/Dependencies.cmake)

enable_testing()
include(GoogleTest)

add_subdirectory(src)
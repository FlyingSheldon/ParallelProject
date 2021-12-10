set(pp_LINKER_LIBS "")
set(pp_INCLUDE_DIRS "")
set(pp_COMPILE_OPTIONS "")

include(FetchContent)

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND pp_LINKER_LIBS PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# ---[ Google-gflags
# include("cmake/External/gflags.cmake")
# list(APPEND pp_INCLUDE_DIRS PUBLIC ${GFLAGS_INCLUDE_DIRS})
# list(APPEND pp_LINKER_LIBS PUBLIC ${GFLAGS_LIBRARIES})

FetchContent_Declare(
  gflags
  URL https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz
)

FetchContent_MakeAvailable(gflags)
list(APPEND pp_LINKER_LIBS PUBLIC gflags::gflags)

# ---[ googletest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---[ Halide
find_package(LLVMD CONFIG PATHS /usr/local/Cellar/)
find_package(Halide)
if( Halide_FOUND )
    find_package(PNG REQUIRED)
    list(APPEND pp_LINKER_LIBS PRIVATE Halide::Halide Halide::Tools ${PNG_LIBRARY})
    list(APPEND pp_INCLUDE_DIRS PRIVATE ${PNG_INCLUDE_DIR})
    set(pp_USE_HALIDE true)
else()
    message("Halide is not found on this system")
endif()

# ---[ JPEG
find_package(JPEG)
list(APPEND pp_INCLUDE_DIRS PUBLIC ${JPEG_INCLUDE_DIRS})
list(APPEND pp_LINKER_LIBS PUBLIC ${JPEG_LIBRARIES})

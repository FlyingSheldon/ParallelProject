set(pp_LINKER_LIBS "")
set(pp_INCLUDE_DIRS "")
set(pp_COMPILE_OPTIONS "")

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND pp_LINKER_LIBS PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# ---[ Google-gflags
include("cmake/External/gflags.cmake")
list(APPEND pp_INCLUDE_DIRS PUBLIC ${GFLAGS_INCLUDE_DIRS})
list(APPEND pp_LINKER_LIBS PUBLIC ${GFLAGS_LIBRARIES})

# ---[ Halide
find_package(LLVM REQUIRED CONFIG PATHS /usr/local/Cellar/)
find_package(Halide REQUIRED)
list(APPEND pp_LINKER_LIBS PRIVATE Halide::Halide)

# ---[ JPEG
find_package(JPEG)
list(APPEND pp_INCLUDE_DIRS PUBLIC ${JPEG_INCLUDE_DIRS})
list(APPEND pp_LINKER_LIBS PUBLIC ${JPEG_LIBRARIES})

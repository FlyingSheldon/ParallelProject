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
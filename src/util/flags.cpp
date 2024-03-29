#include "flags.h"
#include <iostream>

DEFINE_bool(h, false, "Show help message");
DEFINE_double(brightness, 1.0, "Set brightness(0.0 - )");
DEFINE_string(o, "", "Output path");
DEFINE_double(sharpness, 0.0, "Set sharpness(0.0 - 1.0)");
DEFINE_string(impl, "linear",
              "SET image processing implementation(linear, halide, cuda)");
DEFINE_bool(time, false, "Measure time of each step");
DEFINE_int32(schedule, 0, "Choose the scheduler");
DEFINE_bool(gpu, false, "Enable GPU for halide implementation");
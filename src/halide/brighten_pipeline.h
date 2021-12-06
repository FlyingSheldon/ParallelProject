#pragma once
#include "halide/find_target.h"
#include <Halide.h>

class BrightenPipeline {
public:
  Halide::Func brighten;
  Halide::Var x, y, c;
  Halide::Var xOut, xIn;
  Halide::Var yOut, yIn;

  BrightenPipeline(Halide::Buffer<uint8_t> &hImg, double value)
      : x("x"), y("y"), c("c"), xOut("x_o"), xIn("x_i"), yOut("y_o"),
        yIn("y_i") {
    Halide::Expr v = hImg(x, y, c);

    v = Halide::cast<float>(v);

    v = v * static_cast<float>(value);

    v = Halide::min(v, 255.0f);

    v = Halide::cast<uint8_t>(v);

    brighten(x, y, c) = v;
  }

  void ScheduleForCpu() {
    brighten.split(x, xOut, xIn, 4);
    brighten.vectorize(xIn);
    brighten.parallel(y);

    Halide::Target target = Halide::get_host_target();
    brighten.compile_jit(target);
  }

  bool ScheduleForGpu() {
    Halide::Target target = findGpuTarget();
    if (!target.has_gpu_feature()) {
      printf("No GPU supported");
      return false;
    }

    Halide::Var fused, block, thread;

    brighten.reorder(c, x, y).bound(c, 0, 3).unroll(c);

    brighten.fuse(x, y, fused);

    brighten.gpu_tile(fused, block, thread, 256);

    // brighten.gpu_tile(x, y, xOut, yOut, xIn, yIn, 16, 16);

    brighten.compile_jit(target);

    return true;
  }
};

#include "proc/halide_func.h"
#include <Halide.h>

using namespace Halide;

class SharpenGenerator : public Halide::Generator<SharpenGenerator> {
public:
  Input<double> scale{"scale"};
  Input<Buffer<uint8_t>> input{"input", 3};
  Output<Buffer<uint8_t>> sharpen{"sharpen", 3};

  void generate() {}
};

HALIDE_REGISTER_GENERATOR(SharpenGenerator, sharpen_generator)
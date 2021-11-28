#include "conf.h"
#include "flags.h"
#include <cmath>

Conf::Conf(int argc, char **argv) {
  if (argc < 2) {
    confError = "No input file specified";
    return;
  }
  input = std::string(argv[1]);

  output = FLAGS_o;
  if (output.empty()) {
    confError = "No output file specified";
    return;
  }

  sharpness = FLAGS_sharpness;
  if (sharpness > 1.0) {
    sharpness = 1.0;
  } else if (sharpness < 0.0) {
    sharpness = 0.0;
  }

  brightness = std::fabs(FLAGS_brightness);

  if (FLAGS_impl == "cuda") {
    impl = Impl::CUDA;
  } else if (FLAGS_impl == "halide") {
    impl = Impl::HALIDE;
  } else {
    impl = Impl::LINEAR;
  }
}
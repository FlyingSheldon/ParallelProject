#include "proc/halide_func.h"
#include <halide_image_io.h>

Halide::Buffer<uint8_t> LoadImage(std::string filename) {
  Halide::Buffer<uint8_t> hImg = Halide::Tools::load_image(filename);
  return hImg;
}
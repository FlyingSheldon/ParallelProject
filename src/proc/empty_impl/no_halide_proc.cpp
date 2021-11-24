#include "proc/halide_proc.h"
#include <cstdlib>
#include <iostream>

HalideImageProc::HalideImageProc() {
  std::cerr << "Halide not supported in this binary" << std::endl;
  exit(1);
}

void HalideImageProc::Brighten(Image &img, double value) {}

void HalideImageProc::Sharpen(Image &img, double value) {}
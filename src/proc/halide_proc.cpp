#include "proc/halide_proc.h"
#include <iostream>

HalideImageProc::HalideImageProc() {}

void HalideImageProc::Brighten(Image &img, double value) {
  std::cout << "Brightened by Halide" << std::endl;
}

void HalideImageProc::Sharpen(Image &img, double value) {
  std::cout << "Brightened by Halide" << std::endl;
}

bool HalideImageProc::IsSupported() const { return true; }
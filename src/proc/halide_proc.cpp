#include "proc/halide_proc.h"
#include <iostream>

HalideImageProc::HalideImageProc() {}

void HalideImageProc::Brighten(double value) {
  std::cout << "Brightened by Halide" << std::endl;
}

void HalideImageProc::Sharpen(double value) {
  std::cout << "Brightened by Halide" << std::endl;
}

bool HalideImageProc::IsSupported() const { return true; }
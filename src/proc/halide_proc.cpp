#include "proc/halide_proc.h"
#include <halide_image_io.h>
#include <iostream>

HalideImageProc::HalideImageProc() {}

void HalideImageProc::Brighten(double value) {
  std::cout << "Brightened by Halide" << std::endl;
}

void HalideImageProc::Sharpen(double value) {
  std::cout << "Brightened by Halide" << std::endl;
}

bool HalideImageProc::IsSupported() const { return true; }

ImageProc::ImageIOResult HalideImageProc::LoadImage(std::string filename) {
  hImg = Halide::Tools::load_image(filename);
  return {};
}

ImageProc::ImageIOResult HalideImageProc::SaveImage(std::string filename) {
  Halide::Tools::save_image(hImg, filename);
  return {};
}
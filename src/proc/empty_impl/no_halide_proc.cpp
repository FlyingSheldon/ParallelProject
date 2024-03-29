#include "proc/halide_proc.h"
#include <cstdlib>
#include <iostream>

HalideImageProc::HalideImageProc() {}

void HalideImageProc::Brighten(double value) {}

void HalideImageProc::Sharpen(double value) {}

bool HalideImageProc::IsSupported() const { return false; }

ImageProc::ImageIOResult HalideImageProc::LoadImage(std::string filename) {
  return Image::ImageError("not implemented");
}

ImageProc::ImageIOResult HalideImageProc::SaveImage(std::string filename) {
  return Image::ImageError("not implemented");
}

Image *HalideImageProc::GetImage() { return nullptr; }

void HalideImageProc::PrepareBrighten(double value) {}

void HalideImageProc::PrepareSharpen(double value) {}
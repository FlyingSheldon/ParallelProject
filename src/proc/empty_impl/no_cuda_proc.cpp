#include "proc/cuda_proc.h"
#include <cstdlib>
#include <iostream>

CudaImageProc::CudaImageProc() {}

void CudaImageProc::Brighten(double value) {}

void CudaImageProc::Sharpen(double value) {}

bool CudaImageProc::IsSupported() const { return false; }

ImageProc::ImageIOResult CudaImageProc::LoadImage(std::string filename) {
  return Image::ImageError("not implemented");
}

ImageProc::ImageIOResult CudaImageProc::SaveImage(std::string filename) {
  return Image::ImageError("not implemented");
}
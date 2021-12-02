#include "proc/cuda_proc.h"
#include <cstdlib>
#include <iostream>

CudaImageProc::CudaImageProc() {}

void CudaImageProc::Brighten(double value) {
  std::cout << "Brightened by CUDA" << std::endl;
}

void CudaImageProc::Sharpen(double value) {
  std::cout << "Brightened by CUDA" << std::endl;
}

bool CudaImageProc::IsSupported() const { return true; }

ImageProc::ImageIOResult CudaImageProc::LoadImage(std::string filename) {
  std::variant<Image, Image::ImageError> res = Image::OpenImage(filename);

  if (const Image::ImageError *error = std::get_if<Image::ImageError>(&res)) {
    return *error;
  }

  img = std::move(std::get<Image>(res));

  return {};
}

ImageProc::ImageIOResult CudaImageProc::SaveImage(std::string filename) {
  return img.Save(filename);
}
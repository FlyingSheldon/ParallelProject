#include "proc/cuda_proc.h"
#include "cuda/cuda_proc.h"
#include <cstdlib>
#include <iostream>

CudaImageProc::CudaImageProc() {}

void CudaImageProc::Brighten(double value) {
  cudaBrighten(img.GetPixelData(0, 0), img.GetHeight() * img.GetWidth(),
               img.GetPixelSize(), value);
}

void CudaImageProc::Sharpen(double value) {
  cudaSharpen(img.GetPixelData(0, 0), img.GetPixelSize(), img.GetWidth(),
              img.GetHeight(), value, 0.07, 2);
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

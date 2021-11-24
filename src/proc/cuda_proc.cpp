#include "proc/cuda_proc.h"
#include "cuda/cuda_proc.h"
#include <cstdlib>
#include <iostream>

CudaImageProc::CudaImageProc() {}

void CudaImageProc::Brighten(Image &img, double value) {
  cudaBrighten(img.GetPixelData(0, 0), img.GetHeight() * img.GetWidth(),
               img.GetPixelSize(), value);
}

void CudaImageProc::Sharpen(Image &img, double value) {
  std::cout << "Brightened by CUDA" << std::endl;
}
#include "proc/cuda_proc.h"
#include <cstdlib>
#include <iostream>

CudaImageProc::CudaImageProc() {}

void CudaImageProc::Brighten(Image &img, double value) {
  std::cout << "Brightened by CUDA" << std::endl;
}

void CudaImageProc::Sharpen(Image &img, double value) {
  std::cout << "Brightened by CUDA" << std::endl;
}
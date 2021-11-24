#include "proc/cuda_proc.h"
#include <cstdlib>
#include <iostream>

CudaImageProc::CudaImageProc() {
  std::cerr << "CUDA not supported in this binary" << std::endl;
  exit(1);
}

void CudaImageProc::Brighten(Image &img, double value) {}

void CudaImageProc::Sharpen(Image &img, double value) {}
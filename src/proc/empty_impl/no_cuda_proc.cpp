#include "proc/cuda_proc.h"
#include <cstdlib>
#include <iostream>

CudaImageProc::CudaImageProc() {}

void CudaImageProc::Brighten(Image &img, double value) {}

void CudaImageProc::Sharpen(Image &img, double value) {}

bool CudaImageProc::IsSupported() const { return false; }
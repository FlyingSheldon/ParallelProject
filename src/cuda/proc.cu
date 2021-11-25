#include "proc.cuh"
#include <cstdio>

__global__ void sayHi() {
  int i = threadIdx.x;
  printf("CUDA from %d\n", i);
}

__global__ void brighten(uint8_t *img, size_t size, size_t pixelSize,
                         double value) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= size * pixelSize) {
    return;
  }

  uint8_t *px = img + i;
  *px = static_cast<uint8_t>(fminf(255.0, static_cast<double>(*px) * value));
}
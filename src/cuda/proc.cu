#include "proc.cuh"
#include <cstdio>

__global__ void sayHi() {
  int i = threadIdx.x;
  printf("CUDA from %d\n", i);
}

__global__ void brighten(uint8_t *img, size_t size, size_t pixelSize,
                         double value) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  uint8_t *px = img + i * pixelSize;
  for (size_t c = 0; c < pixelSize; c++) {
    px[c] =
        static_cast<uint8_t>(fminf(255.0, static_cast<double>(px[c]) * value));
  }
}
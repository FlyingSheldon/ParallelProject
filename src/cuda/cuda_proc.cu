#include "cuda_proc.h"
#include "proc.cuh"

void cudaSayHi() { sayHi<<<1, 2>>>(); }

void cudaBrighten(uint8_t *img, size_t size, size_t pixelSize, double value) {
  uint8_t *deviceImg;
  cudaMalloc(&deviceImg, size * pixelSize);

  cudaMemcpy(deviceImg, img, size * pixelSize, cudaMemcpyHostToDevice);

  size_t blocksPerGrid = (size + kThreadPerBlock - 1) / kThreadPerBlock;
  brighten<<<blocksPerGrid, kThreadPerBlock>>>(deviceImg, size, pixelSize,
                                               value);
  cudaMemcpy(img, deviceImg, size * pixelSize, cudaMemcpyDeviceToHost);
  cudaFree(deviceImg);
}

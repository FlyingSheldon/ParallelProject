#include "cuda_proc.h"
#include "proc.cuh"

void cudaSayHi() { sayHi<<<1, 2>>>(); }

void cudaBrighten(uint8_t *img, size_t size, size_t pixelSize, double value) {
  uint8_t *deviceImg;
  cudaMalloc(&deviceImg, size * pixelSize * sizeof(uint8_t));

  cudaMemcpy(deviceImg, img, size * pixelSize * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  size_t blocksPerGrid =
      (size * pixelSize + kThreadPerBlock - 1) / kThreadPerBlock;
  brighten<<<blocksPerGrid, kThreadPerBlock>>>(deviceImg, size, pixelSize,
                                               value);
  cudaMemcpy(img, deviceImg, size * pixelSize * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);
  cudaFree(deviceImg);
}

void cudaSharpen(uint8_t *img, size_t pixelSize, size_t width, size_t height,
                 double value, double eth, int lpf) {
  size_t size = width * height;
  uint8_t *deviceImg;
  size_t blockPerGrid = (size + kThreadPerBlock - 1) / kThreadPerBlock;

  double *deviceHsv;
  ValueMinMaxSum *deviceVmms;
  ValueMinMaxSum initialVmms;
  initialVmms.max = 0;
  initialVmms.min = 255;
  initialVmms.sum = 0;
  cudaMalloc(&deviceImg, size * pixelSize * sizeof(uint8_t));
  cudaMalloc(&deviceHsv, size * 3 * sizeof(double));
  cudaMalloc(&deviceVmms, sizeof(ValueMinMaxSum));

  cudaMemcpy(deviceImg, img, size * pixelSize * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceVmms, &initialVmms, sizeof(ValueMinMaxSum),
             cudaMemcpyHostToDevice);

  dim3 blockDim(kBlockEdgeSize, kBlockEdgeSize, 1);
  dim3 gridDim((width + (blockDim.x - 2) - 1) / (blockDim.x - 2),
               (height + (blockDim.y - 2) - 1) / (blockDim.y - 2));

  rgbToHsvAndDeltaReduce<kThreadPerBlock><<<blockPerGrid, kThreadPerBlock>>>(
      deviceImg, deviceHsv, size, deviceVmms);

  cudaDeviceSynchronize();

  edgeSharpen<<<gridDim, blockDim>>>(deviceHsv, width, height, eth, lpf, value,
                                     nullptr, deviceVmms, deviceImg);

  // // just for debug
  // writeEdgeToImage<<<blockPerGrid, kThreadPerBlock>>>(deviceImg, deviceEdges,
  //                                                     size);

  cudaMemcpy(img, deviceImg, size * pixelSize * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);

  cudaFree(deviceImg);
  cudaFree(deviceHsv);
  cudaFree(deviceVmms);
}

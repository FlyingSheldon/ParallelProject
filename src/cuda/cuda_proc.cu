#include "cuda_proc.h"
#include "proc.cuh"
#include <cstdio>
#include <vector>

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

void cudaEdgeLPFDbg(uint8_t *edges, uint8_t *output, size_t width,
                    size_t height, int lpf) {
  size_t size = width * height;
  uint8_t *deviceEdges, *deviceLPF;

  cudaMalloc(&deviceEdges, size * sizeof(uint8_t));
  cudaMalloc(&deviceLPF, size * sizeof(uint8_t));

  cudaMemcpy(deviceEdges, edges, size * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  dim3 blockDim(kBlockEdgeSize, kBlockEdgeSize, 1);
  dim3 gridDim((width + (blockDim.x - 2) - 1) / (blockDim.x - 2),
               (height + (blockDim.y - 2) - 1) / (blockDim.y - 2));

  edgeLPFDbg<<<gridDim, blockDim>>>(deviceEdges, deviceLPF, width, height, lpf);

  cudaMemcpy(output, deviceLPF, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(edges, deviceEdges, size, cudaMemcpyDeviceToHost);
}

void cudaEdgeDetect(uint8_t *img, uint8_t *edges, size_t pixelSize,
                    size_t width, size_t height, double eth, double *hsv) {
  size_t size = width * height;
  uint8_t *deviceImg, *deviceEdges;
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
  cudaMalloc(&deviceEdges, size * sizeof(uint8_t));

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

  edgeDetect<<<gridDim, blockDim>>>(deviceHsv, deviceEdges, width, height, eth);

  cudaMemcpy(edges, deviceEdges, size * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);

  if (hsv) {
    cudaMemcpy(hsv, deviceHsv, size * pixelSize * sizeof(double),
               cudaMemcpyDeviceToHost);
  }
}

void cudaSharpen(uint8_t *img, size_t pixelSize, size_t width, size_t height,
                 double value, double eth, int lpf, double *hsv) {
  size_t size = width * height;
  uint8_t *deviceImg, *deviceEdges;
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
  cudaMalloc(&deviceEdges, size * sizeof(uint8_t));

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

  edgeDetect<<<gridDim, blockDim>>>(deviceHsv, deviceEdges, width, height, eth);

  cudaDeviceSynchronize();

  edgeSharpen<<<gridDim, blockDim>>>(deviceHsv, width, height, value, eth, lpf,
                                     deviceEdges, deviceVmms, deviceImg);

  cudaMemcpy(img, deviceImg, size * pixelSize * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hsv, deviceHsv, size * pixelSize * sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(deviceImg);
  cudaFree(deviceEdges);
  cudaFree(deviceHsv);
  cudaFree(deviceVmms);
}

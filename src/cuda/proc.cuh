#pragma once
#include "img_helper.cuh"
#include <cstdint>

constexpr size_t kBlockEdgeSize = 16;
__device__ const size_t kDBlockEdgeSize = 16;
constexpr size_t kThreadPerBlock = kBlockEdgeSize * kBlockEdgeSize;

struct ValueMinMaxSum {
  unsigned long long min, max, sum;
};

__global__ void sayHi();

__global__ void brighten(uint8_t *img, size_t size, size_t pixelSize,
                         double value);

__global__ void sharpen(uint8_t *img, size_t size, size_t pixelSize,
                        size_t width, size_t height, double value, double eth,
                        double lpf);

__global__ void edgeSharpen(double *hsv, size_t width, size_t height,
                            double value, double eth, int lpf, uint8_t *edges,
                            ValueMinMaxSum *vmms, uint8_t *img);

__global__ void writeEdgeToImage(uint8_t *img, uint8_t *edges, size_t size);

__device__ inline void minMaxSum(volatile ValueMinMaxSum &lhs,
                                 volatile ValueMinMaxSum &rhs) {
  lhs.max = max(lhs.max, lhs.max);
  lhs.min = min(lhs.min, rhs.min);
  lhs.sum += rhs.sum;
}

__device__ inline void minMaxSum(ValueMinMaxSum &lhs, ValueMinMaxSum &rhs) {
  lhs.max = max(lhs.max, lhs.max);
  lhs.min = min(lhs.min, rhs.min);
  lhs.sum += rhs.sum;
}

template <unsigned int blockSize>
__device__ void warpReduceMinMaxSum(volatile ValueMinMaxSum *sdata,
                                    unsigned int tid) {
  if (blockSize >= 64) {
    minMaxSum(sdata[tid], sdata[tid + 32]);
  }
  if (blockSize >= 32) {
    minMaxSum(sdata[tid], sdata[tid + 16]);
  }
  if (blockSize >= 16) {
    minMaxSum(sdata[tid], sdata[tid + 8]);
  }
  if (blockSize >= 8) {
    minMaxSum(sdata[tid], sdata[tid + 4]);
  }
  if (blockSize >= 4) {
    minMaxSum(sdata[tid], sdata[tid + 2]);
  }
  if (blockSize >= 2) {
    minMaxSum(sdata[tid], sdata[tid + 1]);
  }
}

template <unsigned int blockSize>
__global__ void rgbToHsvAndDeltaReduce(uint8_t *rgb, double *hsv, size_t size,
                                       ValueMinMaxSum *vmms) {
  __shared__ ValueMinMaxSum smms[blockSize];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + tid;

  uchar3 *rgbVec = reinterpret_cast<uchar3 *>(rgb);
  double3 *hsvVec = reinterpret_cast<double3 *>(hsv);

  uchar3 pRgb = rgbVec[i];
  int v = max(pRgb.x, max(pRgb.y, pRgb.z));
  smms[tid].min = v;
  smms[tid].max = v;
  smms[tid].sum = v;

  if (blockSize >= 512) {
    if (tid < 256) {
      minMaxSum(smms[tid], smms[tid + 256]);
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      minMaxSum(smms[tid], smms[tid + 128]);
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      minMaxSum(smms[tid], smms[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < 32) {
    warpReduceMinMaxSum<blockSize>(smms, tid);
  }

  if (tid == 0) {
    atomicMax(&vmms->max, smms[0].max);
    atomicMin(&vmms->min, smms[0].min);
    atomicAdd(&vmms->sum, smms[0].sum);
  }

  hsvVec[i] = rgbToHsv(pRgb);
}
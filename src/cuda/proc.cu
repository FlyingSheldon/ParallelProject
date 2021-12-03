#include "img_helper.cuh"
#include "proc.cuh"
#include <cstdint>
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

__global__ void sharpen(uint8_t *img, size_t size, size_t pixelSize,
                        size_t width, size_t height, double value, double eth,
                        double lpf) {}

__device__ inline double calcDelta(ValueMinMaxSum *vmms, size_t size) {
  double cmax = static_cast<double>(vmms->max) / 255.0;
  double cmin = static_cast<double>(vmms->min) / 255.0;
  double cavg =
      static_cast<double>(vmms->sum) / static_cast<double>(size) / 255.0;
  double cmid = (cmax + cmin) / 2.0;

  return ((cmax / 8.0) * (cavg / cmid));
}

__device__ inline double3 getHsvPixel(double3 *hsvVec, int px, int py,
                                      size_t width, size_t height) {
  if (px >= 0 && py >= 0 && px < width && py < height) {
    return hsvVec[py * width + px];
  } else if (px < 0 && py >= 0) {
    return hsvVec[py * width + px + 1];
  } else if (py < 0 && px >= 0) {
    return hsvVec[(py + 1) * width + px];
  } else {
    return make_double3(-1.0, -1.0, -1.0);
  }
}

__device__ inline uint8_t getEdgePixel(uint8_t *edges, int px, int py,
                                       size_t width, size_t height) {
  if (px >= 0 && py >= 0 && px < width && py < height) {
    return edges[py * width + px];
  } else {
    return 0;
  }
}

__device__ inline bool pixelEdgeDetect(double3 *localHsvGrid, size_t tx,
                                       size_t ty, double eth) {
  return ((fabs(localHsvGrid[ty * kBlockEdgeSize + tx].z -
                localHsvGrid[ty * kBlockEdgeSize + tx - 1].z) >= eth) ||
          (fabs(localHsvGrid[ty * kBlockEdgeSize + tx].z -
                localHsvGrid[(ty - 1) * kBlockEdgeSize + tx].z) >= eth));
}

__device__ inline bool pixelLowPassFilter(uint8_t *localEdges, size_t tx,
                                          size_t ty, int lpf, size_t width,
                                          size_t height, int px, int py) {
  static constexpr int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  static constexpr int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
  int cnt = 0;
  for (int j = 0; j < 8; j++) {
    int xx = (int)tx + dx[j];
    int yy = (int)ty + dy[j];
    cnt += localEdges[yy * kBlockEdgeSize + xx];
  }
  return cnt >= lpf;
}

__device__ inline double pixelLocalMean(double3 *localHsvGrid, size_t tx,
                                        size_t ty, size_t width, size_t height,
                                        int px, int py) {
  static constexpr int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  static constexpr int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
  int cnt = 1;
  double sum = localHsvGrid[ty * kBlockEdgeSize + tx].z;

  for (int j = 0; j < 8; j++) {
    int cpx = px + dx[j];
    int cpy = py + dy[j];
    if (cpx >= 0 && cpy >= 0 && cpx < width && cpy < height) {
      cnt++;
      sum += localHsvGrid[(ty + dy[j]) * kBlockEdgeSize + (tx + dx[j])].z;
    }
  }

  return sum / static_cast<double>(cnt);
}

__device__ inline double boundValue(double v) {
  return fmax(fmin(v, 1.0), 0.0);
}

__global__ void edgeDetect(double *hsv, uint8_t *edges, size_t width,
                           size_t height, double eth) {

  static_assert(kBlockEdgeSize >= 4,
                "kDBlcokEdgeSize has to be larger than or equal to 4");

  __shared__ double3 localHsvGrid[kBlockEdgeSize * kBlockEdgeSize];
  constexpr size_t realGridEdgeSize = kBlockEdgeSize - 2;

  double3 *hsvVec = reinterpret_cast<double3 *>(hsv);

  const size_t tx = threadIdx.x;
  const size_t ty = threadIdx.y;

  const int px = static_cast<int>(blockIdx.x * realGridEdgeSize + tx) - 1;
  const int py = static_cast<int>(blockIdx.y * realGridEdgeSize + ty) - 1;

  const int pxIdx = py * width + px;

  localHsvGrid[ty * kBlockEdgeSize + tx] =
      getHsvPixel(hsvVec, px, py, width, height);

  __syncthreads();

  if (tx > 0 && ty > 0 && localHsvGrid[ty * kBlockEdgeSize + tx].z != -1.0) {
    edges[pxIdx] = pixelEdgeDetect(localHsvGrid, tx, ty, eth) ? 1 : 0;
  }
}

__global__ void edgeLPFDbg(uint8_t *edges, uint8_t *output, size_t width,
                           size_t height, int lpf) {
  static_assert(kBlockEdgeSize >= 4,
                "kDBlcokEdgeSize has to be larger than or equal to 4");

  __shared__ uint8_t localEdges[kBlockEdgeSize * kBlockEdgeSize];
  __shared__ uint8_t localLF[kBlockEdgeSize * kBlockEdgeSize];
  constexpr size_t realGridEdgeSize = kBlockEdgeSize - 2;

  const size_t tx = threadIdx.x;
  const size_t ty = threadIdx.y;

  const int px = static_cast<int>(blockIdx.x * realGridEdgeSize + tx) - 1;
  const int py = static_cast<int>(blockIdx.y * realGridEdgeSize + ty) - 1;

  const int pxIdx = py * width + px;

  localEdges[ty * kBlockEdgeSize + tx] =
      getEdgePixel(edges, px, py, width, height);

  __syncthreads();

  if (tx > 0 && ty > 0 && tx < kBlockEdgeSize - 1 && ty < kBlockEdgeSize - 1 &&
      localEdges[ty * kBlockEdgeSize + tx] == 1) {
    localLF[ty * kBlockEdgeSize + tx] =
        pixelLowPassFilter(localEdges, tx, ty, lpf, width, height, px, py) ? 1
                                                                           : 0;
  } else {
    localLF[ty * kBlockEdgeSize + tx] = 0;
  }

  __syncthreads();

  if (tx > 0 && ty > 0 && tx < kBlockEdgeSize - 1 && ty < kBlockEdgeSize - 1 &&
      px < width && py < height) {
    output[pxIdx] = localLF[ty * kBlockEdgeSize + tx];
  }
}

__global__ void edgeSharpen(double *hsv, size_t width, size_t height,
                            double value, double eth, int lpf, uint8_t *edges,
                            ValueMinMaxSum *vmms, uint8_t *img) {

  static_assert(kBlockEdgeSize >= 4,
                "kDBlcokEdgeSize has to be larger than or equal to 4");

  __shared__ double3 localHsvGrid[kBlockEdgeSize * kBlockEdgeSize];
  __shared__ uint8_t localEdges[kBlockEdgeSize * kBlockEdgeSize];
  __shared__ uint8_t localLF[kBlockEdgeSize * kBlockEdgeSize];
  constexpr size_t realGridEdgeSize = kBlockEdgeSize - 2;

  double3 *hsvVec = reinterpret_cast<double3 *>(hsv);
  uchar3 *imgVec = reinterpret_cast<uchar3 *>(img);

  const size_t tx = threadIdx.x;
  const size_t ty = threadIdx.y;

  const int px = static_cast<int>(blockIdx.x * realGridEdgeSize + tx) - 1;
  const int py = static_cast<int>(blockIdx.y * realGridEdgeSize + ty) - 1;

  const int pxIdx = py * width + px;
  double3 pHsv;

  // double delta = calcDelta(vmms, width * height);
  double delta = 0.076665;

  localHsvGrid[ty * kBlockEdgeSize + tx] =
      getHsvPixel(hsvVec, px, py, width, height);

  localEdges[ty * kBlockEdgeSize + tx] =
      getEdgePixel(edges, px, py, width, height);

  __syncthreads();

  if (tx > 0 && ty > 0 && tx < kBlockEdgeSize - 1 && ty < kBlockEdgeSize - 1 &&
      localEdges[ty * kBlockEdgeSize + tx] == 1) {
    localLF[ty * kBlockEdgeSize + tx] =
        pixelLowPassFilter(localEdges, tx, ty, lpf, width, height, px, py) ? 1
                                                                           : 0;
  } else {
    localLF[ty * kBlockEdgeSize + tx] = localEdges[ty * kBlockEdgeSize + tx];
  }

  __syncthreads();

  if (tx > 0 && ty > 0 && tx < kBlockEdgeSize - 1 && ty < kBlockEdgeSize - 1 &&
      localLF[ty * kBlockEdgeSize + tx] == 1) {
    double localMean =
        pixelLocalMean(localHsvGrid, tx, ty, width, height, px, py);
    pHsv = localHsvGrid[ty * kBlockEdgeSize + tx];
    double v = pHsv.z;
    double factor = v < localMean ? (-v) / localMean : localMean / v;

    pHsv.z = boundValue(v + value * delta * factor);
  } else {
    pHsv = localHsvGrid[ty * kBlockEdgeSize + tx];
  }

  if (tx > 0 && ty > 0 && tx < kBlockEdgeSize - 1 && ty < kBlockEdgeSize - 1 &&
      px >= 0 && py >= 0 && px < width && py < height) {
    imgVec[pxIdx] = hsvToRgb(pHsv);
  }
}

__global__ void writeEdgeToImage(uint8_t *img, uint8_t *edges, size_t size) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size) {
    return;
  }

  if (edges[idx] == 1) {
    img[idx * 3] = 255;
    img[idx * 3 + 1] = 255;
    img[idx * 3 + 2] = 255;
  } else {
    img[idx * 3] = 0;
    img[idx * 3 + 1] = 0;
    img[idx * 3 + 2] = 0;
  }
}
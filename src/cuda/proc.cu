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

__global__ void edgeSharpen(double *hsv, size_t width, size_t height,
                            double eth, int lpf, uint8_t *edges) {
  static constexpr int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  static constexpr int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
  static_assert(kBlockEdgeSize >= 4,
                "kDBlcokEdgeSize has to be larger than or equal to 4");

  __shared__ double3 localHsvGrid[kBlockEdgeSize * kBlockEdgeSize];
  __shared__ uint8_t localEdges[kBlockEdgeSize * kBlockEdgeSize];
  constexpr size_t realGridEdgeSize = kBlockEdgeSize - 2;

  double3 *hsvVec = reinterpret_cast<double3 *>(hsv);

  const size_t tx = threadIdx.x;
  const size_t ty = threadIdx.y;

  const int px = static_cast<int>(blockIdx.x * realGridEdgeSize + tx) - 1;
  const int py = static_cast<int>(blockIdx.y * realGridEdgeSize + ty) - 1;

  const int pxIdx = py * width + px;

  if (px >= 0 && py >= 0 && px < width && py < height) {
    localHsvGrid[ty * kBlockEdgeSize + tx] = hsvVec[pxIdx];
  } else if (px < 0 && py >= 0) {
    localHsvGrid[ty * kBlockEdgeSize + tx] = hsvVec[py * width + px + 1];
  } else if (py < 0 && px >= 0) {
    localHsvGrid[ty * kBlockEdgeSize + tx] = hsvVec[(py + 1) * width + px];
  } else {
    localHsvGrid[ty * kBlockEdgeSize + tx] = make_double3(-1.0, -1.0, -1.0);
  }

  __syncthreads();

  if (tx > 0 && ty > 0 && localHsvGrid[ty * kBlockEdgeSize + tx].z != -1.0) {
    localEdges[ty * kBlockEdgeSize + tx] =
        ((fabs(localHsvGrid[ty * kBlockEdgeSize + tx].z -
               localHsvGrid[ty * kBlockEdgeSize + tx - 1].z) >= eth) ||
         (fabs(localHsvGrid[ty * kBlockEdgeSize + tx].z -
               localHsvGrid[(ty - 1) * kBlockEdgeSize + tx].z) >= eth))
            ? 1
            : 0;

  } else {
    localEdges[ty * kBlockEdgeSize + tx] = 0;
  }

  __syncthreads();

  if (tx > 0 && ty > 0 && tx < kBlockEdgeSize - 1 && ty < kBlockEdgeSize - 1 &&
      localEdges[ty * kBlockEdgeSize + tx] == 1) {
    int cnt = 0;
    for (int j = 0; j < 8; j++) {
      int xx = tx + dx[j];
      int yy = ty + dy[j];
      cnt += localEdges[yy * kBlockEdgeSize + xx];
    }
    localEdges[ty * kBlockEdgeSize + tx] = (cnt >= lpf) ? 1 : 0;
  }

  if (tx > 0 && ty > 0 && tx < kBlockEdgeSize && ty < kBlockEdgeSize &&
      px < width && py < height) {
    hsvVec[pxIdx] = localHsvGrid[ty * kBlockEdgeSize + tx];
    edges[pxIdx] = localEdges[ty * kBlockEdgeSize + tx];
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
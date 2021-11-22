#include "proc.h"
#include <limits>
#include <math.h>   

static inline uint8_t truncDown(int value) {
  return static_cast<uint8_t>(
      std::min(value, static_cast<int>(std::numeric_limits<uint8_t>::max())));
}

static inline uint8_t truncUp(int value) {
  return static_cast<uint8_t>(std::max(value, 0));
}

static void brigtenImpl(Image &image, int value) {
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      uint8_t *px = image.GetPixelData(x, y);
      px[0] = truncDown(static_cast<int>(px[0]) + value);
      px[1] = truncDown(static_cast<int>(px[1]) + value);
      px[2] = truncDown(static_cast<int>(px[2]) + value);
    }
  }
}

static void darkenImpl(Image &image, int value) {
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      uint8_t *px = image.GetPixelData(x, y);
      px[0] = truncUp(static_cast<int>(px[0]) - value);
      px[1] = truncUp(static_cast<int>(px[1]) - value);
      px[2] = truncUp(static_cast<int>(px[2]) - value);
    }
  }
}

namespace linear {

void brighten(Image &image, int value) {
  if (value > 0) {
    brigtenImpl(image, value);
  } else {
    darkenImpl(image, -value);
  }
}

} // namespace linear

void rgbToHsv(Image &image) {
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      image.RGB2HSV(x, y);
    }
  }
}

void hsvToRgb(Image &image) {
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      image.HSV2RGB(x, y);
    }
  }
}

bool edgeDetectPixel(Image &image, size_t x, size_t y, double eth) {
  bool ans = false;
  double value = image.GetValue(x, y);


  if (x >= 1) {
    double value_west = image.GetValue(x - 1, y);
    if (abs(value - value_west) >= eth) {
      ans = true;
    }
  }
  
  if (y >= 1) {
    double value_north = image.GetValue(x, y - 1);

    if (abs(value - value_north) >= eth) {
      ans = true;
    }
  }

  return ans;
}

std::vector<bool> edgeDetect(Image &image, double eth) {
  std::vector<bool> g(image.GetHeight() * image.GetWidth(), false);
  for (size_t y = 0; y < image.GetHeight(); y++) {
    for (size_t x = 0; x < image.GetWidth(); x++) {
      int index = y * image.GetWidth() + x;
      if (edgeDetectPixel(image, x, y, eth)) {
        g[index] = true;
        // printf("add edge x:%zu, y:%zu\n", x, y);
      }
    }
  }
  return g;

}

bool lowPassFilterPixel(Image &image, std::vector<bool> g, size_t x, size_t y, int lpf) {
  int cnt = 0;
  for (int i = 0; i < 8; i++) {
    size_t newX = x + dx[i];
    size_t newY = y + dy[i];
    if (newX >= 0 && newX < image.GetWidth() && newY >= 0 && newY < image.GetHeight()) {
      int index = newY * image.GetWidth() + newX;
      if (g[index]) cnt++;
    }
  }
  return cnt >= lpf;
}

void lowPassFilter(Image &image, std::vector<bool> &g, int lpf) {
  printf("in lpf\n");

  std::vector<bool> g_copy(g);
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      int index = y * image.GetWidth() + x;
      if (g_copy[index] && !lowPassFilterPixel(image, g_copy, x, y, lpf)) {
        g[index] = false;     // remove isolated pixels
        // printf("remove x:%d, y:%d\n", x, y);
      }
    }
  }
}

double additiveMaginitude(Image &image) {
  double max = std::numeric_limits<double>::min();
  double min = std::numeric_limits<double>::max();
  double total = 0;

  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      const double value = image.GetValue(x, y);
      max = std::max(max, value);
      min = std::min(min, value);
      total += value;
    }
  }

  double mid = (max + min) / 2.0;
  double avg = (total / (double) image.GetHeight()) / (double) image.GetWidth();

  double delta = (max / 8.0) * (avg / mid);
  printf("max:%f, mid:%f, avg:%f, delta: %f\n", max, mid, avg, delta);
  return delta;
}

double computelocalMean(Image &image, size_t x, size_t y) {
  double localSum = image.GetValue(x, y);
  int localCnt = 1;
  for (int i = 0; i < 8; i++) {
    size_t newX = x + dx[i];
    size_t newY = y + dy[i];
    if (newX >= 0 && newX < image.GetWidth() && newY >= 0 && newY < image.GetHeight()) {
      localSum += image.GetValue(newX, newY);
      localCnt ++;
    }
  }
  double localMean = localSum / (double) localCnt;
  return localMean;
}

void edgeSharpen(Image &image, std::vector<bool> g, double s, double delta) {
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      int index = y * image.GetWidth() + x;
      if (!g[index]) continue;  // nonedge pixels are kept unaltered

      double localMean = computelocalMean(image, x, y);
      double *value = image.GetValueData(x, y);

      double factor = *value < localMean ? (- *value) / localMean : localMean / *value;
      double value_change = s * delta * factor;
      *value = *value + value_change;

      if (*value > 1) {
        *value = 1;
      }

      if (*value < 0) {
        *value = 0;
      }
    }
  }
}
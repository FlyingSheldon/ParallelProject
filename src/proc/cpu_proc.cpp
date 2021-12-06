#include "proc.h"
#include <limits>
#include <math.h>

void LinearImageProc::Brighten(float value) { linear::brighten(img, value); }

void LinearImageProc::Sharpen(float value) { linear::sharpen(img, value); }

ImageProc::ImageIOResult LinearImageProc::LoadImage(std::string filename) {
  std::variant<Image, Image::ImageError> res = Image::OpenImage(filename);

  if (const Image::ImageError *error = std::get_if<Image::ImageError>(&res)) {
    return *error;
  }

  img = std::move(std::get<Image>(res));

  return {};
}

ImageProc::ImageIOResult LinearImageProc::SaveImage(std::string filename) {
  return img.Save(filename);
}

namespace linear {

void brighten(Image &image, float value) {
  for (size_t y = 0; y < image.GetHeight(); y++) {
    for (size_t x = 0; x < image.GetWidth(); x++) {
      uint8_t *p = image.GetPixelData(x, y);
      p[0] = static_cast<uint8_t>(
          std::min(255.0f, static_cast<float>(p[0]) * value));
      p[1] = static_cast<uint8_t>(
          std::min(255.0f, static_cast<float>(p[1]) * value));
      p[2] = static_cast<uint8_t>(
          std::min(255.0f, static_cast<float>(p[2]) * value));
    }
  }
}

void sharpen(Image &img, float value) {
  float eth = 0.07;
  int lpf = 2;
  rgbToHsv(img);
  std::vector<bool> g = edgeDetect(img, eth);
  lowPassFilter(img, g, lpf);
  float delta = additiveMaginitude(img);
  edgeSharpen(img, g, value, delta);
  hsvToRgb(img);
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

bool edgeDetectPixel(Image &image, size_t x, size_t y, float eth) {
  bool ans = false;
  float value = image.GetValue(x, y);

  if (x >= 1) {
    float value_west = image.GetValue(x - 1, y);
    if (abs(value - value_west) >= eth) {
      ans = true;
    }
  }

  if (y >= 1) {
    float value_north = image.GetValue(x, y - 1);

    if (abs(value - value_north) >= eth) {
      ans = true;
    }
  }

  return ans;
}

std::vector<bool> edgeDetect(Image &image, float eth) {
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

bool lowPassFilterPixel(Image &image, std::vector<bool> &g, size_t x, size_t y,
                        int lpf) {
  int cnt = 0;
  for (int i = 0; i < 8; i++) {
    size_t newX = x + dx[i];
    size_t newY = y + dy[i];
    if (newX >= 0 && newX < image.GetWidth() && newY >= 0 &&
        newY < image.GetHeight()) {
      int index = newY * image.GetWidth() + newX;
      if (g[index])
        cnt++;
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
        g[index] = false; // remove isolated pixels
        // printf("remove x:%d, y:%d\n", x, y);
      }
    }
  }
}

float additiveMaginitude(Image &image) {
  float max = std::numeric_limits<float>::min();
  float min = std::numeric_limits<float>::max();
  float total = 0;

  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      const float value = image.GetValue(x, y);
      max = std::max(max, value);
      min = std::min(min, value);
      total += value;
    }
  }

  float mid = (max + min) / 2.0;
  float avg = (total / (float)image.GetHeight()) / (float)image.GetWidth();

  float delta = (max / 8.0) * (avg / mid);
  printf("max:%f, mid:%f, avg:%f, delta: %f\n", max, mid, avg, delta);
  return delta;
}

float computelocalMean(Image &image, size_t x, size_t y) {
  float localSum = image.GetValue(x, y);
  int localCnt = 1;
  for (int i = 0; i < 8; i++) {
    size_t newX = x + dx[i];
    size_t newY = y + dy[i];
    if (newX >= 0 && newX < image.GetWidth() && newY >= 0 &&
        newY < image.GetHeight()) {
      localSum += image.GetValue(newX, newY);
      localCnt++;
    }
  }
  float localMean = localSum / (float)localCnt;
  return localMean;
}

void edgeSharpen(Image &image, std::vector<bool> &g, float s, float delta) {
  std::vector<float> hsvCopy(image.GetHeight() * image.GetWidth());
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      int index = y * image.GetWidth() + x;
      if (!g[index]) {
        hsvCopy[y * image.GetWidth() + x] = *image.GetValueData(x, y);
        continue; // nonedge pixels are kept unaltered
      }

      float localMean = computelocalMean(image, x, y);
      const float *value = image.GetValueData(x, y);

      float factor =
          *value < localMean ? (-*value) / localMean : localMean / *value;
      float value_change = s * delta * factor;

      hsvCopy[y * image.GetWidth() + x] =
          std::max(std::min(*value + value_change, 1.0f), 0.0f);
    }
  }
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      *image.GetValueData(x, y) = hsvCopy[y * image.GetWidth() + x];
    }
  }
}
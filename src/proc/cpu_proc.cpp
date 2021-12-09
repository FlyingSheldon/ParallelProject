#include "proc.h"
#include <limits>
#include <math.h>

void LinearImageProc::Brighten(double value) { linear::brighten(img, value); }

void LinearImageProc::Sharpen(double value) { linear::sharpen(img, value); }

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

void brighten(Image &image, double value) {
  for (size_t y = 0; y < image.GetHeight(); y++) {
    for (size_t x = 0; x < image.GetWidth(); x++) {
      uint8_t *p = image.GetPixelData(x, y);
      p[0] = static_cast<uint8_t>(
          std::min(255.0, static_cast<double>(p[0]) * value));
      p[1] = static_cast<uint8_t>(
          std::min(255.0, static_cast<double>(p[1]) * value));
      p[2] = static_cast<uint8_t>(
          std::min(255.0, static_cast<double>(p[2]) * value));
    }
  }
}

void sharpen(Image &img, double value) {
  double eth = 0.07;
  int lpf = 2;
  rgbToHsv(img);
  std::vector<bool> g = edgeDetect(img, eth);
  lowPassFilter(img, g, lpf);
  double delta = additiveMaginitude(img);
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
  std::vector<bool> g_copy(g);
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      int index = y * image.GetWidth() + x;
      if (g_copy[index] && !lowPassFilterPixel(image, g_copy, x, y, lpf)) {
        g[index] = false; // remove isolated pixels
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
  double avg = (total / (double)image.GetHeight()) / (double)image.GetWidth();

  double delta = (max / 8.0) * (avg / mid);
  return delta;
}

double computelocalMean(Image &image, size_t x, size_t y) {
  double localSum = image.GetValue(x, y);
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
  double localMean = localSum / (double)localCnt;
  return localMean;
}

void edgeSharpen(Image &image, std::vector<bool> &g, double s, double delta) {
  std::vector<double> hsvCopy(image.GetHeight() * image.GetWidth());
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      int index = y * image.GetWidth() + x;
      if (!g[index]) {
        hsvCopy[y * image.GetWidth() + x] = *image.GetValueData(x, y);
        continue; // nonedge pixels are kept unaltered
      }

      double localMean = computelocalMean(image, x, y);
      const double *value = image.GetValueData(x, y);

      float factor;
      if (abs(*value - localMean) < 1e-4) {
        factor = 1;
      } else {
        factor =
            *value < localMean ? (-*value) / localMean : localMean / *value;
      }
      double value_change = s * delta * factor;

      hsvCopy[y * image.GetWidth() + x] =
          std::max(std::min(*value + value_change, 1.0), 0.0);
    }
  }
  for (int y = 0; y < image.GetHeight(); y++) {
    for (int x = 0; x < image.GetWidth(); x++) {
      *image.GetValueData(x, y) = hsvCopy[y * image.GetWidth() + x];
    }
  }
}
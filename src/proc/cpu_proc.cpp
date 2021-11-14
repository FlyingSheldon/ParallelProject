#include "proc.h"
#include <limits>

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
#pragma once
#include "image/image.h"

class ImageProc {
public:
  virtual void Brigten(Image &image, int value);
  virtual void Sharpen(Image &image, int value);
};

namespace linear {
void brighten(Image &image, int value);
void sharpen(Image &image, int value);
} // namespace linear
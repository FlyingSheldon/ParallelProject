#pragma once
#include "proc/proc.h"

class HalideImageProc : public ImageProc {
public:
  HalideImageProc();
  virtual void Brighten(Image &image, double value) override;
  virtual void Sharpen(Image &image, double value) override;
};
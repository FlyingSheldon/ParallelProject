#pragma once
#include "proc/proc.h"

class HalideImageProc : public ImageProc {
public:
  HalideImageProc();
  virtual void Brighten(Image &image, double value) override;
  virtual void Sharpen(Image &image, double value) override;
  virtual bool IsSupported() const override;
  virtual std::string Name() const override { return "halide"; }
};
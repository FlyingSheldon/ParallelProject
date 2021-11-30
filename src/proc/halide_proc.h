#pragma once
#include "proc/proc.h"

class HalideImageProc : public ImageProc {
public:
  HalideImageProc();
  virtual void Brighten(double value) override;
  virtual void Sharpen(double value) override;
  virtual bool IsSupported() const override;
  virtual std::string Name() const override { return "halide"; }
};
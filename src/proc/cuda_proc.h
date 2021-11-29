#pragma once
#include "proc/proc.h"

class CudaImageProc : public ImageProc {
public:
  CudaImageProc();
  virtual void Brighten(Image &image, double value) override;
  virtual void Sharpen(Image &image, double value) override;
  virtual bool IsSupported() const override;
  virtual std::string Name() const override { return "cuda"; }
};
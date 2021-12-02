#pragma once
#include "proc/proc.h"

#ifdef PP_USE_HALIDE
#include <Halide.h>
#endif

class HalideImageProc : public ImageProc {
public:
  HalideImageProc();
  virtual void Brighten(double value) override;
  virtual void Sharpen(double value) override;
  virtual bool IsSupported() const override;
  virtual std::string Name() const override { return "halide"; }
  virtual ImageIOResult LoadImage(std::string filename) override;
  virtual ImageIOResult SaveImage(std::string filename) override;
  virtual Image *GetImage() override;

#ifdef PP_USE_HALIDE
private:
  Halide::Buffer<uint8_t> hImg;
  Image img;
#endif
};
#pragma once
#include "proc/proc.h"

class CudaImageProc : public ImageProc {
public:
  CudaImageProc();
  virtual void Brighten(double value) override;
  virtual void Sharpen(double value) override;
  virtual bool IsSupported() const override;
  virtual std::string Name() const override { return "cuda"; }
  virtual ImageIOResult LoadImage(std::string filename) override;
  virtual ImageIOResult SaveImage(std::string filename) override;
  virtual Image *GetImage() override { return &img; }

private:
  Image img;
};
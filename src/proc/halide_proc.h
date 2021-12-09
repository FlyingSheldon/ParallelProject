#pragma once
#include "proc/proc.h"
#include <memory>

#ifdef PP_USE_HALIDE
#include "proc/halide_func.h"
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
  virtual void PrepareBrighten(double value) override;
  virtual void PrepareSharpen(double value) override;

#ifdef PP_USE_HALIDE
  virtual Halide::Buffer<float> rgbToHsv();
  virtual void hsvToRgb();
  virtual Halide::Buffer<uint8_t> edgeDetect(double eth);
  virtual Halide::Buffer<uint8_t> lowPassFilter(Halide::Buffer<uint8_t> g,
                                                int lpf);
  virtual Halide::Buffer<float> additiveMaginitude();
  virtual Halide::Buffer<float> edgeSharpen(Halide::Buffer<uint8_t> g, double s,
                                            Halide::Buffer<float> delta);

  virtual Halide::Func LoadImageFunc(std::string filename);
  virtual Halide::Func rgbToHsvFunc(Halide::Func input);

private:
  Halide::Buffer<uint8_t> hImg;
  Halide::Buffer<float> hHSV;

  std::unique_ptr<SharpenPipeline> sharpenPipeline;
  double sharpenParam = -1.0;

  std::unique_ptr<BrightenPipeline> brightenPipeline;
  double brightenParam = -1.0;

  Image img;
#endif
};
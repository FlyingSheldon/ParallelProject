#include "proc/halide_proc.h"
#include "halide/brighten_pipeline.h"
#include <halide_image_io.h>
#include <iostream>

HalideImageProc::HalideImageProc(bool gpu) : useGpu(gpu) {}

void HalideImageProc::Brighten(double value) {
  BrightenPipeline p(hImg, value);
  bool onGpu = false;

  if (useGpu && p.ScheduleForGpu()) {
    onGpu = true;
  } else {
    p.ScheduleForCpu();
    onGpu = false;
  }

  Halide::Buffer<uint8_t> output =
      p.brighten.realize({hImg.width(), hImg.height(), hImg.channels()});

  if (onGpu) {
    output.copy_to_host();
  }

  // p.brighten.print_loop_nest();

  std::swap(hImg, output);
}

void HalideImageProc::Sharpen(double value) {
  std::cout << "Brightened by Halide" << std::endl;
}

bool HalideImageProc::IsSupported() const { return true; }

ImageProc::ImageIOResult HalideImageProc::LoadImage(std::string filename) {
  hImg = Halide::Tools::load_image(filename);
  return {};
}

ImageProc::ImageIOResult HalideImageProc::SaveImage(std::string filename) {
  Halide::Tools::save_image(hImg, filename);
  return {};
}

Image *HalideImageProc::GetImage() {
  img = Image(hImg.width(), hImg.height(), 3, J_COLOR_SPACE::JCS_RGB);
  auto buf = hImg.get();
  size_t r = 0;
  size_t g = hImg.width() * hImg.height() * 1;
  size_t b = hImg.width() * hImg.height() * 2;
  const uint8_t *ptr = hImg.get()->begin();
  for (int y = 0; y < img.GetHeight(); y++) {
    for (int x = 0; x < img.GetWidth(); x++) {
      uint8_t *p = img.GetPixelData(x, y);
      p[0] = ptr[r++];
      p[1] = ptr[g++];
      p[2] = ptr[b++];
    }
  }
  return &img;
}
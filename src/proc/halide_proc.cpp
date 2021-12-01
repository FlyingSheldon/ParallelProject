#include "proc/halide_proc.h"
#include <halide_image_io.h>
#include <iostream>

HalideImageProc::HalideImageProc() {}

void HalideImageProc::Brighten(double value) {
  Halide::Func brighter;

  Halide::Var x, y, c;

  Halide::Expr v = hImg(x, y, c);

  v = Halide::cast<float>(v);

  v = v * static_cast<float>(value);

  v = Halide::min(v, 255.0f);

  v = Halide::cast<uint8_t>(v);

  brighter(x, y, c) = v;

  Halide::Buffer<uint8_t> output =
      brighter.realize({hImg.width(), hImg.height(), hImg.channels()});

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
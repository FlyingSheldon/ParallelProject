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

Halide::Buffer<float> HalideImageProc::rgbToHsv() {
  Halide::Func max_ch, min_ch, diff, hsv;
  Halide::Var x, y, c;


  Halide::Expr R = hImg(x, y, 0) / 255.0f;
  Halide::Expr G = hImg(x, y, 1) / 255.0f;
  Halide::Expr B = hImg(x, y, 2) / 255.0f;

  max_ch(x, y) = Halide::max(R, G, B);
  max_ch.trace_stores();
  min_ch(x, y) = Halide::min(R, G, B);
  diff(x ,y) = max_ch(x, y) - min_ch(x, y);

  Halide::Expr V = max_ch(x, y);
  Halide::Expr C = diff(x, y);

  Halide::Expr H = Halide::select(C == 0, 0, 
                                  R == V && G >= B, 60 * (0 + (G - B) / C),
                                  R == V && G < B, 60 * (6 + (G - B) / C),
                                  G == V, 60 * (2 + (B - R) / C), 60 * (4 + (R - G) / C));

  Halide::Expr S = Halide::select(V == 0, 0, C / V);

  hsv(x, y, c) = Halide::select(c == 0, H,
                        c == 1, S, V);

  hsv.bound(c, 0, 3)
    .reorder(c, x, y)
    .unroll(c, 3);

  Halide::Buffer<float> result = hsv.realize({hImg.width(), hImg.height(), 3});
  
  std::swap(hHSV, result);
  return hHSV;
}

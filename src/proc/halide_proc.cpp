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

void HalideImageProc::hsvToRgb() {
  Halide::Func rgb;
  Halide::Var x, y, c;

  Halide::Expr H = hHSV(x, y, 0);
  Halide::Expr S = hHSV(x, y, 1);
  Halide::Expr V = hHSV(x, y, 2);

  Halide::Expr i = H / 60.0f;
  i = Halide::cast<int>(i);
  i = i % 6;

  Halide::Expr i_float = Halide::cast<float>(i);
  Halide::Expr f = H / 60.0f - i_float;
  Halide::Expr p = V * (1.0f - S);
  Halide::Expr q = V * (1.0f - f * S);
  Halide::Expr t = V * (1.0f - (1.0f - f) * S);
  
  Halide::Expr r = Halide::select(i == 0 || i == 5, V,
                                  i == 1, q,
                                  i == 2 || i == 3, p, t);

  Halide::Expr g = Halide::select(i == 0, t,
                                  i == 1 || i == 2, V,
                                  i == 3, q, p);
  
  Halide::Expr b = Halide::select(i == 0 || i == 1, p,
                                  i == 2, t,
                                  i == 3 || i == 4, V, q);
  

  r = r * 255.0f;
  r = Halide::cast<uint8_t>(r);
  g = g * 255.0f;
  g = Halide::cast<uint8_t>(g);
  b = b * 255.0f;
  b = Halide::cast<uint8_t>(b);
  
  rgb(x, y, c) = Halide::select(c == 0, r,
                                c == 1, g, b);
  
  rgb.bound(c, 0, 3)
    .reorder(c, x, y)
    .unroll(c, 3);

  Halide::Buffer<uint8_t> result = rgb.realize({hHSV.width(), hHSV.height(), 3});
  std::swap(hImg, result);

}

Halide::Buffer<uint8_t> HalideImageProc::edgeDetect(double eth) {
  Halide::Func edge;
  Halide::Var x, y, c;

  // 
  Halide::Func hsv;
  hsv(x, y, c) = hHSV(x, y, c);
  hsv.trace_loads();

  // clamp
  Halide::Func clamped;
  Halide::Expr clamped_x = Halide::clamp(x, 0, hHSV.width() - 1);
  Halide::Expr clamped_y = Halide::clamp(x, 0, hHSV.height() - 1);
  clamped(x, y, c) = hHSV(clamped_x, clamped_y, c);

  Halide::Expr one = 1;
  one = Halide::cast<uint8_t>(one);
  Halide::Expr zero = 0;
  zero = Halide::cast<uint8_t>(zero);
  Halide::Expr eth_float = (float)eth;
  eth_float = Halide::cast<float>(eth_float);

  Halide::Expr local = clamped(x, y, 2);
  Halide::Expr west = clamped(x - 1, y, 2);
  Halide::Expr north = clamped(x, y - 1, 2);

  local = Halide::print_when(x == 48 && y == 0, local, "<- this is local at x, y == (0, 0)" );
  west = Halide::print_when(x == 48 && y == 0, west, "<- this is west at x, y == (0, 0)" );
  north = Halide::print_when(x == 48 && y == 0, north, "<- this is north at x, y == (0, 0)" );
  

  edge(x, y) = Halide::select(Halide::abs(local - west) >= eth_float || 
                Halide::abs(local - north) >= eth_float, one, zero);

  clamped.trace_loads();


  // local = Halide::print_when(x == 0 && y == 0, local, "<- this is local at x, y == (0, 0)");

  // Halide::Func test;
  // test(x, y) = Halide::select( Halide::abs(hHSV(x, y, 2) - hHSV(x - 1, y, 2)) >= (float) eth || 
  //                             Halide::abs(hHSV(x, y, 2) - hHSV(x, y - 1, 2)) >= (float) eth, 1, 0);

  // Halide::Expr west = Halide::select(x >= 1, hHSV(x - 1, y, 2), local + (float)eth + 1.0f);
  // west = Halide::print_when(x == 0 && y == 0, west, "<- this is west at x, y == (0, 0)");

  // Halide::Expr north = Halide::select(y >= 1, hHSV(x, y - 1, 2), local + (float)eth + 1.0f);
  // Halide::Expr abs_west = Halide::abs(local - west);
  // Halide::Expr abs_north = Halide::abs(local - north);
  
  // edge(x, y) = Halide::select(abs_north >= (float)eth || abs_west >= (float)eth, 1, 0);
  std::cout << "6" << std::endl;

  Halide::Buffer<uint8_t> result = edge.realize({hHSV.width(), hHSV.height(), 1});
  std::cout << "7" << std::endl;
  return result;
}

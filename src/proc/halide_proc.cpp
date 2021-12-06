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
  // Halide::Func hsv;
  // hsv(x, y, c) = hHSV(x, y, c);
  // hsv.trace_loads();

  // clamp
  Halide::Func clamped;
  Halide::Expr clamped_x = Halide::clamp(x, 0, hHSV.width() - 1);
  Halide::Expr clamped_y = Halide::clamp(y, 0, hHSV.height() - 1);
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

  // local = Halide::print_when(x == 48 && y == 0, local, "<- this is local at x, y == (0, 0)" );
  // west = Halide::print_when(x == 48 && y == 0, west, "<- this is west at x, y == (0, 0)" );
  // north = Halide::print_when(x == 48 && y == 0, north, "<- this is north at x, y == (0, 0)" );
  

  edge(x, y) = Halide::select(Halide::abs(local - west) >= eth_float || 
                Halide::abs(local - north) >= eth_float, one, zero);

  Halide::Buffer<uint8_t> result = edge.realize({hHSV.width(), hHSV.height(), 1});
  return result;
}

Halide::Buffer<uint8_t> HalideImageProc::lowPassFilter(Halide::Buffer<uint8_t> g, int lpf)  {
  Halide::Func lowPass;
  Halide::Var x, y, c;

  Halide::Expr one = 1;
  one = Halide::cast<uint8_t>(one);
  Halide::Expr zero = 0;
  zero = Halide::cast<uint8_t>(zero);

  Halide::Func clamped;
  Halide::Expr clamped_x = Halide::clamp(x, 0, g.width() - 1);
  Halide::Expr clamped_y = Halide::clamp(y, 0, g.height() - 1);
  clamped(x, y) = g(clamped_x, clamped_y);
  clamped = Halide::BoundaryConditions::constant_exterior(g, zero);

  Halide::Expr p0 = clamped(x - 1, y - 1);
  Halide::Expr p1 = clamped(x, y - 1);
  Halide::Expr p2 = clamped(x + 1, y - 1);
  Halide::Expr p3 = clamped(x - 1, y);
  Halide::Expr p4 = clamped(x, y);    // exclude my self
  Halide::Expr p5 = clamped(x + 1, y);
  Halide::Expr p6 = clamped(x - 1, y + 1);
  Halide::Expr p7 = clamped(x, y + 1);
  Halide::Expr p8 = clamped(x + 1, y + 1);
  
  Halide::Expr count = 0;
  count = Halide::cast<uint8_t>(count);

  count = Halide::select(p0 == one, count + one, count);
  count = Halide::select(p1 == one, count + one, count);
  count = Halide::select(p2 == one, count + one, count);
  count = Halide::select(p3 == one, count + one, count);
  count = Halide::select(p5 == one, count + one, count);
  count = Halide::select(p6 == one, count + one, count);
  count = Halide::select(p7 == one, count + one, count);
  count = Halide::select(p8 == one, count + one, count);

  Halide::Expr lpf_expr = lpf;
  lpf_expr = Halide::cast<uint8_t>(lpf_expr);
  lowPass(x, y) = Halide::select(p4 == one && count >= lpf_expr, one, zero);
  
  Halide::Buffer<uint8_t> result = lowPass.realize({g.width(), g.height(), 1});
  return result;
}

Halide::Buffer<float> HalideImageProc::additiveMaginitude() {
  Halide::Func max, min, mid, sum_ch, avg, delta;
  Halide:: Var x, y, c;

  Halide::Expr v = hHSV(x, y, 2);
  Halide::Expr two = (float) 2;
  two = Halide::cast<float> (two);
  Halide::Expr eight = (float) 8;
  eight = Halide::cast<float> (eight);
  
  // reduction
  Halide::RDom whole(0, hHSV.width(), 0, hHSV.height());
  max(x, y) = Halide::maximum(hHSV(x + whole.x, y + whole.y, 2));
  min(x, y) = Halide::minimum(hHSV(x + whole.x, y + whole.y, 2));
  mid(x, y) = ( max(x, y) + min(x, y)) / two;
  sum_ch(x, y) = Halide::sum(hHSV(x + whole.x, y + whole.y, 2));
  avg(x, y) = sum_ch(x, y) / ((float)hHSV.width() * (float) hHSV.height());
  delta(x, y) = (max(x, y) / eight) * (avg(x, y) / mid(x, y));

  Halide::Buffer<float> delta_result = delta.realize({1, 1});

  return delta_result;
}

Halide::Buffer<float>  HalideImageProc::edgeSharpen(Halide::Buffer<uint8_t> g, double s, Halide::Buffer<float> delta) {
  Halide::Func sharpen;
  Halide::Var x, y, c;

  Halide::Expr one = 1;
  one = Halide::cast<uint8_t>(one);
  Halide::Expr zero = 0;
  zero = Halide::cast<uint8_t>(zero);
  Halide::Expr two = 2;
  two = Halide::cast<uint8_t>(two);  // used to mark that point doesn't exist
  Halide::Expr minus_one = -1;
  minus_one = Halide::cast<float>(minus_one);
  Halide::Expr one_float = 1;
  one_float = Halide::cast<uint8_t>(one_float);
  Halide::Expr zero_float = 0;
  zero_float = Halide::cast<uint8_t>(zero_float);
  Halide::Expr s_float = (float)s;
  s_float = Halide::cast<float>(s_float);
  float delta_value = delta.get()->begin()[0];
  Halide::Expr delta_float = delta_value;
  delta_float = Halide::cast<float>(delta_float);

  Halide::Func clamped_hsv;
  Halide::Expr clamped_x = Halide::clamp(x, 0, hHSV.width() - 1);
  Halide::Expr clamped_y = Halide::clamp(y, 0, hHSV.height() - 1);
  clamped_hsv(x, y, c) = hHSV(clamped_x, clamped_y, c);
  
  Halide::Func clamped_g;
  Halide::Expr clamped_xg = Halide::clamp(x, 0, g.width() - 1);
  Halide::Expr clamped_yg = Halide::clamp(y, 0, g.height() - 1);
  clamped_g(x, y) = g(clamped_xg, clamped_yg);
  clamped_g = Halide::BoundaryConditions::constant_exterior(g, two);  // mark as not exist


  // pixel
  Halide::Expr p0 = clamped_hsv(x - 1, y - 1, 2);
  Halide::Expr p1 = clamped_hsv(x, y - 1, 2);
  Halide::Expr p2 = clamped_hsv(x + 1, y - 1, 2);
  Halide::Expr p3 = clamped_hsv(x - 1, y, 2);
  Halide::Expr p4 = clamped_hsv(x, y, 2);    // exclude my self
  Halide::Expr p5 = clamped_hsv(x + 1, y, 2);
  Halide::Expr p6 = clamped_hsv(x - 1, y + 1, 2);
  Halide::Expr p7 = clamped_hsv(x, y + 1, 2);
  Halide::Expr p8 = clamped_hsv(x + 1, y + 1, 2);

  // gate
  Halide::Expr g0 = clamped_g(x - 1, y - 1);
  Halide::Expr g1 = clamped_g(x, y - 1);
  Halide::Expr g2 = clamped_g(x + 1, y - 1);
  Halide::Expr g3 = clamped_g(x - 1, y);
  Halide::Expr g4 = clamped_g(x, y);    // exclude my self
  Halide::Expr g5 = clamped_g(x + 1, y);
  Halide::Expr g6 = clamped_g(x - 1, y + 1);
  Halide::Expr g7 = clamped_g(x, y + 1);
  Halide::Expr g8 = clamped_g(x + 1, y + 1);
  
  Halide::Expr count = 0;
  count = Halide::cast<uint8_t>(count);
  count = Halide::select(g0 != two, count + one, count);
  count = Halide::select(g1 != two, count + one, count);
  count = Halide::select(g2 != two, count + one, count);
  count = Halide::select(g3 != two, count + one, count);
  count = Halide::select(g5 != two, count + one, count);
  count = Halide::select(g6 != two, count + one, count);
  count = Halide::select(g7 != two, count + one, count);
  count = Halide::select(g8 != two, count + one, count);

  Halide::Expr sum = 0;
  sum = Halide::cast<float>(sum);
  sum = Halide::select(g0 != two, sum + p0, sum);
  sum = Halide::select(g1 != two, sum + p1, sum);
  sum = Halide::select(g2 != two, sum + p2, sum);
  sum = Halide::select(g3 != two, sum + p3, sum);
  sum = Halide::select(g5 != two, sum + p5, sum);
  sum = Halide::select(g6 != two, sum + p6, sum);
  sum = Halide::select(g7 != two, sum + p7, sum);
  sum = Halide::select(g8 != two, sum + p8, sum);

  Halide::Expr mean = sum / count;
  Halide::Expr oldV = hHSV(x, y, 2);

  Halide::Expr factor = Halide::select(oldV < mean, minus_one * oldV / mean,
                                     mean / oldV);

  // Halide::Expr newV = Halide::max(Halide::min(oldV + s_float * delta_float * factor, one_float), zero_float);   

  // Alternative
  Halide::Expr newV = Halide::select( g4 == zero, oldV,
                                      oldV + s_float * delta_float * factor);
  newV = Halide::max(Halide::min(newV, one_float), zero_float);

  Halide::Expr oldH = hHSV(x, y, 0);
  Halide::Expr oldS = hHSV(x, y, 1);
  
  sharpen(x, y, c) = Halide::select(c == 0, oldH, 
                                    c == 1, oldS, newV);

  Halide::Buffer<float> result = sharpen.realize({hHSV.width(), hHSV.height(), 3});
  hHSV = result;
  return result;
}
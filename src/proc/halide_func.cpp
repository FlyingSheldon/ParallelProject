#include "proc/halide_func.h"
#include <halide_image_io.h>

Halide::Buffer<uint8_t> LoadImage(std::string filename) {
  Halide::Buffer<uint8_t> hImg = Halide::Tools::load_image(filename);
  return hImg;
}

Halide::Func LoadImageFunc(Halide::Buffer<uint8_t> hImg) {
  Halide::Func output;
  Halide::Var x, y, c;
  output(x, y, c) = hImg(x, y, c);
  return output;
}

Halide::Func rgbToHsvFunc(Halide::Func input) {
  Halide::Func max_ch, min_ch, diff, hsv;
  Halide::Var x, y, c;


  Halide::Expr R = input(x, y, 0) / 255.0f;
  Halide::Expr G = input(x, y, 1) / 255.0f;
  Halide::Expr B = input(x, y, 2) / 255.0f;

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
  
  return hsv;
}

Halide::Func hsvToRgbFunc(Halide::Func hsv) {
  Halide::Func rgb;
  Halide::Var x, y, c;

  Halide::Expr H = hsv(x, y, 0);
  Halide::Expr S = hsv(x, y, 1);
  Halide::Expr V = hsv(x, y, 2);

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
  return rgb;
}

Halide::Func edgeDetect(Halide::Func hsv, double eth, int width, int height) {
  Halide::Func edge;
  Halide::Var x, y, c;

  // 
  // Halide::Func hsv;
  // hsv(x, y, c) = hHSV(x, y, c);
  // hsv.trace_loads();

  // clamp
  Halide::Func clamped;
  Halide::Expr clamped_x = Halide::clamp(x, 0, width - 1);
  Halide::Expr clamped_y = Halide::clamp(y, 0, height - 1);
  clamped(x, y, c) = hsv(clamped_x, clamped_y, c);

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

  return edge;
}

Halide::Func lowPassFilter(Halide::Func edge, int lpf, int width, int height) {
  Halide::Func lowPass;
  Halide::Var x, y, c;

  Halide::Expr one = 1;
  one = Halide::cast<uint8_t>(one);
  Halide::Expr zero = 0;
  zero = Halide::cast<uint8_t>(zero);

  Halide::Func clamped;
  Halide::Expr clamped_x = Halide::clamp(x, 0, width - 1);
  Halide::Expr clamped_y = Halide::clamp(y, 0, height - 1);
  clamped(x, y) = edge(clamped_x, clamped_y);
  
  // set up the range, not sure whether it's OK
  Halide::Range x_range(0, width);
  Halide::Range y_range(0, height);
  Halide::Region region;
  region.push_back(x_range);
  region.push_back(y_range);
  clamped = Halide::BoundaryConditions::constant_exterior(edge, zero, region);

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
  return lowPass;
}

Halide::Func additiveMaginitude(Halide::Func hsv, int width, int height) {
  Halide::Func max, min, mid, sum_ch, avg, delta;
  Halide:: Var x, y, c;

  Halide::Expr v = hsv(x, y, 2);
  Halide::Expr two = (float) 2;
  two = Halide::cast<float> (two);
  Halide::Expr eight = (float) 8;
  eight = Halide::cast<float> (eight);
  
  // reduction
  Halide::RDom whole(0, width, 0, height);
  max(x, y) = Halide::maximum(hsv(x + whole.x, y + whole.y, 2));
  min(x, y) = Halide::minimum(hsv(x + whole.x, y + whole.y, 2));
  mid(x, y) = ( max(x, y) + min(x, y)) / two;
  sum_ch(x, y) = Halide::sum(hsv(x + whole.x, y + whole.y, 2));
  avg(x, y) = sum_ch(x, y) / ((float)width * (float)height);
  delta(x, y) = (max(x, y) / eight) * (avg(x, y) / mid(x, y));

  return delta;
}

Halide::Func edgeSharpen(Halide::Func hsv, Halide::Func lowPass, double s, Halide::Func delta, int width, int height) {
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
  Halide::Expr delta_float = delta(0, 0);  // ????
  delta_float = Halide::cast<float>(delta_float);

  Halide::Func clamped_hsv;
  Halide::Expr clamped_x = Halide::clamp(x, 0, width - 1);
  Halide::Expr clamped_y = Halide::clamp(y, 0, height - 1);
  clamped_hsv(x, y, c) = hsv(clamped_x, clamped_y, c);
  
  Halide::Func clamped_g;
  Halide::Expr clamped_xg = Halide::clamp(x, 0, width - 1);
  Halide::Expr clamped_yg = Halide::clamp(y, 0, height - 1);
  clamped_g(x, y) = lowPass(clamped_xg, clamped_yg);
  // set up the range, not sure whether it's OK
  Halide::Range x_range(0, width);
  Halide::Range y_range(0, height);
  Halide::Region region;
  region.push_back(x_range);
  region.push_back(y_range);
  clamped_g = Halide::BoundaryConditions::constant_exterior(lowPass, two, region);  // mark as not exist

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
  count = Halide::select(g4 != two, count + one, count);
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
  sum = Halide::select(g4 != two, sum + p4, sum);
  sum = Halide::select(g5 != two, sum + p5, sum);
  sum = Halide::select(g6 != two, sum + p6, sum);
  sum = Halide::select(g7 != two, sum + p7, sum);
  sum = Halide::select(g8 != two, sum + p8, sum);

  Halide::Expr mean = sum / count;
  Halide::Expr oldV = hsv(x, y, 2);

  oldV = Halide::print_when(x == 179 && y == 0, oldV, "oldV when x=", x, ", y=", y);

  Halide::Expr epo = (float)1e-4;
  epo = Halide::cast<float> (epo);
  Halide::Expr factor = Halide::select(Halide::abs(oldV - mean) < epo, 1,
                                      oldV < mean, minus_one * oldV / mean,
                                      mean / oldV);


  // Alternative
  Halide::Expr newV = Halide::select( g4 == zero, oldV,
                                      oldV + s_float * delta_float * factor);
  newV = Halide::max(Halide::min(newV, one_float), zero_float);

  newV = Halide::print_when(x == 179 && y == 0, newV, "newV when x=", x, ", y=", y);

  Halide::Expr oldH = hsv(x, y, 0);
  Halide::Expr oldS = hsv(x, y, 1);
  
  sharpen(x, y, c) = Halide::select(c == 0, oldH, 
                                    c == 1, oldS, newV);

  return sharpen;
}





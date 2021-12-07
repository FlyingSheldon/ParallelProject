#pragma once
#include <Halide.h>

Halide::Buffer<uint8_t> LoadImage(std::string filename);

class SharpenPipeline {
private:
  Halide::Func hsvFunc{"hsv"}, edgeFunc{"edge"}, lowPassFunc{"lowPass"},
      deltaFunc{"delta"}, sharpenHsvFunc{"sharpenHsv"};

  static constexpr int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  static constexpr int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

public:
  Halide::Func sharpen{"sharpen"};
  Halide::Var x{"x"}, y{"y"}, c{"c"};
  int width, height;

  SharpenPipeline(Halide::Buffer<uint8_t> &img, double s)
      : width(img.width()), height(img.height()) {
    Halide::Func input;
    input(x, y, c) = img(x, y, c);

    double eth = 0.07;
    int lpf = 2;

    hsvFunc = this->rgbToHsvFunc(input);
    edgeFunc = this->edgeDetect(hsvFunc, eth);
    lowPassFunc = this->lowPassFilter(edgeFunc, lpf);
    deltaFunc = this->additiveMaginitude(hsvFunc);
    sharpenHsvFunc = this->edgeSharpen(hsvFunc, lowPassFunc, s, deltaFunc);
    sharpen = this->hsvToRgbFunc(sharpenHsvFunc);

    deltaFunc.compute_root();
  }

  Halide::Func rgbToHsvFunc(Halide::Func input) {
    Halide::Func max_ch, min_ch, diff, hsv;

    Halide::Expr R = input(x, y, 0) / 255.0f;
    Halide::Expr G = input(x, y, 1) / 255.0f;
    Halide::Expr B = input(x, y, 2) / 255.0f;

    max_ch(x, y) = Halide::max(R, G, B);
    min_ch(x, y) = Halide::min(R, G, B);
    diff(x, y) = max_ch(x, y) - min_ch(x, y);

    Halide::Expr V = max_ch(x, y);
    Halide::Expr C = diff(x, y);

    Halide::Expr H =
        Halide::select(C == 0, 0, R == V && G >= B, 60 * (0 + (G - B) / C),
                       R == V && G < B, 60 * (6 + (G - B) / C), G == V,
                       60 * (2 + (B - R) / C), 60 * (4 + (R - G) / C));

    Halide::Expr S = Halide::select(V == 0, 0, C / V);

    hsv(x, y, c) = Halide::select(c == 0, H, c == 1, S, V);

    return hsv;
  }

  Halide::Func edgeDetect(Halide::Func hsv, double eth) {
    Halide::Func edge;

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

    edge(x, y) = Halide::select(Halide::abs(local - west) >= eth_float ||
                                    Halide::abs(local - north) >= eth_float,
                                one, zero);

    return edge;
  }

  Halide::Func lowPassFilter(Halide::Func edge, int lpf) {
    Halide::Func lowPass;

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

    Halide::Expr count = 0;
    count = Halide::cast<uint8_t>(count);

    for (int i = 0; i < 8; i++) {
      Halide::Expr p = clamped(x + dx[i], y + dy[i]);
      count = Halide::select(p == one, count + one, count);
    }

    Halide::Expr lpf_expr = lpf;
    lpf_expr = Halide::cast<uint8_t>(lpf_expr);
    lowPass(x, y) =
        Halide::select(clamped(x, y) == one && count >= lpf_expr, one, zero);
    return lowPass;
  }

  Halide::Func additiveMaginitude(Halide::Func hsv) {
    Halide::Func max, min, mid, sum_ch, avg, delta;

    Halide::Expr v = hsv(x, y, 2);
    Halide::Expr OUT_BOUND = (float)2;
    OUT_BOUND = Halide::cast<float>(OUT_BOUND);
    Halide::Expr eight = (float)8;
    eight = Halide::cast<float>(eight);

    // reduction
    Halide::RDom whole(0, width, 0, height);
    // max(0, 0) = Halide::maximum(hsv(whole.x, whole.y, 2));
    max(x, y) = Halide::maximum(hsv(x + whole.x, y + whole.y, 2));
    // min(0, 0) = Halide::minimum(hsv(whole.x, whole.y, 2));
    min(x, y) = Halide::minimum(hsv(x + whole.x, y + whole.y, 2));
    // mid(0, 0) = (max(0, 0) + min(0, 0)) / OUT_BOUND;
    mid(x, y) = (max(x, y) + min(x, y)) / OUT_BOUND;
    // sum_ch(0, 0) = Halide::sum(hsv(whole.x, whole.y, 2));
    sum_ch(x, y) = Halide::sum(hsv(x + whole.x, y + whole.y, 2));
    // avg(0, 0) = sum_ch(0, 0) / ((float)width * (float)height);
    avg(x, y) = sum_ch(x, y) / ((float)width * (float)height);
    // delta(0, 0) = (max(0, 0) / eight) * (avg(0, 0) / mid(0, 0));
    delta(x, y) = (max(x, y) / eight) * (avg(x, y) / mid(x, y));

    return delta;
  }

  Halide::Func edgeSharpen(Halide::Func hsv, Halide::Func lowPass, double s,
                           Halide::Func delta) {
    Halide::Func sharpen;

    Halide::Expr one = 1;
    one = Halide::cast<uint8_t>(one);
    Halide::Expr zero = 0;
    zero = Halide::cast<uint8_t>(zero);
    Halide::Expr OUT_BOUND = 2;
    OUT_BOUND = Halide::cast<uint8_t>(
        OUT_BOUND); // used to mark that point doesn't exist
    Halide::Expr minus_one = -1;
    minus_one = Halide::cast<float>(minus_one);
    Halide::Expr one_float = 1;
    one_float = Halide::cast<uint8_t>(one_float);
    Halide::Expr zero_float = 0;
    zero_float = Halide::cast<uint8_t>(zero_float);
    Halide::Expr s_float = (float)s;
    s_float = Halide::cast<float>(s_float);
    Halide::Expr delta_float = delta(0, 0); // ????
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
    clamped_g = Halide::BoundaryConditions::constant_exterior(
        lowPass, OUT_BOUND, region); // mark as not exist

    Halide::Expr count = one;
    Halide::Expr sum = clamped_hsv(x, y, 2);

    for (int i = 0; i < 8; i++) {
      Halide::Expr g = clamped_g(x + dx[i], y + dy[i]);
      Halide::Expr p = clamped_hsv(x + dx[i], y + dy[i], 2);
      count = Halide::select(g != OUT_BOUND, count + one, count);
      sum = Halide::select(g != OUT_BOUND, sum + p, sum);
    }

    Halide::Expr mean = sum / count;
    Halide::Expr oldV = hsv(x, y, 2);

    Halide::Expr epo = (float)1e-4;
    epo = Halide::cast<float>(epo);
    Halide::Expr factor =
        Halide::select(Halide::abs(oldV - mean) < epo, 1, oldV < mean,
                       minus_one * oldV / mean, mean / oldV);

    // Alternative
    Halide::Expr newV = Halide::select(clamped_g(x, y) == zero, oldV,
                                       oldV + s_float * delta_float * factor);
    newV = Halide::max(Halide::min(newV, one_float), zero_float);

    Halide::Expr oldH = hsv(x, y, 0);
    Halide::Expr oldS = hsv(x, y, 1);

    sharpen(x, y, c) = Halide::select(c == 0, oldH, c == 1, oldS, newV);

    return sharpen;
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

    Halide::Expr r =
        Halide::select(i == 0 || i == 5, V, i == 1, q, i == 2 || i == 3, p, t);

    Halide::Expr g =
        Halide::select(i == 0, t, i == 1 || i == 2, V, i == 3, q, p);

    Halide::Expr b =
        Halide::select(i == 0 || i == 1, p, i == 2, t, i == 3 || i == 4, V, q);

    r = r * 255.0f;
    r = Halide::cast<uint8_t>(r);
    g = g * 255.0f;
    g = Halide::cast<uint8_t>(g);
    b = b * 255.0f;
    b = Halide::cast<uint8_t>(b);

    rgb(x, y, c) = Halide::select(c == 0, r, c == 1, g, b);
    return rgb;
  }
};
#pragma once
#include <Halide.h>
#include <vector>

Halide::Buffer<uint8_t> LoadImage(std::string filename);

class SharpenPipeline {
private:
  Halide::Func hsv{"hsv"}, edge{"edge"}, lowPass{"lowPass"}, delta{"delta"},
      sharpenHsv{"sharpenHsv"}, input{"input"}, reductionInter{"intermediate"},
      changed{"changed"};
  Halide::Var x{"x"}, y{"y"}, c{"c"}, xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};

  static constexpr int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  static constexpr int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

  void Schedule(int i) {
    switch (i) {
    case 0: {
      hsv.compute_root().parallel(y);
      delta.compute_root();
      reductionInter.compute_root().update().parallel(y);
      edge.compute_at(lowPass, y);
      lowPass.compute_root().parallel(y);
      sharpen.parallel(y);
      break;
    }
    case 1: {
      hsv.compute_root().parallel(y);
      delta.compute_root();
      reductionInter.compute_root().update().parallel(y);
      edge.compute_at(lowPass, xo);
      lowPass.compute_root().tile(x, y, xo, yo, xi, yi, 128, 128).parallel(yo);
      sharpen.parallel(y);
      break;
    }
    case 2: {
      hsv.compute_root().parallel(y);
      delta.compute_root();
      reductionInter.compute_root().update().parallel(y);
      edge.store_at(lowPass, yo).compute_at(lowPass, yi);
      lowPass.compute_root().split(y, yo, yi, 16).parallel(yo);
      sharpen.parallel(y);
      break;
    }
    default: {
      hsv.compute_root().parallel(y);
      delta.compute_root();
      reductionInter.compute_root().update().parallel(y);
      edge.compute_root().parallel(y);
      lowPass.compute_root().parallel(y);
      sharpen.parallel(y);
      break;
    }
    }
  }

public:
  Halide::Func sharpen{"sharpen"};
  int width, height;

  SharpenPipeline(Halide::Buffer<uint8_t> &img, double s)
      : width(img.width()), height(img.height()) {
    input(x, y, c) = img(x, y, c);

    double eth = 0.07;
    int lpf = 2;

    rgbToHsvFunc();
    additiveMaginitude();
    edgeDetect(eth);
    lowPassFilter(lpf);
    edgeSharpen(s);
    hsvToRgbFunc();
  }

  void ScheduleForCpu(int i = 0) { Schedule(i); }

  Halide::Func rgbToHsvFunc() {
    Halide::Func max_ch, min_ch, diff, hf("hf"), sf("sf"), vf("vf");

    Halide::Expr R = input(x, y, 0) / 255.0f;
    Halide::Expr G = input(x, y, 1) / 255.0f;
    Halide::Expr B = input(x, y, 2) / 255.0f;

    max_ch(x, y) = Halide::max(R, G, B);
    min_ch(x, y) = Halide::min(R, G, B);
    diff(x, y) = max_ch(x, y) - min_ch(x, y);

    vf(x, y) = max_ch(x, y);
    Halide::Expr V = vf(x, y);
    Halide::Expr C = diff(x, y);

    hf(x, y) =
        Halide::select(C == 0, 0, R == V && G >= B, 60 * (0 + (G - B) / C),
                       R == V && G < B, 60 * (6 + (G - B) / C), G == V,
                       60 * (2 + (B - R) / C), 60 * (4 + (R - G) / C));
    Halide::Expr H = hf(x, y);

    sf(x, y) = Halide::select(V == 0, 0, C / V);
    Halide::Expr S = sf(x, y);

    hsv(x, y, c) = Halide::select(c == 0, H, c == 1, S, V);

    hsv.reorder(c, x, y).bound(c, 0, 3).unroll(c, 3);
    hf.compute_at(hsv, x);
    sf.compute_at(hsv, x);
    vf.compute_at(hsv, x);

    return hsv;
  }

  Halide::Func edgeDetect(double eth) {
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

  Halide::Func lowPassFilter(int lpf) {
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

  Halide::Func additiveMaginitude() {
    Halide::Func mms("mms");
    Halide::Var z;
    Halide::Expr mid, avg;

    Halide::Expr v = hsv(x, y, 2);
    Halide::Expr two = (float)2;
    two = Halide::cast<float>(two);
    Halide::Expr eight = (float)8;
    eight = Halide::cast<float>(eight);

    // Halide::RDom whole(0, width, 0, height);

    reductionInter(x, y, z) = Halide::select(z == 0, 1.0f, 0.0f);
    mms(x, y, z) = Halide::select(z == 0, 1.0f, 0.0f);

    Halide::RDom rx(0, width);
    reductionInter(0, y, z) = Halide::select(
        z == 0, Halide::min(reductionInter(0, y, z), hsv(rx, y, 2)), z == 1,
        Halide::max(reductionInter(0, y, z), hsv(rx, y, 2)),
        reductionInter(0, y, z) + hsv(rx, y, 2));

    Halide::RDom ry(0, height);

    mms(0, 0, z) = Halide::select(
        z == 0, Halide::min(mms(0, 0, z), reductionInter(0, ry, 2)), z == 1,
        Halide::max(mms(0, 0, z), reductionInter(0, ry, 2)),
        mms(0, 0, z) + reductionInter(0, ry, 2));

    mid = (mms(x, y, 1) + mms(x, y, 0)) / two;
    avg = mms(x, y, 2) / ((float)width * (float)height);

    delta(x, y) = (mms(x, y, 1) / eight) * (avg / mid);

    return delta;
  }

  Halide::Func edgeSharpen(double s) {
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

    sharpenHsv(x, y, c) = Halide::select(c == 0, oldH, c == 1, oldS, newV);

    return sharpenHsv;
  }

  Halide::Func hsvToRgbFunc() {
    Halide::Func rf("rf"), gf("gf"), bf("bf");

    Halide::Expr H = sharpenHsv(x, y, 0);
    Halide::Expr S = sharpenHsv(x, y, 1);
    Halide::Expr V = sharpenHsv(x, y, 2);

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

    rf(x, y) = r;
    gf(x, y) = g;
    bf(x, y) = b;

    sharpen(x, y, c) = Halide::select(c == 0, r, c == 1, g, b);

    // sharpen(x, y, c) =
    //     Halide::select(lowPass(x, y) == 0, input(x, y, c), changed(x, y, c));
    sharpen.reorder(c, x, y).bound(c, 0, 3).unroll(c, 3);
    rf.compute_at(sharpen, x);
    gf.compute_at(sharpen, x);
    bf.compute_at(sharpen, x);

    return changed;
  }
};
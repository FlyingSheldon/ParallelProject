#include "img_helper.cuh"

__device__ double3 rgbToHsv(uchar3 px) {
  double r = static_cast<double>(px.x) / 255.0;
  double g = static_cast<double>(px.y) / 255.0;
  double b = static_cast<double>(px.z) / 255.0;
  double cmax = fmax(r, fmax(g, b));
  double cmin = fmin(r, fmin(g, b));
  double diff = cmax - cmin;

  double h = -1.0, s = -1.0;

  if (cmax == cmin) {
    h = 0.0;
  } else if (cmax == r) {
    h = fmod((60.0 * ((g - b) / diff) + 360.0), 360.0);
  } else if (cmax == g) {
    h = fmod((60.0 * ((r - g) / diff) + 360.0), 360.0);
  } else if (cmax == b) {
    h = fmod((60.0 * ((r - g) / diff) + 360.0), 360.0);
  }

  if (cmax == 0.0) {
    s = 0.0;
  } else {
    s = diff / cmax;
  }

  return make_double3(h, s, cmax);
}

__device__ uchar3 hsvToRbg(double3 px) {
  double h = px.x;
  double s = px.y;
  double v = px.z;

  int i = static_cast<int>(floor(h / 60.0)) % 6;
  double f = h / 60.0 - static_cast<double>(i);
  double p = v * (1.0 - s);
  double q = v * (1.0 - f * s);
  double t = v * (1.0 - (1.0 - f) * s);
  double r, g, b;

  switch (i) {
  case 0:
    r = v, g = t, b = p;
    break;
  case 1:
    r = q, g = v, b = p;
    break;
  case 2:
    r = p, g = v, b = t;
    break;
  case 3:
    r = p, g = q, b = v;
    break;
  case 4:
    r = t, g = p, b = v;
    break;
  case 5:
    r = v, g = p, b = q;
    break;
  default:
    break;
  }

  return make_uchar3(static_cast<unsigned char>(r),
                     static_cast<unsigned char>(g),
                     static_cast<unsigned char>(b));
}
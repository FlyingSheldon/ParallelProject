#include "image/image.h"
#include <filesystem>
#include <iostream>
#include "proc/halide_proc.h"
#include "proc/halide_func.h"


int main(int argc, char **argv) {
  LinearImageProc linearProc;

  double eth = 0.07;
  int lpf = 2;
  double scale = 0.5;

  Halide::Buffer<uint8_t> hImg = LoadImage("test.jpg");  // get width and height
  Halide::Func input = LoadImageFunc(hImg);
  Halide::Func hsv = rgbToHsvFunc(input);
  Halide::Func edge = edgeDetect(hsv, eth, hImg.width(), hImg.height());
  Halide::Func lowPass = lowPassFilter(edge, lpf, hImg.width(), hImg.height());
  Halide::Func delta = additiveMaginitude(hsv, hImg.width(), hImg.height());  // just return a float ?
  Halide::Func sharpen = edgeSharpen(hsv, lowPass, scale, delta, hImg.width(), hImg.height());
  Halide::Func rgb = hsvToRgbFunc(sharpen);
  Halide::Buffer<uint8_t> result = rgb.realize({hImg.width(), hImg.height(), 3});

}
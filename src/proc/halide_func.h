#pragma once
#include <Halide.h>

Halide::Buffer<uint8_t> LoadImage(std::string filename);
Halide::Func LoadImageFunc(Halide::Buffer<uint8_t> hImg);
Halide::Func rgbToHsvFunc(Halide::Func input);
Halide::Func hsvToRgbFunc(Halide::Func hsv);
Halide::Func edgeDetect(Halide::Func hsv, double eth, int w, int h);
Halide::Func lowPassFilter(Halide::Func edge, int lpf, int width, int height);
Halide::Func additiveMaginitude(Halide::Func hsv, int width, int height);
Halide::Func edgeSharpen(Halide::Func hsv, Halide::Func lowPass, double s, Halide::Func delta, int width, int height);
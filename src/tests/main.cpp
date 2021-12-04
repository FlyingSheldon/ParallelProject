#include "image/image.h"
#include <filesystem>
#include <iostream>
#include "proc/halide_proc.h"


int main(int argc, char **argv) {
  HalideImageProc halideProc;
  LinearImageProc linearProc;

  double eth = 0.07;
  int lpf = 2;
  halideProc.LoadImage("test.jpg");
  halideProc.rgbToHsv(); 
  Halide::Buffer<uint8_t> g1 = halideProc.edgeDetect(eth);
  Halide::Buffer<uint8_t> g1_filter = halideProc.lowPassFilter(g1, lpf);
  Halide::Buffer<float> result = halideProc.additiveMaginitude();

  auto res2 = linearProc.LoadImage("test.jpg");
  Image &img2 = *linearProc.GetImage();
  rgbToHsv(img2);
  std::vector<bool> g2 = edgeDetect(img2, eth);
  lowPassFilter(img2, g2, lpf);
  additiveMaginitude(img2);
}
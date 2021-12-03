#include "image/image.h"
#include <filesystem>
#include <iostream>
#include "proc/halide_proc.h"


int main(int argc, char **argv) {
  std::cout << "Please check test path: " << std::filesystem::current_path()
            << std::endl;

  HalideImageProc halideProc;
  LinearImageProc linearProc;

  double eth = 0.07;
  halideProc.LoadImage("test.jpg");
  halideProc.rgbToHsv(); 
  Halide::Buffer<uint8_t> g1 = halideProc.edgeDetect(eth);

  auto res2 = linearProc.LoadImage("test.jpg");
  Image &img2 = *linearProc.GetImage();
  rgbToHsv(img2);
  std::vector<bool> g2 = edgeDetect(img2, eth);

  std::cout << "g1 dim " << g1.dimensions() << std::endl;
  std::cout << "g1 channel " << g1.channels() << std::endl;

  const uint8_t *ptr = g1.get()->begin();
  size_t i = 0; 

//   for (int y = 0; y < g1.height(); y++) {
//     for (int x = 0; x < g1.width(); x++) {
//       int index = y * g1.width() + x;
//       uint8_t value2 = g2[index] ? 1 : 0;

//       ASSERT_EQ(value2, ptr[i++])
//           << "Pixel " << x << " " << y << " g not equal";
//     }
//   }
}
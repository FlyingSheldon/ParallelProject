#include "image/image.h"
#include <filesystem>
#include <iostream>
#include "proc/halide_proc.h"


int main(int argc, char **argv) {
    std::cout << "Please check test path: " << std::filesystem::current_path()
            << std::endl;

  HalideImageProc halideProc;
  LinearImageProc linearProc;

  halideProc.LoadImage("test.jpg");
  halideProc.rgbToHsv(); 
  halideProc.hsvToRgb();
  Image &img = *halideProc.GetImage();

  auto res2 = linearProc.LoadImage("test.jpg");
  Image &img2 = *linearProc.GetImage();

//   ASSERT_EQ(img.GetWidth(), img2.GetWidth());
//   ASSERT_EQ(img.GetHeight(), img2.GetHeight());

//   for (int y = 0; y < img.GetHeight(); y++) {
//     for (int x = 0; x < img.GetWidth(); x++) {
//       auto p = img.GetPixelData(x, y);
//       auto p2 = img2.GetPixelData(x, y);
//       ASSERT_EQ((int)p[0], (int)p2[0])
//           << "Pixel " << x << " " << y << " red not equal";
//       ASSERT_EQ((int)p[1], (int)p2[1])
//           << "Pixel " << x << " " << y << " green not equal";
//       ASSERT_EQ((int)p[2], (int)p2[2])
//           << "Pixel " << x << " " << y << " blue not equal";
//     }
//   }
}
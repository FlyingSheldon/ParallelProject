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
  Halide::Buffer<float> hsv = halideProc.rgbToHsv(); 
  
  Image &img2 = *linearProc.GetImage();
  rgbToHsv(img2);

  size_t h = 0;
  size_t s = hsv.width() * hsv.height() * 1;
  size_t v = hsv.width() * hsv.height() * 2;
  const float *ptr = hsv.get()->begin();
  double near = 0.001;

  for (int y = 0; y < img2.GetHeight(); y++) {
    for (int x = 0; x < img2.GetWidth(); x++) {
      const double *hsvp = img2.GetHSV(x, y);

      // ASSERT_DOUBLE_EQ(hsvp[0], ptr[h++])
      //   << "Pixel " << x << " " << y << " H not equal";
      auto p2 = img2.GetPixelData(x, y);
    //   std::cout << "Pixel" << x << " " << y 
    //   << "R:" << (int)p2[0] << " G:" << (int)p2[1] << " B:" << (int)p2[2];

    //   ASSERT_NEAR(hsvp[0], ptr[h], near)
    //       << "Pixel " << x << " " << y << " H not equal";
    //   ASSERT_NEAR(hsvp[1], ptr[s], near)
    //       << "Pixel " << x << " " << y << " S not equal";
    //   ASSERT_NEAR(hsvp[2], ptr[v], near)
    //       << "Pixel " << x << " " << y << " V not equal" 
    //       << "R:" << (int)p2[0] << " G:" << (int)p2[1] << " B:" << (int)p2[2];

      h++;
      s++;
      v++;
    }
  }
}
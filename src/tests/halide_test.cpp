#include "image/image.h"
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>

#ifdef PP_USE_HALIDE
#include "proc/halide_proc.h"

// TEST(HalideTest, BrightenTest) {
//   std::cout << "Please check test path: " << std::filesystem::current_path()
//             << std::endl;

//   HalideImageProc halideProc;
//   LinearImageProc linearProc;

//   auto res = halideProc.LoadImage("test.jpg");
//   auto res2 = linearProc.LoadImage("test.jpg");

//   Image::ImageError *error = std::get_if<Image::ImageError>(&res);

//   ASSERT_FALSE(error) << "Failed to open the file: " << *error;

//   halideProc.Brighten(1.5);
//   linearProc.Brighten(1.5);

//   Image &img = *halideProc.GetImage();
//   Image &img2 = *linearProc.GetImage();

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
// }

// TEST(HalideTest, Rgb2HsvTest) {
//   std::cout << "Please check test path: " << std::filesystem::current_path()
//             << std::endl;

//   HalideImageProc halideProc;
//   LinearImageProc linearProc;


//   halideProc.LoadImage("test.jpg");
//   Halide::Buffer<float> hsv = halideProc.rgbToHsv(); 

//   auto res2 = linearProc.LoadImage("test.jpg");
//   Image &img2 = *linearProc.GetImage();
//   rgbToHsv(img2);

//   size_t h = 0;
//   size_t s = hsv.width() * hsv.height() * 1;
//   size_t v = hsv.width() * hsv.height() * 2;
//   const float *ptr = hsv.get()->begin();
//   double near = 0.001;

//   for (int y = 0; y < img2.GetHeight(); y++) {
//     for (int x = 0; x < img2.GetWidth(); x++) {
//       const double *hsvp2 = img2.GetHSV(x, y);
//       auto p2 = img2.GetPixelData(x, y);

//       ASSERT_NEAR((float)hsvp2[0], ptr[h], near)
//           << "Pixel " << x << " " << y << " H not equal";
//       ASSERT_NEAR((float)hsvp2[1], ptr[s], near)
//           << "Pixel " << x << " " << y << " S not equal";
//       ASSERT_NEAR((float)hsvp2[2], ptr[v], near)
//           << "Pixel " << x << " " << y << " V not equal" 
//           << "R:" << (int)p2[0] << " G:" << (int)p2[1] << " B:" << (int)p2[2];

//       h++;
//       s++;
//       v++;
//     }
//   }
// }

// TEST(HalideTest, Hsv2RgbTest) {
//   std::cout << "Please check test path: " << std::filesystem::current_path()
//             << std::endl;

//   HalideImageProc halideProc;
//   LinearImageProc linearProc;

//   halideProc.LoadImage("test.jpg");
//   halideProc.rgbToHsv(); 
//   halideProc.hsvToRgb();
//   Image &img = *halideProc.GetImage();

//   auto res2 = linearProc.LoadImage("test.jpg");
//   Image &img2 = *linearProc.GetImage();

//   ASSERT_EQ(img.GetWidth(), img2.GetWidth());
//   ASSERT_EQ(img.GetHeight(), img2.GetHeight());

//   double near = 1.0;

//   for (int y = 0; y < img.GetHeight(); y++) {
//     for (int x = 0; x < img.GetWidth(); x++) {
//       auto p = img.GetPixelData(x, y);
//       auto p2 = img2.GetPixelData(x, y);
//       ASSERT_NEAR((int)p[0], (int)p2[0], near)
//           << "Pixel " << x << " " << y << " red not equal";
//       ASSERT_NEAR((int)p[1], (int)p2[1], near)
//           << "Pixel " << x << " " << y << " green not equal";
//       ASSERT_NEAR((int)p[2], (int)p2[2], near)
//           << "Pixel " << x << " " << y << " blue not equal";
//     }
//   }
// }

TEST(HalideTest, EdgeDetectTest) {
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

  const uint8_t *ptr = g1.get()->begin();
  size_t i = 0; 
  std::cout << "g1 dim " << g1.dimensions() << std::endl;

  for (int y = 0; y < g1.height(); y++) {
    for (int x = 0; x < g1.width(); x++) {
      int index = y * g1.width() + x;
      uint8_t value2 = g2[index] ? 1 : 0;

      ASSERT_EQ(value2, ptr[i++])
          << "Pixel " << x << " " << y << " g not equal";
    }
  }
}


#else

TEST(HalideTest, PlaceHolderTest) {
  std::cout << "Halide is not supported, not tested!" << std::endl;
}

#endif // #ifdef PP_USE_HALIDE
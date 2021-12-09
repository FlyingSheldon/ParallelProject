#include "image/image.h"
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>

#ifdef PP_USE_HALIDE
#include "proc/halide_func.h"
#include "proc/halide_proc.h"

TEST(HalideTest, BrightenTest) {
  std::cout << "Please check test path: " << std::filesystem::current_path()
            << std::endl;

  HalideImageProc halideProc;
  LinearImageProc linearProc;

  auto res = halideProc.LoadImage("test.jpg");
  auto res2 = linearProc.LoadImage("test.jpg");

  Image::ImageError *error = std::get_if<Image::ImageError>(&res);

  ASSERT_FALSE(error) << "Failed to open the file: " << *error;

  halideProc.Brighten(1.5);
  linearProc.Brighten(1.5);

  Image &img = *halideProc.GetImage();
  Image &img2 = *linearProc.GetImage();

  ASSERT_EQ(img.GetWidth(), img2.GetWidth());
  ASSERT_EQ(img.GetHeight(), img2.GetHeight());

  for (int y = 0; y < img.GetHeight(); y++) {
    for (int x = 0; x < img.GetWidth(); x++) {
      auto p = img.GetPixelData(x, y);
      auto p2 = img2.GetPixelData(x, y);
      ASSERT_EQ((int)p[0], (int)p2[0]) << "Pixel " << x << " " << y
                                       << " red not equal";
      ASSERT_EQ((int)p[1], (int)p2[1]) << "Pixel " << x << " " << y
                                       << " green not equal";
      ASSERT_EQ((int)p[2], (int)p2[2]) << "Pixel " << x << " " << y
                                       << " blue not equal";
    }
  }
}

TEST(HalideTest, SharpenTest) {
  std::cout << "Please check test path: " << std::filesystem::current_path()
            << std::endl;

  HalideImageProc halideProc;
  LinearImageProc linearProc;

  double eth = 0.07;
  int lpf = 2;
  double scale = 0.5;
  halideProc.LoadImage("test.jpg");
  halideProc.Sharpen(scale);
  Image &img = *halideProc.GetImage();

  auto res2 = linearProc.LoadImage("test.jpg");
  linearProc.Sharpen(scale);
  Image &img2 = *linearProc.GetImage();

  ASSERT_EQ(img.GetWidth(), img2.GetWidth());
  ASSERT_EQ(img.GetHeight(), img2.GetHeight());

  double near = 2.0;

  for (int y = 0; y < img.GetHeight(); y++) {
    for (int x = 0; x < img.GetWidth(); x++) {
      auto p = img.GetPixelData(x, y);
      auto p2 = img2.GetPixelData(x, y);
      ASSERT_NEAR((int)p[0], (int)p2[0], near) << "Pixel " << x << " " << y
                                               << " red not equal";
      ASSERT_NEAR((int)p[1], (int)p2[1], near) << "Pixel " << x << " " << y
                                               << " green not equal";
      ASSERT_NEAR((int)p[2], (int)p2[2], near) << "Pixel " << x << " " << y
                                               << " blue not equal";
    }
  }
}

TEST(HalideTest, FuncTest) {
  LinearImageProc linearProc;

  double eth = 0.07;
  int lpf = 2;
  double scale = 0.5;

  Halide::Buffer<uint8_t> hImg = LoadImage("test.jpg"); // get width and height
  SharpenPipeline p(hImg, scale);

  p.ScheduleForCpu();
  Halide::Buffer<uint8_t> result =
      p.sharpen.realize({hImg.width(), hImg.height(), 3});

  auto res2 = linearProc.LoadImage("test.jpg");
  Image &img2 = *linearProc.GetImage();
  rgbToHsv(img2);
  std::vector<bool> g2 = edgeDetect(img2, eth);
  lowPassFilter(img2, g2, lpf);
  float delta2 = additiveMaginitude(img2);
  edgeSharpen(img2, g2, scale, delta2);
  hsvToRgb(img2);

  size_t h = 0;
  size_t s = result.width() * result.height() * 1;
  size_t v = result.width() * result.height() * 2;
  const uint8_t *ptr = result.get()->begin();
  double near = 2.0;

  for (int y = 0; y < img2.GetHeight(); y++) {
    for (int x = 0; x < img2.GetWidth(); x++) {
      auto p2 = img2.GetPixelData(x, y);

      ASSERT_NEAR((int)p2[0], ptr[h], near) << "Pixel " << x << " " << y
                                            << " R not equal";
      ASSERT_NEAR((int)p2[1], ptr[s], near) << "Pixel " << x << " " << y
                                            << " G not equal";
      ASSERT_NEAR((int)p2[2], ptr[v], near)
          << "Pixel " << x << " " << y << " B not equal"
          << "R:" << (int)p2[0] << " G:" << (int)p2[1] << " B:" << (int)p2[2];

      h++;
      s++;
      v++;
    }
  }
}

#else

TEST(HalideTest, PlaceHolderTest) {
  std::cout << "Halide is not supported, not tested!" << std::endl;
}

#endif // #ifdef PP_USE_HALIDE
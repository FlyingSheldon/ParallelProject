#include "image/image.h"
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>

#ifdef PP_USE_HALIDE
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
      ASSERT_EQ((int)p[0], (int)p2[0])
          << "Pixel " << x << " " << y << " red not equal";
      ASSERT_EQ((int)p[1], (int)p2[1])
          << "Pixel " << x << " " << y << " green not equal";
      ASSERT_EQ((int)p[2], (int)p2[2])
          << "Pixel " << x << " " << y << " blue not equal";
    }
  }
}

#else

TEST(HalideTest, PlaceHolderTest) {
  std::cout << "Halide is not supported, not tested!" << std::endl;
}

#endif // #ifdef PP_USE_HALIDE
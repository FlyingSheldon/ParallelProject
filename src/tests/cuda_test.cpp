#include "image/image.h"
#include "proc/cuda_proc.h"
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>

TEST(CudaTest, BrightenTest) {
  std::cout << "Please check test path: " << std::filesystem::current_path()
            << std::endl;

  auto imageResult = Image::OpenImage("test.jpg");
  const Image::ImageError *error = std::get_if<Image::ImageError>(&imageResult);

  ASSERT_FALSE(error) << "Failed to open the file";

  Image img = std::move(std::get<Image>(imageResult));
  Image img2(img);

  CudaImageProc cudaProc;
  LinearImageProc linearProc;

  cudaProc.Brighten(img, 1.5);
  linearProc.Brighten(img2, 1.5);

  for (int y = 0; y < img.GetHeight(); y++) {
    for (int x = 0; x < img.GetWidth(); x++) {
      auto p = img.GetPixelData(x, y);
      auto p2 = img.GetPixelData(x, y);
      ASSERT_EQ(p[0], p2[0]) << "Pixel " << x << " " << y << " red not equal";
      ASSERT_EQ(p[1], p2[1]) << "Pixel " << x << " " << y << " green not equal";
      ASSERT_EQ(p[2], p2[2]) << "Pixel " << x << " " << y << " blue not equal";
    }
  }
}

TEST(CudaTest, SharpenTest) {
  std::cout << "Please check test path: " << std::filesystem::current_path()
            << std::endl;

  auto imageResult = Image::OpenImage("test.jpg");
  const Image::ImageError *error = std::get_if<Image::ImageError>(&imageResult);

  ASSERT_FALSE(error) << "Failed to open the file";

  Image img = std::move(std::get<Image>(imageResult));
  Image img2(img);

  CudaImageProc cudaProc;
  LinearImageProc linearProc;

  cudaProc.Sharpen(img, 0.5);
  linearProc.Sharpen(img2, 0.5);

  for (int y = 0; y < img.GetHeight(); y++) {
    for (int x = 0; x < img.GetWidth(); x++) {
      auto p = img.GetPixelData(x, y);
      auto p2 = img.GetPixelData(x, y);
      ASSERT_EQ(p[0], p2[0]) << "Pixel " << x << " " << y << " red not equal";
      ASSERT_EQ(p[1], p2[1]) << "Pixel " << x << " " << y << " green not equal";
      ASSERT_EQ(p[2], p2[2]) << "Pixel " << x << " " << y << " blue not equal";
    }
  }
}
// #include "cuda/cuda_proc.h"
#include "image/image.h"
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>

#ifdef PP_USE_CUDA
#include "proc/cuda_proc.h"

TEST(CudaTest, BrightenTest) {
  // std::cout << "Please check test path: " << std::filesystem::current_path()
  //           << std::endl;

  CudaImageProc cudaProc;
  LinearImageProc linearProc;

  auto res = cudaProc.LoadImage("test.jpg");
  auto res2 = linearProc.LoadImage("test.jpg");

  Image::ImageError *error = std::get_if<Image::ImageError>(&res);

  ASSERT_FALSE(error) << "Failed to open the file: " << *error;

  cudaProc.Brighten(1.5);
  linearProc.Brighten(1.5);

  Image &img = *cudaProc.GetImage();
  Image &img2 = *linearProc.GetImage();

  ASSERT_EQ(img.GetWidth(), img2.GetWidth());
  ASSERT_EQ(img.GetHeight(), img2.GetHeight());

  for (int y = 0; y < img.GetHeight(); y++) {
    for (int x = 0; x < img.GetWidth(); x++) {
      auto p = img.GetPixelData(x, y);
      auto p2 = img2.GetPixelData(x, y);
      ASSERT_EQ(p[0], p2[0]) << "Pixel " << x << " " << y << " red not equal";
      ASSERT_EQ(p[1], p2[1]) << "Pixel " << x << " " << y << " green not equal";
      ASSERT_EQ(p[2], p2[2]) << "Pixel " << x << " " << y << " blue not equal";
    }
  }
}

// TEST(CudaTest, EdgeLPFTest) {
//   auto imgResult = Image::OpenImage("test.jpg");
//   Image::ImageError *error = std::get_if<Image::ImageError>(&imgResult);

//   ASSERT_FALSE(error) << "Failed to open the file: " << *error;

//   double eth = 0.07;
//   int lpf = 2;

//   Image img = std::move(std::get<Image>(imgResult));

//   size_t size = img.GetHeight() * img.GetWidth();

//   std::vector<uint8_t> edges(size);
//   std::vector<uint8_t> lpfOut(size);
//   std::vector<double> hsv(size * 3);

//   cudaEdgeDetect(img.GetPixelData(0, 0), edges.data(), 3, img.GetWidth(),
//                  img.GetHeight(), 0.07, hsv.data());

//   cudaEdgeLPFDbg(edges.data(), lpfOut.data(), img.GetWidth(),
//   img.GetHeight(),
//                  lpf);

//   rgbToHsv(img);
//   auto g = edgeDetect(img, eth);

//   for (size_t i = 0; i < size * 3; i++) {
//     ASSERT_NEAR(hsv[i], img.GetHSVData(0, 0)[i], 0.001);
//   }

//   for (size_t i = 0; i < size; i++) {
//     ASSERT_EQ((edges[i] == 1), g[i]);
//   }

//   lowPassFilter(img, g, lpf);

//   for (size_t i = 0; i < size; i++) {
//     ASSERT_EQ((lpfOut[i] == 1), g[i]) << i << " not equal";
//   }
// }

// TEST(CudaTest, EdgeDetectTest) {
//   auto imgResult = Image::OpenImage("test.jpg");
//   Image::ImageError *error = std::get_if<Image::ImageError>(&imgResult);

//   ASSERT_FALSE(error) << "Failed to open the file: " << *error;

//   double eth = 0.07;

//   Image img = std::move(std::get<Image>(imgResult));

//   size_t size = img.GetHeight() * img.GetWidth();

//   std::vector<uint8_t> edges(size);
//   std::vector<double> hsv(size * 3);

//   cudaEdgeDetect(img.GetPixelData(0, 0), edges.data(), 3, img.GetWidth(),
//                  img.GetHeight(), 0.07, hsv.data());

//   rgbToHsv(img);
//   auto g = edgeDetect(img, eth);

//   for (size_t i = 0; i < size * 3; i++) {
//     ASSERT_NEAR(hsv[i], img.GetHSVData(0, 0)[i], 0.000001);
//   }
//   std::cerr << "1522, 0: " << hsv[1522 * 3 - 1] << std::endl;

//   for (size_t i = 0; i < size; i++) {
//     ASSERT_EQ((edges[i] == 1), g[i]);
//   }
// }

TEST(CudaTest, SharpenTest) {
  CudaImageProc cudaProc;
  LinearImageProc linearProc;

  auto res = cudaProc.LoadImage("test.jpg");
  auto res2 = linearProc.LoadImage("test.jpg");

  Image::ImageError *error = std::get_if<Image::ImageError>(&res);

  ASSERT_FALSE(error) << "Failed to open the file: " << *error;

  cudaProc.Sharpen(0.5);
  linearProc.Sharpen(0.5);

  Image &img = *cudaProc.GetImage();
  Image &img2 = *linearProc.GetImage();

  for (int y = 0; y < img.GetHeight(); y++) {
    for (int x = 0; x < img.GetWidth(); x++) {
      auto p = img.GetPixelData(x, y);
      auto p2 = img2.GetPixelData(x, y);
      ASSERT_TRUE(std::abs((int)p[0] - (int)p2[0]) <= 2)
          << "Pixel " << x << " " << y << " red not equal:   " << (int)p[0]
          << " " << (int)p2[0];
      ASSERT_TRUE(std::abs((int)p[1] - (int)p2[1] <= 2))
          << "Pixel " << x << " " << y << " green not equal: " << (int)p[1]
          << " " << (int)p2[1];
      ASSERT_TRUE(std::abs((int)p[2] - (int)p2[2] <= 2))
          << "Pixel " << x << " " << y << " blue not equal:  " << (int)p[2]
          << " " << (int)p2[2];
    }
  }
}
#else

TEST(CudaTest, PlaceHolderTest) {
  std::cout << "Cuda is not supported, not tested!" << std::endl;
}

#endif // #ifdef PP_USE_CUDA

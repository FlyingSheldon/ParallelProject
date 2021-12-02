#pragma once
#include "image/image.h"
#include <variant>

const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

class ImageProc {
public:
  using ImageIOResult = std::variant<std::monostate, Image::ImageError>;
  virtual void Brighten(double value) {}
  virtual void Sharpen(double value) {}
  virtual bool IsSupported() const { return false; }
  virtual std::string Name() const { return ""; }
  virtual ImageIOResult LoadImage(std::string filename) {
    return Image::ImageError("Not implemented");
  }
  virtual ImageIOResult SaveImage(std::string filename) {
    return Image::ImageError("Not implemented");
  }
  virtual Image *GetImage() { return nullptr; }
};

class LinearImageProc : public ImageProc {
public:
  virtual void Brighten(double value) override;
  virtual void Sharpen(double value) override;
  virtual bool IsSupported() const override { return true; }
  virtual std::string Name() const override { return "linear"; }
  virtual ImageIOResult LoadImage(std::string filename) override;
  virtual ImageIOResult SaveImage(std::string filename) override;
  virtual Image *GetImage() override { return &img; }

private:
  Image img;
};

namespace linear {
void brighten(Image &image, double value);
void sharpen(Image &image, double value);
} // namespace linear

void rgbToHsv(Image &image);
void hsvToRgb(Image &image);

/**
 * @brief Return a HVD image
 *
 * @param image
 * @param eth      threshold, [8,18] is suitable
 * @return         HVD image
 */
std::vector<bool> edgeDetect(Image &image, double eth);
bool edgeDetectPixel(Image &image, size_t x, size_t y, double eth);

/**
 * @brief Add LPF to the HVD image
 *
 * @param image
 * @param[out] g   HVD image with LPF
 * @param lpf      threshold, [0,8]
 */
void lowPassFilter(Image &image, std::vector<bool> &g, int lpf);
bool lowPassFilterPixel(Image &image, std::vector<bool> &g, size_t x, size_t y,
                        int lpf);

double additiveMaginitude(Image &image);
double computelocalMean(Image &image, size_t x, size_t y);

/**
 * @brief
 *
 * @param[out] image
 * @param g            HVD image with LPF
 * @param s            scaling factor controls degree of sharpness, [0, 1]
 * @param delta        additive magnitude
 */
void edgeSharpen(Image &image, std::vector<bool> &g, double s, double delta);

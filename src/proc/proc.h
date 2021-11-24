#pragma once
#include "image/image.h"

const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

class ImageProc {
public:
  virtual void Brigten(Image &image, double value);
  virtual void Sharpen(Image &image, double value);
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
bool lowPassFilterPixel(Image &image, std::vector<bool> g, size_t x, size_t y,
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
void edgeSharpen(Image &image, std::vector<bool> g, double s, double delta);

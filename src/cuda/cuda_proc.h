#pragma once
#include <cstdint>
#include <cstdlib>

void cudaSayHi();

void cudaBrighten(uint8_t *img, size_t size, size_t pixelSize, double value);
void cudaSharpen(uint8_t *img, size_t pixelSize, size_t width, size_t height,
                 double value, double eth, int lpf, double *hsv = nullptr);

void cudaEdgeDetect(uint8_t *img, uint8_t *edges, size_t pixelSize,
                    size_t width, size_t height, double eth,
                    double *hsv = nullptr);

void cudaEdgeLPFDbg(uint8_t *edges, uint8_t *output, size_t width,
                    size_t height, int lpf);

void cudaHSVDbg(uint8_t *img, double *hsv, size_t width, size_t height);

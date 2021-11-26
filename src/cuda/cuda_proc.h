#pragma once
#include <cstdint>
#include <cstdlib>

void cudaSayHi();

void cudaBrighten(uint8_t *img, size_t size, size_t pixelSize, double value);
void cudaSharpen(uint8_t *img, size_t pixelSize, size_t width, size_t height,
                 double value, double eth, int lpf);
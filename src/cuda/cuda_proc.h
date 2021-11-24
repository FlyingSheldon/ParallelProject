#pragma once
#include <cstdint>
#include <cstdlib>

void cudaSayHi();

void cudaBrighten(uint8_t *img, size_t size, size_t pixelSize, double value);
#pragma once
#include <cstdint>

constexpr static size_t kThreadPerBlock = 256;

__global__ void sayHi();

__global__ void brighten(uint8_t *img, size_t size, size_t pixelSize,
                         double value);
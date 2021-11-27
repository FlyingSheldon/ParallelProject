#pragma once

__device__ double3 rgbToHsv(uchar3 px);
__device__ uchar3 hsvToRgb(double3 px);
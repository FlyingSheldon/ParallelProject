#include "proc/halide_proc.h"
#include <cstdlib>
#include <iostream>

HalideImageProc::HalideImageProc() {}

void HalideImageProc::Brighten(Image &img, double value) {}

void HalideImageProc::Sharpen(Image &img, double value) {}

bool HalideImageProc::IsSupported() const { return false; }
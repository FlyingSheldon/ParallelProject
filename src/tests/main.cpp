#include "image/image.h"
#include "proc/halide_func.h"
#include "proc/halide_proc.h"
#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
  LinearImageProc linearProc;

  double eth = 0.07;
  int lpf = 2;
  double scale = 0.5;

  Halide::Buffer<uint8_t> hImg = LoadImage("test.jpg"); // get width and height
  SharpenPipeline p(hImg, scale);
  p.ScheduleForCpu(2);
  Halide::Buffer<uint8_t> result =
      p.sharpen.realize({hImg.width(), hImg.height(), 3});
  p.sharpen.print_loop_nest();
}
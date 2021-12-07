#include "image/image.h"
#include "proc/cuda_proc.h"
#include "proc/halide_proc.h"
#include "proc/proc.h"
#include "util/conf.h"
#include "util/flags.h"
#include "util/timer.h"
#include <cstdio>
#include <iostream>
#include <memory>

int main(int argc, char **argv) {
  Flags f(&argc, &argv);
  Conf conf(argc, argv);

  if (conf.confError) {
    std::cerr << conf.confError.value() << std::endl;
    return 1;
  }

  std::unique_ptr<ImageProc> proc;

  switch (conf.impl) {
  case Impl::CUDA:
    proc = std::make_unique<CudaImageProc>();
    break;
  case Impl::HALIDE:
    proc = std::make_unique<HalideImageProc>();
    break;
  default:
    proc = std::make_unique<LinearImageProc>();
    break;
  }

  if (!proc->IsSupported()) {
    std::cerr << proc->Name() << " is not supported in this binary"
              << std::endl;
  }

  auto imageResult = proc->LoadImage(conf.input);
  if (const Image::ImageError *error =
          std::get_if<Image::ImageError>(&imageResult)) {
    std::cerr << *error << std::endl;
    return 1;
  }

  auto brightStart = Timer::Now();
  if (conf.brightness != 1.0) {
    proc->Brighten(conf.brightness);
  }
  auto brightEnd = Timer::Now();

  auto sharpStart = Timer::Now();
  if (conf.sharpness != 0.0) {
    proc->Sharpen(conf.sharpness);
  }
  auto sharpEnd = Timer::Now();

  if (conf.showTime) {
    std::cout << "Brigten: "
              << Timer::DurationInMillisecond(brightStart, brightEnd) << " ms"
              << std::endl;
    std::cout << "Sharpen: "
              << Timer::DurationInMillisecond(sharpStart, sharpEnd) << " ms"
              << std::endl;
  }

  imageResult = proc->SaveImage(conf.output);
  if (const Image::ImageError *error =
          std::get_if<Image::ImageError>(&imageResult)) {
    std::cerr << *error << std::endl;
    return 1;
  }
}

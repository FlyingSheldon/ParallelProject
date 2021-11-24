#include "image/image.h"
#include "proc/cuda_proc.h"
#include "proc/halide_proc.h"
#include "proc/proc.h"
#include "util/conf.h"
#include "util/flags.h"
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

  auto imageResult = Image::OpenImage(conf.input);
  if (const Image::ImageError *error =
          std::get_if<Image::ImageError>(&imageResult)) {
    std::cerr << *error << std::endl;
    return 1;
  }

  Image img = std::move(std::get<Image>(imageResult));

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

  if (conf.brightness != 1.0) {
    proc->Brighten(img, conf.brightness);
  }

  if (conf.sharpness != 0.0) {
    proc->Sharpen(img, conf.sharpness);
  }

  img.Save(conf.output);
}

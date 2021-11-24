#include "image/image.h"
#include "proc/proc.h"
#include "util/conf.h"
#include "util/flags.h"
#include <cstdio>
#include <iostream>

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

  if (conf.brightness != 1.0) {
    linear::brighten(img, conf.brightness);
  }

  if (conf.sharpness != 0.0) {
    linear::sharpen(img, conf.sharpness);
  }

  img.Save(conf.output);

  printf("All good\n");
}

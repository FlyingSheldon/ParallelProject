#include "image/image.h"
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

  Image img(conf.input);

  for (size_t y = 0; y < img.GetHeight(); y++) {
    for (size_t x = 0; x < img.GetWidth(); x++) {
      uint8_t *pp = img.GetPixelData(x, y);
      for (size_t i = 0; i < img.GetPixelSize(); i++) {
        pp[i] = std::min(255, 10 + (int)pp[i]);
      }
    }
  }

  img.Save(conf.output);

  printf("All good\n");
}

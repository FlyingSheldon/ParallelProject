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

  // Parameter
  double eth = 0.03;
  int lpf = 2;
  double s = 1;  // scaling factor


  // linear::brighten(img, 10);

  rgbToHsv(img);
  printf("rgbToHsv\n");

  std::vector<bool> g = edgeDetect(img, eth);
  printf("edgeDetect\n");

  lowPassFilter(img, g, lpf);
  printf("lpf\n");

  double delta = additiveMaginitude(img);
  printf("delta\n");
  edgeSharpen(img, g, s, delta);
  printf("edgeSharpen\n");

  hsvToRgb(img);
  printf("hsvToRgb\n");

  img.Save(conf.output);

  printf("All good\n");
}

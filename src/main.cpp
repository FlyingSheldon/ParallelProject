#include "flags.h"
#include <Halide.h>
#include <cstdio>
#include <iostream>

int main(int argc, char **argv) {
  Flags f(argc, argv);

  Halide::Func gradient;
  Halide::Var x, y;

  Halide::Expr e = x + y;

  gradient(x, y) = e;

  Halide::Buffer<int32_t> output = gradient.realize({800, 600});

  for (int j = 0; j < output.height(); j++) {
    for (int i = 0; i < output.width(); i++) {
      // We can access a pixel of an Buffer object using similar
      // syntax to defining and using functions.
      if (output(i, j) != i + j) {
        printf("Something went wrong!\n"
               "Pixel %d, %d was supposed to be %d, but instead it's %d\n",
               i, j, i + j, output(i, j));
        return -1;
      }
    }
  }

  printf("All good\n");
}

#include "proc.cuh"
#include <cstdio>

__global__ void sayHi() {
  int i = threadIdx.x;
  printf("CUDA from %d\n", i);
}
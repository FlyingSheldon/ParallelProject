#include "cuda_proc.h"
#include "proc.cuh"

void cudaSayHi() { sayHi<<<1, 2>>>(); }
# ParallelProject (PP)


## How to build

`PP` has 3 main dependencies: LLVM, Halide and CUDA. Not all dependencies are required, either `Halide` or `CUDA` can be optional.

The minimum of LLVM version required depends on the Halide version, the oldest version we've tested is Halide 10.0.0 with LLVM 9.

Also, `cmake` is required to build `PP`. Visit [here](https://cmake.org/download/) for how to install `cmake`.

### macOS

Install LLVM and Halide:

```shell
brew install llvm halide
```

Configure the project:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

`cmake` will try to find all the dependencies, but it might not find `llvm` or `halide` in your system, if you find something warning you about `llvm` or `halide` not found. You can run this complete command to guide cmake to find `llvm` and `halide` (possibly under `user/local/Cellar` if you install them through home-brew:

```shell
cmake -S . -DCMAKE_BUILD_TYPE:STRING=Release -Dllvm_DIR=${LLVM_PATH}/lib/cmake/llvm -DHalide_DIR=${HALIDE_PATH}/lib/cmake/Halide -DHalideHelpers_DIR=${HALIDE_PATH}/lib/cmake/HalideHelpers  -B build
```

Build the project:

```shell
cmake --build build
```

Verify the main target is built:

```
build/src/pp --help
```

Then you can move `build/src/pp` anywhere to execute it.



### Linux

First make sure the dependencies of Halide is installed: `zlib`, `libjpeg` and `libpng`. 

For example, using apt:

```shell
apt-get install libjpeg-dev libpng-dev
```

Install [LLVM](https://llvm.org/) via package manager all binaries. 

Install [Halide](https://halide-lang.org/) via binaries of source.

Configure the project:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

`cmake` will try to find all the dependencies, but it might not find `llvm` or `halide` in your system, if you find something warning you about `llvm` or `halide` not found. You can run this complete command to guide cmake to find `llvm` and `halide`:

```shell
cmake -S . -DCMAKE_BUILD_TYPE:STRING=Release -Dllvm_DIR=${LLVM_PATH}/lib/cmake/llvm -DHalide_DIR=${HALIDE_PATH}/lib/cmake/Halide -DHalideHelpers_DIR=${HALIDE_PATH}/lib/cmake/HalideHelpers  -B build
```

CUDA will be auto detected if you have the environment. You can run `nvcc` to check. 

Build the project:

```shell
cmake --build build
```

Verify the main target is built:

```
build/src/pp --help
```

Then you can move `build/src/pp` anywhere to execute it.

## How to use pp

Run `pp --help` for helper message.

Generally, you can run something like:

```shell
pp input.jpg -o out.jpg --brightness 1.5 --sharpness 0.5 --impl ${IMPLEMENTATION}
```

The `${IMPLEMENTATION}` can be `linear`, `halide` or `cuda`. Whether the implementation is supported is determined at build time. If a implementation is not supported in the binary, you will get a message if you specify that implementation. 

## Stage UI

Stage UI is a graphic user interface of pp. Currently we haven't finished packaing steps. If you want to try it, you can start the development version:

```shell
cd stage
npm install
npm run dev
```


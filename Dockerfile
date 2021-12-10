FROM ubuntu:20.04

RUN bash -c "apt-get update && \
    apt-get -y install libjpeg-dev libpng-dev curl xz-utils build-essential"


RUN bash -c "curl -SL https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz  \
    | tar -xJC . && \
    mv clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04 llvm && \
    curl -SL https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.tar.gz | tar -xz && \
    mv cmake-3.22.1-linux-x86_64 cmake && \
    curl -SL https://github.com/halide/Halide/releases/download/v13.0.1/Halide-13.0.1-x86-64-linux-fb39d7e3149c0ee1e848bb9957be2ea18765c35d.tar.gz | tar -xz && \
    mv Halide-13.0.1-x86-64-linux halide && \
    mkdir pp"

WORKDIR /pp

COPY . .

RUN bash -c "export PATH=/cmake/bin:$PATH && \
    export CC=/llvm/bin/clang && \
    export CXX=/llvm/bin/clang++ && \
    cmake -S . -DCMAKE_BUILD_TYPE=Release \
    -Dllvm_DIR=/llvm/lib/cmake/llvm \
    -DHalide_DIR=/halide/lib/cmake/Halide \
    -DHalideHelpers_DIR=/halide/lib/cmake/HalideHelpers \
    -B build && \
    cmake --build build -j $(nproc) && \
    cp build/src/pp /usr/local/bin/pp"

CMD [ "/bin/bash" ]


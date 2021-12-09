FROM ubuntu:20.04

RUN bash -c "apt-get update && \
    apt-get -y install libjpeg-dev libpng-dev && \
    bash -c '$(wget -O - https://apt.llvm.org/llvm.sh)' && \
    "



COPY . .


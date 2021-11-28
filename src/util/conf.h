#pragma once
#include <optional>
#include <string>

enum class Impl { LINEAR, CUDA, HALIDE };

struct Conf {
  std::string input;
  std::string output;
  double sharpness, brightness;
  Impl impl;
  std::optional<std::string> confError;

  Conf(int argc, char **argv);
};
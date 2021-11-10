#pragma once
#include <optional>
#include <string>

struct Conf {
  std::string input;
  std::string output;
  std::optional<std::string> confError;

  Conf(int argc, char **argv);
};
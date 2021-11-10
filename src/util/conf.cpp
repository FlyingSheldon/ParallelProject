#include "conf.h"
#include "flags.h"

Conf::Conf(int argc, char **argv) {
  if (argc < 2) {
    confError = "No input file specified";
    return;
  }
  input = std::string(argv[1]);

  output = FLAGS_o;
  if (output.empty()) {
    confError = "No output file specified";
    return;
  }
}
// Thin wrapper for gflags because of its stupid help message handling
#pragma once

#include <gflags/gflags.h>
#include <iostream>
#include <unordered_set>
#include <vector>

DECLARE_bool(help);
DECLARE_bool(h);
DECLARE_double(brightness);
DECLARE_string(o);
DECLARE_double(sharpness);
DECLARE_string(impl);
DECLARE_bool(time);
DECLARE_int32(schedule);
DECLARE_bool(gpu);

class Flags {
  const std::unordered_set<std::string> helpFlags{
      "help", "brightness", "o",        "sharpness",
      "impl", "time",       "schedule", "gpu"};

  void showHelpMessage(const char *argv0) {
    std::cout << argv0 << ": " << gflags::ProgramUsage() << std::endl;
    std::vector<gflags::CommandLineFlagInfo> flags;
    gflags::GetAllFlags(&flags);
    for (const gflags::CommandLineFlagInfo &flag : flags) {
      if (helpFlags.find(flag.name) != helpFlags.cend()) {
        std::cout << gflags::DescribeOneFlag(flag);
      }
    }
    std::cout << std::flush;
  }

public:
  Flags(int *argc, char ***argv) {
    gflags::SetUsageMessage("Usage:");
    gflags::SetVersionString("0.0.1");
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);
    if (FLAGS_help || FLAGS_h) {
      FLAGS_help = false;
      showHelpMessage((*argv)[0]);
      exit(0);
    }
    gflags::HandleCommandLineHelpFlags();
  }

  ~Flags() { gflags::ShutDownCommandLineFlags(); }
};

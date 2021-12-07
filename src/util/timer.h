#pragma once
#include <chrono>

class Timer {
public:
  using Time = std::chrono::time_point<std::chrono::steady_clock>;
  static inline Time Now() { return std::chrono::steady_clock::now(); };

  static inline int DurationInMillisecond(Time from, Time to) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(to - from)
        .count();
  }
};
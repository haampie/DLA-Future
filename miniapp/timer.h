#pragma once

#include <chrono>
#include <iostream>

namespace dlaf {
namespace common {

  template <class clock = std::chrono::high_resolution_clock>
  class timer {
    using time_point = std::chrono::time_point<clock>;

    time_point start;

    inline time_point now() const {
      return clock::now();
    }

    public:
    timer() : start(now()) {}

    double elapsed() const {
      return std::chrono::duration_cast<std::chrono::duration<double>>(now() - start).count();
    }

    double elapsed(std::string str, double nr_ops = -1) const {
      return elapsed(str, std::cout, nr_ops);
    }

    template <class Out>
    double elapsed(const std::string str, Out& out, double nr_ops = -1) const {
      double el = elapsed();
      out << str << el << " s";
      if (nr_ops > 0)
        out << ", " << nr_ops / el / 1e9 << " GFlop/s";
      out << std::endl;
      return el;
    }
  };

}
}

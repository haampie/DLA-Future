#pragma once

#include <string>

namespace dlaf {
namespace common {
namespace utils {

template <class T>
std::string join(const T& last) {
  return std::to_string(last);
}

std::string join(const char* last) {
  return {last};
}

template <class T, class ...Ts>
std::string join(const T& current, const Ts& ... list) {
  return join(current) + " " + join(list...);
}

}
}
}

//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <complex>
#include <limits>
#include <cassert>

namespace dlaf {

using SizeType = int;

enum class Device { CPU, GPU };

enum class Backend { MC, GPU };

template <class T>
struct TypeInfo;

template <class T>
struct TypeInfo<const T> : public TypeInfo<T> {};

template <class T>
struct TypeInfo {
  using BaseType = T;
  using Type = T;
  using ComplexType = std::complex<T>;

  static constexpr int ops_add = 1;
  static constexpr int ops_mul = 1;
};

template <class T>
struct TypeInfo<std::complex<T>> {
  using BaseType = T;
  using Type = std::complex<T>;
  using ComplexType = std::complex<T>;

  static constexpr int ops_add = 2;
  static constexpr int ops_mul = 6;
};

template <class T>
using BaseType = typename TypeInfo<T>::BaseType;

template <class T>
using ComplexType = typename TypeInfo<T>::ComplexType;

/// Compute the number of operations
///
/// Given the number of additions and multiplications of type @tparam T
/// it returns the number of basic floating point operations
template <class T>
constexpr size_t total_ops(const size_t add, const size_t mul) {
  return TypeInfo<T>::ops_add * add + TypeInfo<T>::ops_mul * mul;
}

/// Return complex conjugate of a complex number
template <class T>
std::complex<T> conj(const std::complex<T> number) {
  return std::conj(number);
}

/// Return complex conjugate of a real number as a real number
///
/// It differs from std::conj just in the return type. In fact,
/// std::conj always returns a std::complex
template <class T>
T conj(const T number) {
  return number;
}

/// Cast from unisgned to signed integer types
///
/// It performs the cast checking if the given unsigned value can be stored in the destination type
template <class S, class U,
          std::enable_if_t<std::is_integral<U>::value && std::is_unsigned<U>::value &&
                               std::is_integral<S>::value && std::is_signed<S>::value,
                           int> = 0>
S to_signed(const U unsigned_value) {
  assert(std::numeric_limits<S>::max() > unsigned_value);
  return static_cast<S>(unsigned_value);
}

/// Cast from signed to unsigned integer types
///
/// It performs the cast checking if the given signed value is greater than 0 and if the destination type
/// can store the value.
template <class U, class S,
          std::enable_if_t<std::is_integral<U>::value && std::is_unsigned<U>::value &&
                               std::is_integral<S>::value && std::is_signed<S>::value,
                           int> = 0>
U to_unsigned(const S signed_value) {
  assert(signed_value >= 0);
  assert(std::numeric_limits<U>::max() >= signed_value);
  return static_cast<U>(signed_value);
}

/// Helper function for casting from unsigned to dlaf::SizeType
///
/// Useful when passing parameter to the BLAS/LAPACK interface
/// see dlaf::to_signed
auto to_SizeType = [](const auto unsigned_value) { return to_signed<SizeType>(unsigned_value); };

/// Helper function for casting from unsigned to int
///
/// Useful when passing parameters to the MPI interface
/// see dlaf::to_signed
auto to_int = [](const auto unsigned_value) { return to_signed<int>(unsigned_value); };

/// Helper function for casting from signed to std::size_t
///
/// Useful for interaction between std, but not only, with other interfaces that does not use usigned types (e.g. MPI, BLAS, ...)
/// see dlaf::to_unsigned
auto to_sizet = [](const auto signed_value) { return to_unsigned<std::size_t>(signed_value); };
}

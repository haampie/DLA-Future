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

#include "dlaf/memory/memory_view.h"
#include "dlaf/types.h"

namespace dlaf {

/// @brief The Tile object aims to provide an effective way to access the memory as a two dimensional
/// array. It does not allocate any memory, but it references the memory given by a @c MemoryView object.
/// It represents the building block of the Matrix object and of linear algebra algorithms.
///
/// Two levels of constness exist for @c Tile analogously to pointer semantics:
/// the constness of the tile and the constness of the data referenced by the tile.
/// Implicit conversion is allowed from tiles of non-const elements to tiles of const elements.
template <class T, Device device>
class Tile {
public:
  using ElementType = std::remove_const_t<T>;
  friend Tile<const ElementType, device>;

  /// @brief Constructs a (@p m x @p n) Tile.
  /// @throw std::invalid_argument if @p m < 0, @p n < 0 or @p ld < max(1, @p m).
  /// @throw std::invalid_argument if memory_view does not contain enough elements.
  /// The (i, j)-th element of the Tile is stored in the (i+ld*j)-th element of memory_view.
  Tile(SizeType m, SizeType n, memory::MemoryView<T, device> memory_view, SizeType ld)
      : m_(m), n_(n), memory_view_(memory_view), ld_(ld) {
    if (m_ < 0 || n_ < 0)
      throw std::invalid_argument("Error: Invalid Tile sizes");
    if (ld_ < m_ || ld_ < 1)
      throw std::invalid_argument("Error: Invalid Tile leading dimension");
    if (m_ + (n_ - 1) * ld_ > memory_view_.size())
      throw std::invalid_argument("Error: Tile exceeds the MemoryView limits");
  }

  Tile(const Tile&) = delete;

  Tile(Tile&& rhs) : m_(rhs.m_), n_(rhs.n_), memory_view_(std::move(rhs.memory_view_)), ld_(rhs.ld_) {
    rhs.m_ = 0;
    rhs.n_ = 0;
    rhs.ld_ = 1;
  }

  template <class U = T,
            class = typename std::enable_if_t<std::is_const<U>::value && std::is_same<T, U>::value>>
  Tile(Tile<ElementType, device>&& rhs)
      : m_(rhs.m_), n_(rhs.n_), memory_view_(std::move(rhs.memory_view_)), ld_(rhs.ld_) {
    rhs.m_ = 0;
    rhs.n_ = 0;
    rhs.ld_ = 1;
  }

  Tile& operator=(const Tile&) = delete;

  Tile& operator=(Tile&& rhs) {
    m_ = rhs.m_;
    n_ = rhs.n_;
    memory_view_ = std::move(rhs.memory_view_);
    ld_ = rhs.ld_;
    rhs.m_ = 0;
    rhs.n_ = 0;
    rhs.ld_ = 1;

    return *this;
  }

  template <class U = T,
            class = typename std::enable_if_t<std::is_const<U>::value && std::is_same<T, U>::value>>
  Tile& operator=(Tile<ElementType, device>&& rhs) {
    m_ = rhs.m_;
    n_ = rhs.n_;
    memory_view_ = std::move(rhs.memory_view_);
    ld_ = rhs.ld_;
    rhs.m_ = 0;
    rhs.n_ = 0;
    rhs.ld_ = 1;

    return *this;
  }

  /// @brief Returns the (i, j)-th element.
  /// @pre 0 <= @p i < @p m.
  /// @pre 0 <= @p j < @p n.
  T& operator()(SizeType i, SizeType j) const {
    return *ptr(i, j);
  }

  /// @brief Returns the pointer to the (i, j)-th element.
  /// @pre 0 <= @p i < @p m.
  /// @pre 0 <= @p j < @p n.
  T* ptr(SizeType i, SizeType j) const {
    assert(i >= 0 && i < m_);
    assert(j >= 0 && j < n_);
    return memory_view_(i + ld_ * j);
  }

  /// @brief Returns the number of rows.
  SizeType m() const {
    return m_;
  }
  /// @brief Returns the number of columns.
  SizeType n() const {
    return n_;
  }
  /// @brief Returns the leading dimension.
  SizeType ld() const {
    return ld_;
  }

private:
  SizeType m_;
  SizeType n_;
  memory::MemoryView<T, device> memory_view_;
  SizeType ld_;
};

}
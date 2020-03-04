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

#include <mpi.h>
#include "dlaf/common/buffer_basic.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"

namespace dlaf {
namespace comm {
namespace sync {

namespace internal {
namespace reduce {

/// @brief MPI_Reduce wrapper (sender side)
/// MPI Reduce(see MPI documentation for additional info)
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator
template <class BufferIn, class BufferOut>
void collector(Communicator& communicator, MPI_Op reduce_operation, BufferIn input, BufferOut output) {
  using T = std::remove_const_t<typename common::buffer_traits<BufferIn>::element_t>;

  common::BufferWithMemory<T> tmp_mem_input;
  common::BufferWithMemory<T> tmp_mem_output;

  common::BufferBasic<const T> tmp_in = input;
  common::BufferBasic<T> tmp_out = output;

  // if input is not contiguous, copy it in a contiguous temporary buffer
  if (!input.is_contiguous()) {
    tmp_in = tmp_mem_input = common::create_temporary_buffer(tmp_in);
    common::copy(input, tmp_mem_input);
  }

  // if output is not contiguous, create an intermediate buffer
  if (!output.is_contiguous())
    tmp_out = tmp_mem_output = common::create_temporary_buffer(tmp_out);

  auto&& message_input = comm::make_message(std::move(tmp_in));
  auto&& message_output = comm::make_message(BufferOut(tmp_out));

  MPI_Reduce(message_input.data(), message_output.data(), message_input.count(),
             message_input.mpi_type(), reduce_operation, communicator.rank(), communicator);

  // if output was not contiguous, copy it back!
  if (!output.is_contiguous())
    common::copy(tmp_out, output);
}

/// @brief MPI_Reduce wrapper (receiver side)
/// MPI Reduce(see MPI documentation for additional info)
/// @param rank_root  the rank that will collect the result in output
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator
template <class BufferIn>
void participant(int rank_root, Communicator& communicator, MPI_Op reduce_operation, BufferIn input) {
  using T = std::remove_const_t<typename common::buffer_traits<BufferIn>::element_t>;

  common::BufferWithMemory<T> tmp_mem_input;

  common::BufferBasic<const T> tmp_in = input;

  // if input is not contiguous, copy it in a contiguous temporary buffer
  if (!input.is_contiguous()) {
    tmp_in = tmp_mem_input = common::create_temporary_buffer(tmp_in);
    common::copy(input, tmp_mem_input);
  }

  auto&& message_input = comm::make_message(std::move(tmp_in));

  MPI_Reduce(message_input.data(), nullptr, message_input.count(), message_input.mpi_type(),
             reduce_operation, rank_root, communicator);
}

}
}

/// @brief MPI_Reduce wrapper
/// MPI Reduce(see MPI documentation for additional info)
/// @param rank_root  the rank that will collect the result in output
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator
template <class BufferIn, class BufferOut>
void reduce(const int rank_root, Communicator& communicator, MPI_Op reduce_operation, BufferIn input,
            BufferOut output) {
  if (rank_root == communicator.rank())
    internal::reduce::collector(communicator, reduce_operation, input, output);
  else
    internal::reduce::participant(rank_root, communicator, reduce_operation, input);
}
}
}
}

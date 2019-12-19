//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <iostream>

#include <mpi.h>
#include <hpx/hpx_init.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/mc/cholesky.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"

using T = double;

struct options_t {
  int64_t m;
  int64_t mb;
  int64_t grid_rows;
  int64_t grid_cols;
  int64_t nruns;
};

options_t check_options(hpx::program_options::variables_map& vm);

template <class T>
void init_random_positive_definite(dlaf::Matrix<T, dlaf::Device::CPU>& matrix) {
  using namespace dlaf;

  util_matrix::assertSizeSquare(matrix, __PRETTY_FUNCTION__, "matrix");
  util_matrix::assertBlocksizeSquare(matrix, __PRETTY_FUNCTION__, "matrix");

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(-1., 1.); // TODO make seed different for each tile

  auto& distribution = matrix.distribution();

  const auto offset = 2 * matrix.size().cols();

  for (SizeType local_tile_j = 0; local_tile_j < distribution.localNrTiles().cols(); ++local_tile_j) {
    for (SizeType local_tile_i = 0; local_tile_i < distribution.localNrTiles().rows(); ++local_tile_i) {
      LocalTileIndex current_tile{local_tile_i, local_tile_j};
      matrix(current_tile).then(hpx::util::unwrapping(
          [dis, gen](auto&& tile) mutable {
            for (SizeType j = 0; j < tile.size().cols(); ++j)
              for (SizeType i = 0; i < tile.size().rows(); ++i)
                tile(TileElementIndex(i, j)) = dis(gen);
          }));
    }
  }

  for (SizeType global_j = 0; global_j < matrix.nrTiles().cols(); ++global_j) {
    for (SizeType global_i = 0; global_i < matrix.nrTiles().rows(); ++global_i) {
      GlobalTileIndex index_tile_diag{global_i, global_j};
      const auto owner = distribution.rankGlobalTile(index_tile_diag);

      if (owner != matrix.rankIndex())
        continue;

      matrix(index_tile_diag).then(hpx::util::unwrapping(
            [offset](auto &&tile_diag) {
              for (SizeType j = 0; j < tile_diag.size().cols(); ++j)
                tile_diag(TileElementIndex(j, j)) += offset;
            }));
    }
  }
}

int hpx_main(hpx::program_options::variables_map& vm) {
  options_t opts = check_options(vm);

  dlaf::comm::Communicator world(MPI_COMM_WORLD);
  dlaf::comm::CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols,
                                         dlaf::common::Ordering::ColumnMajor);

  // Allocate memory for the matrix
  dlaf::GlobalElementSize matrix_size(opts.m, opts.m);
  dlaf::TileElementSize block_size(opts.mb, opts.mb);

  dlaf::Matrix<T, dlaf::Device::CPU> matrix(matrix_size, block_size, comm_grid);

  // Run choleksy
  for (auto run_index = 0; run_index < opts.nruns; ++run_index) {
    std::cout << "[" << run_index << "]" << std::endl;

    init_random_positive_definite(matrix);

    // run cholesky
    cholesky(comm_grid, blas::Uplo::Lower, matrix);

    // TODO wait for last task and barrier for all ranks

    // TODO print benchmark results

    // TODO run test (optional)
  }

  return hpx::finalize();
}

int main(int argc, char** argv) {
  // Initialize MPI
  int threading_required = MPI_THREAD_SERIALIZED;
  int threading_provided;
  MPI_Init_thread(&argc, &argv, threading_required, &threading_provided);

  if (threading_provided != threading_required) {
    std::fprintf(stderr, "Provided MPI threading model does not match the required one.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size",
     value<int64_t>()->default_value(4096),
     "Matrix size.")
    ("block-size",
     value<int64_t>()->default_value(128),
     "Block cyclic distribution size.")
    ("grid-rows",
     value<int64_t>()->default_value(1),
     "Number of row processes in the 2D communicator.")
    ("grid-cols",
     value<int64_t>()->default_value(1),
     "Number of column processes in the 2D communicator.")
    ("nruns",
     value<int64_t>()->default_value(1),
     "Number of runs to compute the cholesky")
  ;
  // clang-format on

  auto ret_code = hpx::init(hpx_main, desc_commandline, argc, argv);

  // TODO resources management/scheduler/pool

  MPI_Finalize();

  return ret_code;
}

options_t check_options(hpx::program_options::variables_map& vm) {
  options_t opts = {
      .m = vm["matrix-size"].as<int64_t>(),
      .mb = vm["block-size"].as<int64_t>(),

      .grid_rows = vm["grid-rows"].as<int64_t>(),
      .grid_cols = vm["grid-cols"].as<int64_t>(),

      .nruns = vm["nruns"].as<int64_t>(),
  };

  if (opts.m <= 0)
    throw std::runtime_error("matrix size must be a positive number");
  if (opts.mb <= 0)
    throw std::runtime_error("block size must be a positive number");

  if (opts.grid_rows <= 0)
    throw std::runtime_error("number of grid rows must be a positive number");
  if (opts.grid_cols <= 0)
    throw std::runtime_error("number of grid columns must be a positive number");

  return opts;
}

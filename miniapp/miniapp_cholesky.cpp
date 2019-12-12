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

#include "dlaf/matrix.h"
#include "dlaf/communication/communicator_grid.h"

using T = double;

struct options_t {
  std::size_t m;
  std::size_t mb;
  std::size_t grid_rows;
  std::size_t grid_cols;
};

options_t check_options(hpx::program_options::variables_map& vm, const std::size_t total_ranks);

int hpx_main(hpx::program_options::variables_map& vm) {
  dlaf::comm::Communicator world(MPI_COMM_WORLD);

  // init communicators
  options_t opts = check_options(vm, world.size());
  dlaf::comm::CommunicatorGrid grid(world, opts.grid_rows, opts.grid_cols, dlaf::common::Ordering::ColumnMajor);

  std::cout << world.rank() << " " << opts.m << " " << opts.mb << " " << opts.grid_rows << " " << opts.grid_cols  << std::endl;

  // init matrix (random)

  // run multiple times
  for(; false;) {
    // run cholesky

    // print benchmark results

    // run test (optional)
  }

  return hpx::finalize();
}

int main(int argc, char **argv) {
  // Initialize MPI
  int threading_required = MPI_THREAD_SERIALIZED;
  int threading_provided;
  MPI_Init_thread(&argc, &argv, threading_required, &threading_provided);

  if (threading_provided != threading_required) {
    std::fprintf(stderr, "Provided MPI threading model does not match the required one.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Configure application-specific options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  // options
  desc_commandline.add_options()
    ("matrix-size,m",
     value<std::size_t>()->default_value(4096),
     "Matrix size.")
    ("block-size,nb",
     value<std::size_t>()->default_value(128),
     "Block cyclic distribution size.")
    ("grid-rows,r",
     value<std::size_t>()->default_value(1),
     "Number of row processes in the 2D communicator.")
    ("grid-cols,c",
     value<std::size_t>()->default_value(1),
     "Number of column processes in the 2D communicator.")
    ;
  /*
    ("use-pools,u",     "Enable advanced HPX thread pools and executors")
    ("use-scheduler,s", "Enable custom priority scheduler")
    ("use-numa,a",      "Enable numa sensitive scheduling")
    ("mpi-threads,m", util::po::value<int>()->default_value(1),
      "Number of threads to assign to MPI")
    ("hp-queues,H", util::po::value<int>()->default_value(1),
      "Number of high priority queues to use in custom scheduler")
   */

  auto ret_code = hpx::init(hpx_main, desc_commandline, argc, argv);

  // resources management/scheduler/pool

  MPI_Finalize();

  return ret_code;
}

options_t check_options(hpx::program_options::variables_map& vm, const std::size_t total_ranks) {
  options_t opts = {
    .m = vm["matrix-size"].as<std::size_t>(),
    .mb = vm["block-size"].as<std::size_t>(),

    .grid_rows = vm["grid-rows"].as<std::size_t>(),
    .grid_cols = vm["grid-cols"].as<std::size_t>(),
  };

  if (opts.m <= 0) throw std::runtime_error("invalid matrix size");
  if (opts.mb <= 0) throw std::runtime_error("invalid matrix block size");

  std::size_t specified_ranks = opts.grid_rows * opts.grid_cols;

  if (specified_ranks < total_ranks) std::cerr << "warning! you are using less ranks then existing\n";
  if (specified_ranks > total_ranks) throw std::runtime_error("you are using more ranks then existing");

  return opts;
}

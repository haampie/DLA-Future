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

int hpx_main(int argc, char **argv) {
  for(auto i = 0; i < argc; ++i)
    std::cout << "[" << i << "] " << argv[i] << std::endl;

  // init communicators
  dlaf::comm::Communicator world(MPI_COMM_WORLD);

  std::cout << world.rank() << std::endl;

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
  hpx::program_options::options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  desc_commandline.add_options()
    ("matrix",
     hpx::program_options::value<std::uint64_t>()->default_value(10),
     "n value for the Fibonacci function")
    ;
  // options
  /*
    ("use-pools,u",     "Enable advanced HPX thread pools and executors")
    ("use-scheduler,s", "Enable custom priority scheduler")
    ("use-numa,a",      "Enable numa sensitive scheduling")
    ("mpi-threads,m", util::po::value<int>()->default_value(1),
      "Number of threads to assign to MPI")
    ("hp-queues,H", util::po::value<int>()->default_value(1),
      "Number of high priority queues to use in custom scheduler")
    ("size,n", util::po::value<int>()->default_value(4096), "Matrix size.")
    ("nb", util::po::value<int>()->default_value(128),
      "Block cyclic distribution size.")
    ("row-proc,p", util::po::value<int>()->default_value(1),
      "Number of row processes in the 2D communicator.")
    ("col-proc,q", util::po::value<int>()->default_value(1),
      "Number of column processes in the 2D communicator.")
    ("nruns", util::po::value<int>()->default_value(1), "number of runs")
    ("no-check", "Disable result checking");
   */

  auto ret_code = hpx::init(hpx_main, argc, argv);

  // resources management/scheduler/pool

  MPI_Finalize();

  return ret_code;
}

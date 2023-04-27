#pragma once

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define MPICHECK(cmd) do {                                             \
  int e = cmd;                                                         \
  if( e != MPI_SUCCESS ) {                                             \
    std::cout << " MPI error %"                                        \
      << __FILE__ << ":" << __LINE__ << " " << e << std::endl;         \
    exit(EXIT_FAILURE);                                                \
  }                                                                    \
} while(0)

#define CUDACHECK(cmd) do {                                                  \
  cudaError_t e = cmd;                                                       \
  if (e != cudaSuccess) {                                                    \
    std::string msg = "Cuda error " __FILE__ ":" + std::to_string(__LINE__)  \
      + " " +  cudaGetErrorName(e) + ": " + cudaGetErrorString(e);           \
    throw(EFAException(msg));                                                \
  }                                                                          \
} while(0)

class EFAException : public std::exception {
private:
  std::string msg;	

public:
  EFAException(std::string s) {
    msg = s;
  }

  std::string what() {
    return msg;
  }
};

#define INCLUDE(msg, x) (msg += std::string(#x) + std::string(" is ") + std::to_string(x) + std::string("; "));


#define EFA_ASSERT(expr, msg) if (!(expr)) {                           \
  msg = std::string(#expr) + " -- " + msg;                             \
  throw(EFAException(msg));                                            \
}

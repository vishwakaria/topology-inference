#pragma once

#include <memory>
#include <vector>

#include <mpi.h>

#include "rdma_context.h"
#include "send_request_pool.h"
#include "recv_request_pool.h"

class RdmaResources {
public:
  struct Address {
    Address(): ah(nullptr) {}
    union ibv_gid gid;
    uint32_t qpn[16];
    uint32_t qkey;
    struct ibv_ah *ah;
  };

  struct FederatedAddress {
    struct Address efaAddress[4];
  };

  RdmaResources() = default;
  ~RdmaResources();

  // Initialize all of the rdma resources.
  void init(int efa_idx, int num_qp, size_t send_buff_size, size_t recv_buff_size,
    int read_from_buf_size, int read_into_buf_size);
  
  RDMAContext ctx_;
  size_t send_buff_size_, recv_buff_size_;
  void *send_buf[2], *recv_buf[2];
  void *read_from_buf[2], *read_into_buf[2]; 
  ibv_mr *mr_send[2];
  ibv_mr *mr_recv[2];
  ibv_mr *mr_read_from[2];
  ibv_mr *mr_read_into[2];

  Address addr;
  
  SendRequestPool send_request_pool_;
  RecvRequestPool recv_request_pool_;
};
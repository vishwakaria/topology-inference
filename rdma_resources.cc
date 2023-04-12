#include <cassert>

#include <cuda_runtime.h>
#include <stdio.h>

#include "errorcheck.h"
#include "rdma_resources.h"
#include "errorcheck.h"

#include <thread>
#include <vector>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <thread>
#include <pthread.h>

#include <chrono>

void RdmaResources::init(int efa_idx, int num_qp, size_t send_buff_size, size_t recv_buff_size,
  int read_from_buff_size, int read_into_buff_size) {
  send_buff_size_ = send_buff_size;
  recv_buff_size_ = recv_buff_size;
  ctx_.init(efa_idx, num_qp);

  // GID table entries are created whenever an IP address is configured on one of the Ethernet devices of the NIC's ports
  ibv_query_gid(ctx_.context, ctx_.ibPort, 0, &(addr.gid));
    
  std::string error_msg;
  INCLUDE(error_msg, ctx_.num_queue_pairs);
  INCLUDE(error_msg, sizeof(addr.qpn));
  EFA_ASSERT(ctx_.num_queue_pairs <= sizeof(addr.qpn), error_msg);

  for (int qp_idx = 0; qp_idx < ctx_.num_queue_pairs; qp_idx++) {
    addr.qpn[qp_idx] = ctx_.queuePair[qp_idx]->qp_num;
  }
  addr.qkey = ctx_.MAGIC_QKEY;
  // std::cout << "done before mem reg" << std::endl;

  // each EFA is used by two gpus, register memory for all buffers
  for (int i = 0; i < 2; i++) {
    int flags = IBV_ACCESS_LOCAL_WRITE;
    int flags_with_read = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
    CUDACHECK(cudaSetDevice(efa_idx * 2 + i));
    CUDACHECK(cudaMalloc(&send_buf[i], send_buff_size));
    CUDACHECK(cudaMalloc(&recv_buf[i], recv_buff_size));
    mr_send[i] = ibv_reg_mr(ctx_.protectionDomain, send_buf[i], send_buff_size, flags);
    mr_recv[i] = ibv_reg_mr(ctx_.protectionDomain, recv_buf[i], recv_buff_size, flags);
    CUDACHECK(cudaMalloc(&read_from_buf[i], read_from_buff_size));
    CUDACHECK(cudaMalloc(&read_into_buf[i], read_into_buff_size));
    mr_read_from[i] = ibv_reg_mr(ctx_.protectionDomain, read_from_buf[i], read_from_buff_size, flags_with_read);
    mr_read_into[i] = ibv_reg_mr(ctx_.protectionDomain, read_into_buf[i], read_into_buff_size, flags_with_read);
    error_msg = "Unable to create memory registration";
    EFA_ASSERT(mr_send[i] != nullptr, error_msg);
    EFA_ASSERT(mr_recv[i] != nullptr, error_msg);
    EFA_ASSERT(mr_read_from[i] != nullptr, error_msg);
    EFA_ASSERT(mr_read_into[i] != nullptr, error_msg);
  }
}

RdmaResources::~RdmaResources() {
}

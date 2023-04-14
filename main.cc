#include <iostream>
#include <algorithm>
#include <iomanip>
#include <stdio.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <map>
#include <vector>
#include <chrono>
#include <sched.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <thread>
#include "errorcheck.h"
#include <atomic>
#include <cassert>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <unistd.h>
#include <fstream>
#include "rdma_resources.h"
#include "instance_id.h"

const size_t packet_size = 8*1024;
const int num_efa = 4; // p4d
const int num_gpu = 8;
const int num_gpu_per_efa = 2;
const int num_iter = 10000;
const int warmup_iter = 100;
const int efa_idx = 0;
const int gpu_idx = 0;
const int qp_idx = 0;
// TODO: Generate log files on each node
// TODO: Add doc strings

std::string output_dir;
// std::ofstream log, result;

int world_rank, world_size, local_rank, local_size, num_nodes;

void exit_and_close_files(int ret) {
  // log.close();
  // result.close();
  exit(ret);
}

struct WCStatus {
  ibv_wc_status status;
  int enqueue_error;
  bool completed;
  bool enqueued;
};

struct GPUPair {
  WCStatus send_status;
  WCStatus recv_status;
  WCStatus read_status;
  bool ah_created;
};

int getWorldRank() {
  return atoi(std::getenv("OMPI_COMM_WORLD_RANK"));
}

std::vector<GPUPair> gpu_pairs;
std::vector<double> peer_read_latency;

class FederatedRdmaClient {
public:
  FederatedRdmaClient() = default;
  ~FederatedRdmaClient() = default;

  int allocateResources() {
    int send_buf_size = world_size * packet_size;
    int recv_buf_size = world_size * packet_size;
    int read_from_buf_size = world_size * packet_size;
    int read_into_buf_size = world_size * packet_size;

    std::string bootstrap_error;
    try {
      rdmaResources.init(efa_idx, num_gpu_per_efa, send_buf_size, recv_buf_size, read_from_buf_size, read_into_buf_size);
    } catch (EFAException& e) {
      bootstrap_error += "Node " + std::to_string(world_rank) + " EFA " + std::to_string(efa_idx)
    + " failed to bootstrap with false assertion: " + e.what() + "\n";
      throw std::runtime_error(bootstrap_error);
    }
    return 0;
  }

  void init_records() {
    gpu_pairs.resize(num_gpu * world_size);
    for (auto& pair : gpu_pairs) {
      pair.ah_created = false;
      pair.read_status.enqueued = false;
      pair.read_status.completed = false;
    }
    for (int gpu_id = 0; gpu_id < num_gpu; gpu_id ++) {
      auto& pair = gpu_pairs[gpu_id * world_size + world_rank];
      pair.ah_created = true;
      pair.read_status.enqueued = true;
      pair.read_status.completed = true;
      pair.read_status.status = IBV_WC_SUCCESS;
    }
  }

  void create_address_handles() {
    RdmaResources::Address address = rdmaResources.addr;
    addresses.resize(world_size);
    MPI_Allgather(
      &address, sizeof(address), MPI_BYTE,
      addresses.data(), sizeof(address), MPI_BYTE, MPI_COMM_WORLD);

    for (int peer_idx = 0; peer_idx < world_size; peer_idx ++) {
      if (peer_idx == world_rank) continue;
      ibv_ah_attr ahAttr;
      memset(&ahAttr, 0, sizeof(ahAttr));
      ahAttr.is_global = 1;
      ahAttr.port_num = 1;
      ahAttr.grh.dgid = addresses[peer_idx].gid;
      addresses[peer_idx].ah = ibv_create_ah(
        rdmaResources.ctx_.protectionDomain, &ahAttr);
      gpu_pairs[(efa_idx * num_gpu_per_efa) * world_size + peer_idx].ah_created =
        addresses[peer_idx].ah != nullptr;
      gpu_pairs[(efa_idx * num_gpu_per_efa + 1) * world_size + peer_idx].ah_created =
        addresses[peer_idx].ah != nullptr;
    }
  }

  struct RDMAReadInfo {
    void* base_addr;
    uint32_t rkey;
  };

  std::vector<RDMAReadInfo> rdma_read_info;
  void all_gather_read_from_bufs() {
    rdma_read_info.resize(world_size);
    rdma_read_info[gpu_idx].base_addr = rdmaResources.read_from_buf[qp_idx];
    rdma_read_info[gpu_idx].rkey = rdmaResources.mr_read_from[qp_idx]->rkey;
    MPI_Allgather(
        rdma_read_info.data(), sizeof(RDMAReadInfo) * num_gpu, MPI_BYTE,
        rdma_read_info.data(), sizeof(RDMAReadInfo) * num_gpu, MPI_BYTE, MPI_COMM_WORLD);
  }

  bool post_read_request_to_efa(int peer_idx, size_t offset) {
    ibv_qp_ex *qpEx = rdmaResources.ctx_.queuePairEx[qp_idx];
    ibv_wr_start(qpEx);
    ibv_sge list;
    list.addr = (uint64_t)rdmaResources.read_into_buf[qp_idx] + offset;
    list.length = packet_size;
    list.lkey = rdmaResources.mr_read_into[qp_idx]->lkey;
    qpEx->wr_id = (uint64_t)peer_idx;
    ibv_wr_rdma_read(qpEx, rdma_read_info[peer_idx * num_gpu + gpu_idx].rkey,
      (uint64_t)rdma_read_info[peer_idx * num_gpu + gpu_idx].base_addr + offset);
    ibv_wr_set_sge_list(qpEx, 1, &list);
    ibv_wr_set_ud_addr(qpEx, addresses[peer_idx].ah,
      addresses[peer_idx].qpn[qp_idx],
      addresses[peer_idx].qkey);
    int ret = ibv_wr_complete(qpEx);
    return ret == 0;
  }

  int read_count = 0;
  std::chrono::steady_clock::time_point poll_peer_read_completion() {
    int read_from;
    bool more_to_poll = true;
    while (more_to_poll) {
      more_to_poll = false;
      int num_entries = ibv_poll_cq(rdmaResources.ctx_.sendCQ[qp_idx],
            completion_burst_size_, send_work_completions_.data());
      assert(num_entries >= 0 && "ibv_poll_cq returned negative value");
      read_count += num_entries;
      if (num_entries == 1) {
        return std::chrono::steady_clock::now();
      }
      for (int i = 0; i < num_entries; i ++) {
        read_from = (int) send_work_completions_[i].wr_id;
        gpu_pairs[gpu_idx * world_size + read_from].read_status.completed = true;
        gpu_pairs[gpu_idx * world_size + read_from].read_status.status =
          send_work_completions_[i].status;
      }
      more_to_poll |= (num_entries > 0);
    }
  }

  void compute_peer_read_lat() {
    size_t offset;
    int node_id = world_rank / local_size;
    int peer_idx;
    std::vector<double> all_read_latencies;
    double avg_read_latency;
    peer_read_latency.resize(num_nodes);
    for (int peer_node_idx = 0; peer_node_idx < num_nodes; peer_node_idx ++) {
      if (peer_node_idx == node_id) {
        peer_read_latency[peer_node_idx] = 0;
        continue;
      }
      peer_idx = peer_node_idx * local_size + gpu_idx;
      for (int iter = 0; iter < num_iter; iter ++) {
        auto begin = std::chrono::steady_clock::now();
        auto& pair = gpu_pairs[world_size * gpu_idx + peer_idx];
        if (!pair.ah_created) continue;
        offset = peer_idx * packet_size;
        pair.read_status.enqueued = post_read_request_to_efa(peer_idx, offset);
        auto end = poll_peer_read_completion();
        double timeMicroSeconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        if (iter >= warmup_iter) {
          all_read_latencies.push_back(timeMicroSeconds);
        }
      }
      avg_read_latency = avg_without_outliers(all_read_latencies);
      peer_read_latency[peer_node_idx] = avg_read_latency;
      all_read_latencies.clear();
    }
  }

  double avg_without_outliers(std::vector<double>& latencies) {
    std::sort(latencies.begin(), latencies.end());
    int p90_size = latencies.size() * 0.9;
    // std::cout << "averaging " << p90_size << " elements" << std::endl;
    // if (world_rank == 0) {
    //   for (int i = 0; i < p90_size; i++) {
    //     std::cout << i << ": " << latencies[i] << std::endl;
    //   }
    // }
    return std::reduce(latencies.begin(), latencies.begin() + p90_size, 0.0) / p90_size;
  }

  RdmaResources rdmaResources;
  std::vector<RdmaResources::Address> addresses;
  const int completion_burst_size_ = 1024;
  std::vector<ibv_wc> send_work_completions_{completion_burst_size_};
  std::vector<ibv_wc> recv_work_completions_{completion_burst_size_};
};

void gather_instance_id_to_rank_0() {
  // Instance id is 19 characters long
  std::string instance_id = get_instance_id();
  instance_id += "\0                             "; // 30 spaces padding to avoid illegal mem access
  char instance_ids[world_size][30];
  MPI_Gather(instance_id.c_str(), 30, MPI_BYTE, instance_ids, 30, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (world_rank == 0) {
    std::cout << "Instance id mapping: " << std::endl;
    for (int i = 0; i < world_size; i++) {
      std::cout << "Node " << i << " - " << instance_ids[i] << std::endl;
    }
  }
}

void gather_results_to_rank_0(MPI_Comm comm) {
  if (world_rank == 0)
    peer_read_latency.resize(num_nodes * num_nodes);

  MPI_Gather(peer_read_latency.data(), num_nodes, MPI_DOUBLE, 
    peer_read_latency.data(), num_nodes, MPI_DOUBLE, 0, comm);
}

void crash_process(int time_in_seconds) {
  sleep(time_in_seconds);
  std::cout << "Process has run for " << time_in_seconds
    << " seconds now. Crashing to prevent hang." << std::endl;
  exit_and_close_files(2);
}

int main(int argc, char** argv) {
  // --- Set a background thread to crash process in 30 seconds to prevent hang
  std::thread crash_thread(crash_process, 30);
  crash_thread.detach();

  //--- Init MPI
  setenv("FI_EFA_FORK_SAFE", "1", 1);
  std::cout << "Initializing MPI..." << std::endl;
  MPI_Init(&argc, &argv);
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  local_size = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE"));
  local_rank = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  std::cout << world_rank << ": pid = " << getpid() << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  // revisit this
  assert(world_size <= 1024); // this is the limit on ah

  //--- Open log file
  // Use the second argument as output dir
  if (argc > 1) {
    output_dir = argv[1];
    output_dir += "/";
  }

  //--- Construct instance id mapping on node 0
  gather_instance_id_to_rank_0();

  //--- Booststrap EFA devices
  std::cout << world_rank << ": Loading EFA devices..." << std::endl;
  FederatedRdmaClient driver;
  driver.allocateResources();
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << world_rank << ":Done Loading EFA devices..." << std::endl;

  //--- Creating remote address handles
  std::cout << world_rank << ": All EFA devices have been loaded. Creating address handles now..." << std::endl;
  driver.init_records();
  driver.create_address_handles();
  std::cout << world_rank << ": Done Creating address handles" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  //--- Compute read latencies
  std::cout << world_rank << ": Executing read from peers now." << std::endl;
  num_nodes = world_size / local_size;
  driver.all_gather_read_from_bufs();
  driver.compute_peer_read_lat();
  MPI_Barrier(MPI_COMM_WORLD);

  //--- Gather results across all nodes on rank 0
  gather_results_to_rank_0(MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0) {
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Latency measurements across nodes (microseconds): " << std::endl;
    std::cout << std::setprecision(2) << std::fixed;
    for (int i = 0; i < num_nodes; i ++) {
      for (int j = 0; j < num_nodes; j ++) {
        std::cout << peer_read_latency[num_nodes * i + j] << "\t";
      }
      std::cout << std::endl;
    } 
  }

  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << world_rank << ": calling MPI finalize now " << std::endl;
  MPI_Finalize();
  return 0;
}

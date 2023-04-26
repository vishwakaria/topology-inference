#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <set>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include "errorcheck.h"
#include "rdma_resources.h"
#include "instance_id.h"

namespace defaults {
  const size_t packet_size = 8;
  const int num_iter = 1000;
  const int warmup_iter = 100;
  const int spine_latency_threshold = 60;
  const int num_buckets = 2;
  const int num_efa_needed = 2;
}

int num_gpu = 8, num_gpu_per_efa = 2;
// const int efa_idx = 0, qp_idx = 0;
// const int gpu_idx = 0;
int world_rank, world_size, local_rank, local_size, num_nodes;
std::string output_dir;

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

struct PairwiseLatency {
  int curr_node_idx;
  int peer_node_idx;
  double latency;
};

class FederatedRdmaClient {
public:
  FederatedRdmaClient() = default;
  ~FederatedRdmaClient() = default;

  int allocateResources() {
    int send_buf_size = world_size * defaults::packet_size;
    int recv_buf_size = world_size * defaults::packet_size;
    int read_from_buf_size = world_size * defaults::packet_size;
    int read_into_buf_size = world_size * defaults::packet_size;

    std::string bootstrap_error;
    for (int efa_idx = 0; efa_idx < defaults::num_efa_needed; efa_idx ++) {
      try {
        rdmaResources[efa_idx].init(efa_idx, num_gpu_per_efa, send_buf_size, recv_buf_size, read_from_buf_size, read_into_buf_size);
      } catch (EFAException& e) {
        bootstrap_error += "Node " + std::to_string(world_rank) + " EFA " + std::to_string(efa_idx)
      + " failed to bootstrap with false assertion: " + e.what() + "\n";
        throw std::runtime_error(bootstrap_error);
      }
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
    RdmaResources::FederatedAddress federatedAddress;
    for (int efa_idx = 0; efa_idx < defaults::num_efa_needed; efa_idx++) {
      federatedAddress.efaAddress[efa_idx] = rdmaResources[efa_idx].addr;
    }
    federatedAddresses.resize(world_size);
    MPI_Allgather(
      &federatedAddress, sizeof(federatedAddress), MPI_BYTE,
      federatedAddresses.data(), sizeof(federatedAddress), MPI_BYTE, MPI_COMM_WORLD);

    for (int peer_idx = 0; peer_idx < world_size; peer_idx ++) {
      if (peer_idx == world_rank) continue;
      for (int efa_idx = 0; efa_idx < defaults::num_efa_needed; efa_idx ++) {
        ibv_ah_attr ahAttr;
        memset(&ahAttr, 0, sizeof(ahAttr));
        ahAttr.is_global = 1;
        ahAttr.port_num = 1;
        ahAttr.grh.dgid = federatedAddresses[peer_idx].efaAddress[efa_idx].gid;
        federatedAddresses[peer_idx].efaAddress[efa_idx].ah = ibv_create_ah(
          rdmaResources[efa_idx].ctx_.protectionDomain, &ahAttr);
        gpu_pairs[(efa_idx * num_gpu_per_efa) * world_size + peer_idx].ah_created =
          federatedAddresses[peer_idx].efaAddress[efa_idx].ah != nullptr;
        gpu_pairs[(efa_idx * num_gpu_per_efa + 1) * world_size + peer_idx].ah_created =
          federatedAddresses[peer_idx].efaAddress[efa_idx].ah != nullptr;
      }
    }
  }

  void all_gather_read_from_bufs() {
    rdma_read_info.resize(world_size * num_gpu);
    std::vector<FederatedRdmaClient::RDMAReadInfo> local_rdma_read_info(num_gpu);
    for (int gpu_idx = 0; gpu_idx < num_gpu; gpu_idx ++) {
      int efa_idx = gpu_idx / num_gpu_per_efa;
      int qp_idx = gpu_idx % num_gpu_per_efa;
      local_rdma_read_info[gpu_idx].base_addr = rdmaResources[efa_idx].read_from_buf[qp_idx];
      local_rdma_read_info[gpu_idx].rkey = rdmaResources[efa_idx].mr_read_from[qp_idx]->rkey;
    }
    MPI_Allgather(
        local_rdma_read_info.data(), sizeof(RDMAReadInfo) * num_gpu, MPI_BYTE,
        rdma_read_info.data(), sizeof(RDMAReadInfo) * num_gpu, MPI_BYTE, MPI_COMM_WORLD);
  }

  bool post_read_request_to_efa(int peer_idx, int gpu_idx, size_t offset) {
    int node_id = world_rank / local_size;
    int efa_idx = gpu_idx / num_gpu_per_efa;
    int qp_idx = gpu_idx % num_gpu_per_efa;
    // if (node_id == 5) std::cout << "Enqueuing read request to peer " << peer_idx  << " on EFA " << efa_idx << "\n";
    ibv_qp_ex *qpEx = rdmaResources[efa_idx].ctx_.queuePairEx[qp_idx];
    ibv_wr_start(qpEx);
    ibv_sge list;
    list.addr = (uint64_t)rdmaResources[efa_idx].read_into_buf[qp_idx] + offset;
    list.length = defaults::packet_size;
    list.lkey = rdmaResources[efa_idx].mr_read_into[qp_idx]->lkey;
    qpEx->wr_id = (uint64_t)peer_idx;
    ibv_wr_rdma_read(qpEx, rdma_read_info[peer_idx * num_gpu + gpu_idx].rkey,
      (uint64_t)rdma_read_info[peer_idx * num_gpu + gpu_idx].base_addr + offset);
    ibv_wr_set_sge_list(qpEx, 1, &list);
    ibv_wr_set_ud_addr(qpEx, federatedAddresses[peer_idx].efaAddress[efa_idx].ah,
      federatedAddresses[peer_idx].efaAddress[efa_idx].qpn[qp_idx],
      federatedAddresses[peer_idx].efaAddress[efa_idx].qkey);
    int ret = ibv_wr_complete(qpEx);
    return ret == 0;
  }

  std::chrono::steady_clock::time_point poll_read_completion(int gpu_idx) {
    int efa_idx = gpu_idx / num_gpu_per_efa;
    int qp_idx = gpu_idx % num_gpu_per_efa;
    int node_id = world_rank / local_size;
    int read_count = 0, read_from;
    bool more_to_poll = true;
    while (more_to_poll) {
      more_to_poll = false;
      int num_entries = ibv_poll_cq(rdmaResources[efa_idx].ctx_.sendCQ[qp_idx],
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

  PairwiseLatency measure_read_latency(int peer_idx, int gpu_idx) {
    int node_id = world_rank / local_size;
    size_t offset;
    std::vector<double> all_read_latencies;
    for (int iter = 0; iter < defaults::num_iter; iter ++) {
      auto& pair = gpu_pairs[world_size * gpu_idx + peer_idx];
      if (!pair.ah_created) {
        std::cout << "AH not created yet" << std::endl;
        continue;
      }
      offset = peer_idx * defaults::packet_size;
      auto begin = std::chrono::steady_clock::now();
      pair.read_status.enqueued = post_read_request_to_efa(peer_idx, gpu_idx, offset);
      auto end = poll_read_completion(gpu_idx);
      double timeMicroSeconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
      if (iter >= defaults::warmup_iter) {
        all_read_latencies.push_back(timeMicroSeconds);
      }
    }
    double avg =  avg_without_outliers(all_read_latencies);
    std::cout << world_rank << ": Avg: "  << avg << std::endl;
    PairwiseLatency pair = {node_id, peer_idx, avg};
    return pair;
  }

  double avg_without_outliers(std::vector<double>& latencies) {
    std::sort(latencies.begin(), latencies.end());
    int p90_size = latencies.size();
    return std::reduce(latencies.begin(), latencies.begin() + p90_size, 0.0) / p90_size;
  }

private:
  struct RDMAReadInfo {
    void* base_addr;
    uint32_t rkey;
  };

  RdmaResources rdmaResources[defaults::num_efa_needed];
  std::vector<RdmaResources::FederatedAddress> federatedAddresses;
  const int completion_burst_size_ = 1024;
  std::vector<ibv_wc> send_work_completions_{completion_burst_size_};
  std::vector<ibv_wc> recv_work_completions_{completion_burst_size_};
  std::vector<GPUPair> gpu_pairs;
  std::vector<RDMAReadInfo> rdma_read_info;
};

std::vector<int> topology_mapping;
bool no_buckets_assigned = true;

void init_topology_mapping() {
  topology_mapping.resize(num_nodes);
  for (int i = 0; i < num_nodes; i ++) {
    topology_mapping[i] = -1;
  }
  // std::cout << "Topology mapping initialized" << std::endl;
}

void assign_topology_bucket(PairwiseLatency& pair) {
  // std::cout << "Assigning buckets for " << pair.curr_node_idx << " and " << pair.peer_node_idx << std::endl;
  int nodeBucket = topology_mapping[pair.curr_node_idx];
  int peerNodeBucket = topology_mapping[pair.peer_node_idx];
  // If both nodes are assigned, we are processing the last pair. Return.
  if (nodeBucket != -1 && peerNodeBucket != -1) {
    return;
  }
  
  // If current node is unassigned, we are processing the first pair.
  // Randomly assign current node to bucket 0.
  if (nodeBucket == -1) {
    topology_mapping[pair.curr_node_idx] = 0;
  }

  bool same_spine = (pair.latency > defaults::spine_latency_threshold) ? false : true;
  if (same_spine) {
    topology_mapping[pair.peer_node_idx] = topology_mapping[pair.curr_node_idx];
  } else {
    topology_mapping[pair.peer_node_idx] = 1 - topology_mapping[pair.curr_node_idx];
  }
}

void printBuckets() {
  std::cout << world_rank << ": Buckets: ";
  for (int i = 0; i < topology_mapping.size(); i ++) {
    std::cout << topology_mapping[i] << " ";
  }
  std::cout << std::endl;
}

void compute_topology(FederatedRdmaClient& driver) {
  PairwiseLatency lat_pair;
  init_topology_mapping();

  // Each node reads from the next in ring
  // All even nodes read from the next in ring on EFA 0, GPU 0
  // All odd nodes  read from the next in ring on EFA 1, GPU 2ß
  int next_in_ring = (world_rank + 1) % num_nodes;
  if (world_rank % 2 == 0) {
    lat_pair = driver.measure_read_latency(next_in_ring, 0);
  } else {
    lat_pair = driver.measure_read_latency(next_in_ring, 2);
  }

  // Node 0 receives latency pairs from all other nodes and assigns buckets
  if (world_rank != 0) {
    MPI_Send(&lat_pair, sizeof(lat_pair), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
  } else {
    assign_topology_bucket(lat_pair);
    for (int i = 1; i < num_nodes; i++) {
      MPI_Recv(&lat_pair, sizeof(lat_pair), MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      assign_topology_bucket(lat_pair);
    }
  }
}

void broadcast_topology_mapping() {
  topology_mapping.resize(num_nodes);
  MPI_Bcast(topology_mapping.data(), topology_mapping.size(), MPI_INT, 0, MPI_COMM_WORLD);
}

void write_topology_to_file(std::string output_dir) {
  broadcast_topology_mapping();
  
  std::ofstream outfile;
  std::string filename = output_dir + "cluster_topology.txt";
  std::cout << "Writing topology to " << filename << std::endl;
  outfile.open(filename);
  if (!outfile) {
    std::cout << "Error: Could not open result file " << filename << std::endl;
  }
  for (int i = 0; i < topology_mapping.size(); i ++) {
    outfile << "algo-" << std::to_string(i+1) << " spine" << std::to_string(topology_mapping[i] + 1) << std::endl;
  }
  outfile.close();
}

int main(int argc, char** argv) {
  //--- Init MPI
  setenv("FI_EFA_FORK_SAFE", "1", 1);
  std::cout << "Initializing MPI..." << std::endl;
  MPI_Init(&argc, &argv);
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  local_size = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE"));
  local_rank = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  num_nodes = world_size/local_size;
  std::cout << world_rank << ": " << num_nodes << ": pid = " << getpid() << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  if (argc > 1) {
    output_dir = argv[1];
    output_dir += "/";
    std::cout << "output_dir = " << output_dir << std::endl;
  }
    
  //--- Booststrap EFA devices
  std::cout << world_rank << ": Loading EFA devices..." << std::endl;
  FederatedRdmaClient driver;
  driver.allocateResources();
  MPI_Barrier(MPI_COMM_WORLD);

  //--- Creating remote address handles
  std::cout << world_rank << ": All EFA devices have been loaded. Creating address handles now..." << std::endl;
  driver.init_records();
  driver.create_address_handles();
  MPI_Barrier(MPI_COMM_WORLD);

  //--- Compute topology mapping for nodes
  std::cout << world_rank << ": Computing tolopogy information for the cluster..." << std::endl;
  driver.all_gather_read_from_bufs();
  auto begin = std::chrono::steady_clock::now();
  compute_topology(driver);
  auto end = std::chrono::steady_clock::now();
  double timeMilliSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  if (world_rank == 0) {
    printBuckets();
    std::cout <<  "Compute time: " << timeMilliSeconds << " ms" << std::endl;
  }
  // Write topology mapping to output directory passed in args
  write_topology_to_file(output_dir);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <set>
#include <stdlib.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "errorcheck.h"
#include "rdma_resources.h"
#include "instance_id.h"

namespace defaults {
  const size_t packet_size = 8;
  const int num_iter = 1000;
  const int warmup_iter = 100;
  const int spine_latency_threshold_us = 60;
  const int num_efa = 4;
  const int num_gpu = 8, num_gpu_per_efa = 2;
}

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

class FederatedRdmaClient {
public:
  FederatedRdmaClient() = default;
  ~FederatedRdmaClient() = default;

  int allocate_resources() {
    int send_buf_size = world_size * defaults::packet_size;
    int recv_buf_size = world_size * defaults::packet_size;
    int read_from_buf_size = world_size * defaults::packet_size;
    int read_into_buf_size = world_size * defaults::packet_size;

    std::string bootstrap_error;
    for (int efa_idx = 0; efa_idx < defaults::num_efa; efa_idx ++) {
      try {
        rdmaResources[efa_idx].init(efa_idx, defaults::num_gpu_per_efa, send_buf_size, recv_buf_size, read_from_buf_size, read_into_buf_size);
      } catch (EFAException& e) {
        bootstrap_error += "Node " + std::to_string(world_rank) + " EFA " + std::to_string(efa_idx)
      + " failed to bootstrap with false assertion: " + e.what() + "\n";
        throw std::runtime_error(bootstrap_error);
      }
    }
    return 0;
  }

  void init_records() {
    gpu_pairs.resize(defaults::num_gpu * world_size);
    for (auto& pair : gpu_pairs) {
      pair.ah_created = false;
      pair.read_status.enqueued = false;
      pair.read_status.completed = false;
    }
    for (int gpu_id = 0; gpu_id < defaults::num_gpu; gpu_id ++) {
      auto& pair = gpu_pairs[gpu_id * world_size + world_rank];
      pair.ah_created = true;
      pair.read_status.enqueued = true;
      pair.read_status.completed = true;
      pair.read_status.status = IBV_WC_SUCCESS;
    }
    return;
  }

  void create_address_handles() {
    RdmaResources::FederatedAddress federatedAddress;
    for (int efa_idx = 0; efa_idx < defaults::num_efa; efa_idx++) {
      federatedAddress.efaAddress[efa_idx] = rdmaResources[efa_idx].addr;
    }
    federatedAddresses.resize(world_size);
    MPI_Allgather(
      &federatedAddress, sizeof(federatedAddress), MPI_BYTE,
      federatedAddresses.data(), sizeof(federatedAddress), MPI_BYTE, MPI_COMM_WORLD);

    for (int peer_idx = 0; peer_idx < world_size; peer_idx ++) {
      if (peer_idx == world_rank) continue;
      for (int efa_idx = 0; efa_idx < defaults::num_efa; efa_idx ++) {
        ibv_ah_attr ahAttr;
        memset(&ahAttr, 0, sizeof(ahAttr));
        ahAttr.is_global = 1;
        ahAttr.port_num = 1;
        ahAttr.grh.dgid = federatedAddresses[peer_idx].efaAddress[efa_idx].gid;
        federatedAddresses[peer_idx].efaAddress[efa_idx].ah = ibv_create_ah(
          rdmaResources[efa_idx].ctx_.protectionDomain, &ahAttr);
        gpu_pairs[(efa_idx * defaults::num_gpu_per_efa) * world_size + peer_idx].ah_created =
          federatedAddresses[peer_idx].efaAddress[efa_idx].ah != nullptr;
        gpu_pairs[(efa_idx * defaults::num_gpu_per_efa + 1) * world_size + peer_idx].ah_created =
          federatedAddresses[peer_idx].efaAddress[efa_idx].ah != nullptr;
      }
    }
    return;
  }

  void all_gather_read_from_bufs() {
    rdma_read_info.resize(world_size * defaults::num_gpu);
    std::vector<FederatedRdmaClient::RDMAReadInfo> local_rdma_read_info(defaults::num_gpu);
    for (int gpu_idx = 0; gpu_idx < defaults::num_gpu; gpu_idx ++) {
      int efa_idx = gpu_idx / defaults::num_gpu_per_efa;
      int qp_idx = gpu_idx % defaults::num_gpu_per_efa;
      local_rdma_read_info[gpu_idx].base_addr = rdmaResources[efa_idx].read_from_buf[qp_idx];
      local_rdma_read_info[gpu_idx].rkey = rdmaResources[efa_idx].mr_read_from[qp_idx]->rkey;
    }
    MPI_Allgather(
        local_rdma_read_info.data(), sizeof(RDMAReadInfo) * defaults::num_gpu, MPI_BYTE,
        rdma_read_info.data(), sizeof(RDMAReadInfo) * defaults::num_gpu, MPI_BYTE, MPI_COMM_WORLD);
    return;
  }

  bool post_read_request_to_efa(int peer_idx, int gpu_idx, size_t offset) {
    int efa_idx = gpu_idx / defaults::num_gpu_per_efa;
    int qp_idx = gpu_idx % defaults::num_gpu_per_efa;
    ibv_qp_ex *qpEx = rdmaResources[efa_idx].ctx_.queuePairEx[qp_idx];
    ibv_wr_start(qpEx);
    ibv_sge list;
    list.addr = (uint64_t)rdmaResources[efa_idx].read_into_buf[qp_idx] + offset;
    list.length = defaults::packet_size;
    list.lkey = rdmaResources[efa_idx].mr_read_into[qp_idx]->lkey;
    qpEx->wr_id = (uint64_t)peer_idx;
    ibv_wr_rdma_read(qpEx, rdma_read_info[peer_idx * defaults::num_gpu + gpu_idx].rkey,
      (uint64_t)rdma_read_info[peer_idx * defaults::num_gpu + gpu_idx].base_addr + offset);
    ibv_wr_set_sge_list(qpEx, 1, &list);
    ibv_wr_set_ud_addr(qpEx, federatedAddresses[peer_idx].efaAddress[efa_idx].ah,
      federatedAddresses[peer_idx].efaAddress[efa_idx].qpn[qp_idx],
      federatedAddresses[peer_idx].efaAddress[efa_idx].qkey);
    int ret = ibv_wr_complete(qpEx);
    return ret == 0;
  }

  std::chrono::steady_clock::time_point poll_read_completion(int gpu_idx) {
    int efa_idx = gpu_idx / defaults::num_gpu_per_efa;
    int qp_idx = gpu_idx % defaults::num_gpu_per_efa;
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

  double measure_read_latency(int peer_idx, int efa_idx = 0) {
    int gpu_idx = efa_idx * defaults::num_gpu_per_efa;
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
    // std::cout << world_rank << ": Read latency from peer " << peer_idx << ": "  << avg << std::endl;
    // return 100; // To simulate inter-spine latency for testing
    return avg;
  }

  double avg_without_outliers(std::vector<double>& latencies) {
    std::sort(latencies.begin(), latencies.end());
    int p90_size = latencies.size() *  0.9;
    return std::reduce(latencies.begin(), latencies.begin() + p90_size, 0.0) / p90_size;
  }

private:
  struct RDMAReadInfo {
    void* base_addr;
    uint32_t rkey;
  };

  RdmaResources rdmaResources[defaults::num_efa];
  std::vector<RdmaResources::FederatedAddress> federatedAddresses;
  const int completion_burst_size_ = 1024;
  std::vector<ibv_wc> send_work_completions_{completion_burst_size_};
  std::vector<ibv_wc> recv_work_completions_{completion_burst_size_};
  std::vector<GPUPair> gpu_pairs;
  std::vector<RDMAReadInfo> rdma_read_info;
};

class TopologyCalculator {
public:
  TopologyCalculator() = default;
  ~TopologyCalculator() = default;

  void compute_topology_single_efa(FederatedRdmaClient& driver) {
    init_topology_mapping();

    for (int i = 0; i < num_nodes; i ++) { 
      if (i == world_rank && node_to_spine_mapping[i] == -1) { 
        for (int j = 0; j < num_spines; j++) {
          int nodeInSpineJ = find_node_in_spine(j);
          if (is_same_spine(nodeInSpineJ, driver)) {
            assign_node_to_spine(j);
            break;
          }
        }
        if (node_to_spine_mapping[i] == -1) { 
          assign_node_to_spine(num_spines);
        }
      }
      broadcast_topology_mapping(i); // also acts as a barrier
      MPI_Barrier(MPI_COMM_WORLD);
    }
    return;
  }

  void compute_topology_multi_efa(FederatedRdmaClient& driver) {
    init_topology_mapping();

    for (int i = 1; i < num_nodes; i ++) { 
      if (i == world_rank && node_to_spine_mapping[i] == -1) {
        // std::cout << "num_spines for i = " << i << " is " << num_spines << std::endl; 
        std::vector<std::thread> threads(defaults::num_efa);
        std::vector<int> results(defaults::num_efa, 0);
        for (int j = 0; j < num_spines; j += defaults::num_efa) {
          for (int k = 0; k < defaults::num_efa && (k + j) < num_spines; k++) {
            int nodeInSpine = find_node_in_spine(k+j);
            threads[k] = std::thread(&TopologyCalculator::is_same_spine_threadfn, 
                              this, nodeInSpine, k, std::ref(driver), std::ref(results[k]));
          }
          for (int k = 0; k < defaults::num_efa && (k + j) < num_spines; k++) {
            threads[k].join();
            if (results[k]) {
              assign_node_to_spine(k+j);
            }
          }
        }
        if (node_to_spine_mapping[i] == -1) { 
          assign_node_to_spine(num_spines);
        }
      }
      broadcast_topology_mapping(i); // also acts as a barrier
      MPI_Barrier(MPI_COMM_WORLD);
    }
    return;
  }

  void write_topology_to_file() {
    std::ofstream outfile;
    std::string filename = output_dir + "node_to_spine.txt";
    std::cout << "Writing topology to " << filename << std::endl;
    outfile.open(filename);
    if (!outfile) {
      std::cout << "Error: Could not open result file " << filename << std::endl;
      exit(1);
    }
  
    for (uint32_t i = 0; i < num_nodes; i ++) {
      outfile << "algo-" << std::to_string(i+1) << " spine" << std::to_string(node_to_spine_mapping[i] + 1) << std::endl;
    }
    outfile.close();
    return;
  }

  void print_spines() {
    std::cout << "num_spines = " << num_spines << std::endl;
    for (uint32_t i = 0; i < num_nodes; i++) {
      std::cout << node_to_spine_mapping[i] << " ";
    }
    std::cout << std::endl;
    return;
  }

private:
  std::vector<int> node_to_spine_mapping;
  int num_spines = 1;

  void init_topology_mapping() {
    node_to_spine_mapping.resize(num_nodes);
    for (int i = 0; i < num_nodes; i ++) {
      node_to_spine_mapping[i] = -1;
    }
    node_to_spine_mapping[0] = 0;
    return;
  }
  
  bool is_same_spine(int peer_idx, FederatedRdmaClient& driver) {
    double lat = driver.measure_read_latency(peer_idx);
    return (lat > defaults::spine_latency_threshold_us) ? false : true;
  }

  void is_same_spine_threadfn(int peer_idx, int efa_idx, FederatedRdmaClient& driver, int& result) {
    double lat = driver.measure_read_latency(peer_idx, efa_idx);
    // std::cout << world_rank << ": lat = " << lat << "for peer " << peer_idx << " in efa " << efa_idx << std::endl;
    result = (lat > defaults::spine_latency_threshold_us) ? 0 : 1;
    // std::cout << world_rank << ": is same spine as peer " << peer_idx << ": " << result << std::endl;
  }

  int find_node_in_spine(int spine_idx) {
    for (int i = 0; i < node_to_spine_mapping.size(); i ++) {
      if (node_to_spine_mapping[i] == spine_idx) {
        // std::cout << world_rank << ": Found node " << i << " in spine " << spine_idx << std::endl;
        return i;
      }
    }
    return -1;
  }

  void assign_node_to_spine(int spine_idx) {
    // std::cout << world_rank << ": Assigning node to spine " << spine_idx << std::endl;
    node_to_spine_mapping[world_rank] = spine_idx;
    if (spine_idx == num_spines) {
      num_spines ++;
      // std::cout << world_rank << ": num_spines = " << num_spines << std::endl;
    }
    return;
  }

  void broadcast_topology_mapping(int root) {
    MPI_Bcast(node_to_spine_mapping.data(), num_nodes, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&num_spines, 1, MPI_INT, root, MPI_COMM_WORLD);
    return;
  }
};


int main(int argc, char** argv) {
  //--- Init MPI
  setenv("FI_EFA_FORK_SAFE", "1", 1);
  TopologyCalculator topology_calculator {};

  std::cout << "Initializing MPI..." << std::endl;
  MPI_Init(&argc, &argv);
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  local_size = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE"));
  local_rank = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  num_nodes = world_size/local_size;
  MPI_Barrier(MPI_COMM_WORLD);

  if (argc > 1) {
    output_dir = argv[1];
    output_dir += "/";
    std::cout << "output_dir = " << output_dir << std::endl;
  }
    
  //--- Booststrap EFA devices
  std::cout << world_rank << ": Loading EFA devices..." << std::endl;
  FederatedRdmaClient driver;
  driver.allocate_resources();
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
  // topology_calculator.compute_topology_single_efa(driver);
  topology_calculator.compute_topology_multi_efa(driver);
  auto end = std::chrono::steady_clock::now();
  double time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  std::cout <<  "Compute time: " << time_ms << " ms" << std::endl;
  if (world_rank == 0) {
    topology_calculator.print_spines();
  }
  // Write topology mapping to output directory passed in args
  topology_calculator.write_topology_to_file();

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}


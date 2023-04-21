#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
  std::cout << "Initializing MPI..." << std::endl;
  int world_rank, world_size;
  MPI_Init(&argc, &argv);
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  MPI_Barrier(MPI_COMM_WORLD);

  char top_map[world_size][30];

  // Read results from file 
  std::string output_dir;
  if (argc > 1) {
    output_dir = argv[1];
    output_dir += "/";
  }
  std::string result_file_name = output_dir + "topology_mapping.txt";
  if (world_rank == 0) {
    std::cout << "Reading results from " << result_file_name << std::endl;
    std::ifstream result_file (result_file_name, std::ifstream::in);
    if (result_file.is_open()) {
      std::string line;
      int i = 0;
      while (getline(result_file, line)) {
        line += "\0                             "; // 30 spaces padding to avoid illegal mem access
        top_map[i][0] = line;

      }
      result_file.close();
    }
  }
  MPI_Bcast(top_map.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
}
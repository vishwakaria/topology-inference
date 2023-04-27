#!/bin/bash

echo "----Calculating latency across the cluster----"
/fsx/viskaria/topology-inference/build/latency_calculator $1

echo "----Clustering nodes----"
world_rank=$OMPI_COMM_WORLD_RANK
if [[ $world_rank == 0 ]]; then
  python /fsx/viskaria/topology-inference/clustering/cluster_nodes.py
fi

# Broadcast results to all nodes
echo "Done. Topology information can be found in $1/topology_mapping.txt"
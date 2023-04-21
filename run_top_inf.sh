#!/bin/bash
USER=viskaria

num_nodes=${1:-64}

/fsx/viskaria/smprun $USER -n $num_nodes -v --mpi-path /opt/amazon/openmpi/bin/mpirun \
-d /fsx/viskaria/topology-inference -c viskaria_dev_latest /fsx/viskaria/topology-inference/topology_inference.sh /fsx/viskaria/topology-inference/results

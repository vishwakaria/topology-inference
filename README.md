# topology-inference

### Description
Infer EC2 cluster topology using RDMA reads

- Measure time taken for RDMA read on each peer node
- Cluster the read times to identify which nodes belong to the same spine

### Build
```
rm -rf build
mkdir build
cd build
cmake ..
make -j99
```

### Run
* `smddputils viskaria -hbuild -C dlc-pt1131 -src /fsx/viskaria/topology -script build.sh`
* `ssh compute-dy-p4d24xlarge-10 "mpirun --hostfile /fsx/viskaria/topology/hostlist -np 32 --oversubscribe --allow-run-as-root /fsx/viskaria/topology/build/main"`

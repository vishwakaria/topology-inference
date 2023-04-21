import os
import numpy as np
from sklearn_extra.cluster import KMedoids

latencies = []
with open ("/fsx/viskaria/topology-inference/results/latency_measurements.result", 'r') as f:
    for line in f.readlines():
        latencies.append(line.split())
print(latencies)

kmedoids = KMedoids(n_clusters=2, random_state=32).fit(latencies)
print(kmedoids.labels_.tolist())

# write results to file
hostlist=open('/fsx/viskaria/topology-inference/scripts/hostlist')
hosts=hostlist.readlines()
with open ("/fsx/viskaria/topology-inference/results/topology_mapping.txt", 'w') as f:
    for i in range(len(latencies)):
        f.write(hosts[i].strip() + " " + str(kmedoids.labels_[i]) + "\n")
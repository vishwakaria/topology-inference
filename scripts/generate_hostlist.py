hostfile = open('hostlist', 'w')
print('hello')
for i in range(86):
  host = "compute-st-worker-" + str(i) + " slots=1\n"
  hostfile.write(host)
hostfile.close()
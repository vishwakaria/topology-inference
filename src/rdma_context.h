#pragma once

#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <string>

#include "errorcheck.h"

class RDMAContext {
public:
  RDMAContext() = default;
  virtual ~RDMAContext() = default;
  int init(int efa_idx, int num_qp);
  void destroy();

  struct ibv_context *context = nullptr;
  struct ibv_pd *protectionDomain = nullptr;
  struct ibv_cq **sendCQ = nullptr;
  struct ibv_cq **recvCQ = nullptr;
  struct ibv_qp **queuePair = nullptr;
  struct ibv_qp_ex **queuePairEx = nullptr;
  int maxMsgSize;
  int num_queue_pairs;

  // config
  int txBufferDepth = -1;
  int rxBufferDepth = -1;
  int ibPort = 1;
  static const int MAGIC_QKEY = 1234;

private:
  struct efadv_device_attr efadvAttr;
  static inline int alignDownToPowerOfTwo(int x);
  struct ibv_context *createContext(int efa_idx);
  struct ibv_qp *createQueuePair(int qp_idx);
  int initQueuePair(struct ibv_qp *qp);
};

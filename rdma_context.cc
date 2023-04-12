#include "rdma_context.h"

int RDMAContext::init(int efa_idx, int num_qp) {
  context = createContext(efa_idx);

  // Query EFA properties
  memset(&efadvAttr, 0, sizeof(efadvAttr));
  int err = efadv_query_device(context, &efadvAttr, sizeof(efadvAttr));
  std::string error_msg = "Failed to query EFA device.";
  EFA_ASSERT(err == 0, error_msg);

  // max work requests in queue
  rxBufferDepth = efadvAttr.max_rq_wr;
  txBufferDepth = efadvAttr.max_sq_wr;

  // protection domain
  protectionDomain = ibv_alloc_pd(context);
  error_msg = "Failed to create protection domain.";
  EFA_ASSERT(protectionDomain != nullptr, error_msg);

  // send completion queues
  sendCQ = new ibv_cq*[num_qp];
  for (int i = 0; i < num_qp; i++) {
    sendCQ[i] = ibv_create_cq(context, txBufferDepth, NULL, NULL, 0);
    error_msg = "Failed to create completion queue.";
    EFA_ASSERT(sendCQ[i] != nullptr, error_msg);
  }

  // recv completion queue
  recvCQ = new ibv_cq*[num_qp];
  for (int i=0; i<num_qp; i++) {
    recvCQ[i] = ibv_create_cq(context, rxBufferDepth, NULL, NULL, 0);
    error_msg = "Failed to create completion queue.";
    EFA_ASSERT(sendCQ[i] != nullptr, error_msg);
  }

  num_queue_pairs = num_qp;
  queuePair = (ibv_qp**) malloc(sizeof(struct ibv_qp*) * num_queue_pairs);
  queuePairEx = (ibv_qp_ex**) malloc(sizeof(struct ibv_qp_ex*) * num_queue_pairs);
  for(int qp_idx=0; qp_idx<num_queue_pairs; qp_idx++) {
    // queue pair
    queuePair[qp_idx] = createQueuePair(qp_idx);
    error_msg = "Failed to create queue pair.";
    EFA_ASSERT(queuePair[qp_idx] != nullptr, error_msg);

    // queue pair ex
    queuePairEx[qp_idx] = ibv_qp_to_qp_ex(queuePair[qp_idx]);
    error_msg = "Failed to create queue pair ex.";
    EFA_ASSERT(queuePairEx[qp_idx] != nullptr, error_msg);


    // initialize queue pair
    err = initQueuePair(queuePair[qp_idx]);
    error_msg = "Failed to initialize queue pair.";
    EFA_ASSERT(err == 0, error_msg);
  }

  return 0;
}

void RDMAContext::destroy() {
  for(int i=0; i<num_queue_pairs; i++) {
    if (queuePair[i] != nullptr)
      ibv_destroy_qp(queuePair[i]);
    if (sendCQ[i] != nullptr)
      ibv_destroy_cq(sendCQ[i]);
    if (recvCQ[i] != nullptr)
      ibv_destroy_cq(recvCQ[i]);
  }

  if (protectionDomain != nullptr)
    ibv_dealloc_pd(protectionDomain);
  if (context != nullptr)
    ibv_close_device(context);
}

int RDMAContext::alignDownToPowerOfTwo(int x) {
  int n = x;
  while (n & (n - 1))
    n = n & (n - 1);
  return n;
}

struct ibv_context *RDMAContext::createContext(int efa_idx) {
  struct ibv_context *context = nullptr;
  int numDevices;

  // list the devices
  struct ibv_device **deviceList = ibv_get_device_list(&numDevices);

  std::string error_msg = "Wrong number of EFA devices detected. Expected 4 on p4d";
  INCLUDE(error_msg, numDevices);
  EFA_ASSERT(numDevices == 4, error_msg);

  context = ibv_open_device(deviceList[efa_idx]);

  // free the device list
  ibv_free_device_list(deviceList);

  error_msg = "Unable to create context for EFA " + std::to_string(efa_idx);
  EFA_ASSERT(context != nullptr, error_msg);

  return context;
}

struct ibv_qp *RDMAContext::createQueuePair(int qp_idx) {
  struct ibv_qp_init_attr_ex attrEx;
  memset(&attrEx, 0, sizeof(struct ibv_qp_init_attr_ex));
  attrEx.send_ops_flags = IBV_QP_EX_WITH_SEND | IBV_QP_EX_WITH_RDMA_READ;
  attrEx.pd = protectionDomain;
  attrEx.comp_mask |= IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
  attrEx.send_cq = sendCQ[qp_idx];
  attrEx.recv_cq = recvCQ[qp_idx];
  attrEx.cap.max_send_wr = txBufferDepth;
  attrEx.cap.max_send_sge = 1;
  attrEx.qp_type = IBV_QPT_DRIVER;
  attrEx.cap.max_inline_data = efadvAttr.inline_buf_size;
  attrEx.cap.max_recv_wr = rxBufferDepth;
  attrEx.cap.max_recv_sge = 1;  
  attrEx.qp_context = NULL;
  attrEx.sq_sig_all = 1;

  struct efadv_qp_init_attr efaAttr = {};
  efaAttr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;

  return efadv_create_qp_ex(context, &attrEx, &efaAttr, sizeof(efaAttr));
}

int RDMAContext::initQueuePair(struct ibv_qp *qp) {
  int err;
  struct ibv_qp_attr qpAttr;

  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.port_num = 1;
  qpAttr.qkey = MAGIC_QKEY;
  err = ibv_modify_qp(qp, &qpAttr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_QKEY);
  if (err) {
    printf("init qp step 1 failed");
    return err;
  }

  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_RTR;
  err = ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE);
  if (err) {
    printf("init qp step 2 failed");
    return err;
  }

  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_RTS;
  err = ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_SQ_PSN);
  if (err) {
    printf("init qp step 3 failed");
    return err;
  }

  struct ibv_port_attr portAttr;
  memset(&portAttr, 0, sizeof(portAttr));
  err = ibv_query_port(context, 1, &portAttr);
  if (err) {
    printf("ibv_query_port failed");
    return err;
  }
  maxMsgSize = portAttr.max_msg_sz;

  return 0;
}

#pragma once

#include <infiniband/verbs.h>
#include "noreentry_pool.h"

class RecvRequestPool : public NoReentryPool<ibv_recv_wr> {
public:
  ibv_recv_wr *makeNew() {
    ibv_sge *sge = new ibv_sge();
    memset(sge, 0, sizeof(ibv_sge));
    ibv_recv_wr *wr = new ibv_recv_wr();
    memset(wr, 0, sizeof(ibv_recv_wr));
    wr->sg_list = sge;
    return wr;
  }
};
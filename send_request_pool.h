#pragma once

#include <infiniband/verbs.h>
#include "noreentry_pool.h"

class SendRequestPool : public NoReentryPool<ibv_send_wr> {
public:
  ibv_send_wr *makeNew() {
    ibv_sge *sge = new ibv_sge();
    memset(sge, 0, sizeof(ibv_sge));
    ibv_send_wr *wr = new ibv_send_wr();
    memset(wr, 0, sizeof(ibv_send_wr));
    wr->sg_list = sge;
    return wr;
  }
};

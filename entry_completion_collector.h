#include "semaphore.h"

#include <stdlib.h>
#include <iostream>
#include <thread>
#include <chrono>

// Poll the completion queue to consume a specific amount of messages and record
// the time when all of them have been processed.
template<typename T>
class EntryCompletionCollector {
public:
  EntryCompletionCollector(RdmaClient* client, int completionQueueId, int amountMessages) :
    client(client),
    completionQueueId(completionQueueId),
    amountMessages(amountMessages),
    thread(std::bind(&EntryCompletionCollector::run, this))
  {
  }

  ~EntryCompletionCollector() {
    thread.join();
  }

  std::chrono::steady_clock::time_point waitUntilDone() {
    sem.wait();
    return endTime;
  }

  void run() {
    for (int j = 0; j < amountMessages; j++)
    {
      auto msg = client->popCompletion<T>(completionQueueId);
      client->recycleMsg(msg);
    }
    endTime = std::chrono::steady_clock::now();
    sem.notify();
  }
private:
  RdmaClient* client;
  int completionQueueId;
  int amountMessages;
  std::thread thread;
  std::chrono::steady_clock::time_point endTime;
  Semaphore sem;
};
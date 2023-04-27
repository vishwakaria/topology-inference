#pragma once

#include <memory>
#include <queue>

template <class T>
class NoReentryPool {
public:
  virtual T* makeNew() = 0;

  T* get() {
    if(pool_.size() > 0) {
      T* ret = pool_.front();
      pool_.pop();
      return ret;
    } else {
      return makeNew();
    }
  }

  T* get_if_available() {
    if(pool_.size() > 0) {
      T* ret = pool_.front();
      pool_.pop();
      return ret;
    } else {
      return nullptr;
    }
  }

  void recycle(T* ptr) {
    assert(ptr!=nullptr);
    pool_.push(ptr);
  }

  void terminate() {
    T* ptr;
    while(pool_.size()>0) {
      T* ret = pool_.front();
      pool_.pop();
      delete ptr;
    }
  }

  void reserve(size_t count) {
    for(size_t i=0; i<count; i++) {
      T* ptr = makeNew();
      recycle(ptr);
    }
  }


  std::queue<T*> pool_;
};
#pragma once

#include <mutex> 
#include <condition_variable> 

class Semaphore 
{ 
private: 
    std::mutex mutex_; 
    std::condition_variable condition_; 
    unsigned long count_ = 0; 

public: 
    void notify() { 
        //std::cout << "notify" << std::endl; 
        std::lock_guard<decltype(mutex_)> lock(mutex_); 
        ++count_; 
        condition_.notify_one(); 
    } 

    void notify_multiple(int n) { 
      //std::cout << "notify_multiple" << std::endl; 
      std::lock_guard<decltype(mutex_)> lock(mutex_); 
      count_ += n; 
      //std::cout << "notify_all in notify_multiple" << std::endl; 
      condition_.notify_all(); 
    } 

    void wait() { 
        std::unique_lock<decltype(mutex_)> lock(mutex_); 
        while(!count_) { 
            condition_.wait(lock); 
            //std::cout << "woke. Count: " << count_ << std::endl; 
        } 
        --count_; 
    } 

    bool wait_for(std::chrono::milliseconds timelimit) {
        std::unique_lock<decltype(mutex_)> lock(mutex_); 
        while(!count_) { 
            if (condition_.wait_for(lock, timelimit) == std::cv_status::timeout) {
                return false;
            }
            // std::cout << "woke. Count: " << count_ << std::endl; 
        } 
        --count_; 
        return true;
    } 

    bool try_wait() { 
        std::lock_guard<decltype(mutex_)> lock(mutex_); 
        if(count_) { 
            --count_; 
            return true; 
        } 
        return false; 
    } 

    void reset() {
      std::lock_guard<decltype(mutex_)> lock(mutex_);
      count_ = 0;
    }

    int get_size(){
      return count_;
    }
}; 

// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/distributed/barrier_monitor.h"
#include <gflags/gflags.h>

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <thread>  // NOLINT

#include <ThreadPool.h>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {
bool BarrierMonitor::IncreaseBarrier(const int worker_id,
                                     const std::string &barrier) {
  working_ = true;
  release_ = false;

  if (barrier == BATCH_BARRIER_MESSAGE) {
    send_barrier_queue->Push(worker_id);
  } else if (barrier == FETCH_BARRIER_MESSAGE) {
    recv_barrier_queue->Push(worker_id);
  } else {
    PADDLE_THROW("unknown status");
  }
  return Wait();
}

void BarrierMonitor::Monitor() {
  while (!working_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    VLOG(4) << "Barrier not working currently";
  }

  while (running_) {
    VLOG(3) << "running timer: " << timer << " barrier: " << barrier_type
            << " sendQ:" << send_barrier_queue->Size()
            << " recvQ: " << recv_barrier_queue->Size();

    int timer = 0;
    while (timer < kMaxWaitMS) {
      if (IsReady()) {
        Swap();
        break;
      } else {
        timer++;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    if (timer >= kMaxWaitMS) {
      Invalid();
    }
  }
}

bool BarrierMonitor::IsReady() {
  if (barrier_type == BarrierType::kSendBarrier) {
    return static_cast<int>(send_barrier_queue->Size()) == workers_;
  } else {
    return static_cast<int>(recv_barrier_queue->Size()) == workers_;
  }
}

void BarrierMonitor::Invalid() {
  std::unique_lock<std::mutex> lck(mutex_);
  valid_ = false;
  release_ = true;
  send_barrier_queue->Clear();
  recv_barrier_queue->Clear();
  workder_cv_.notify_all();
}

void BarrierMonitor::Swap() {
  std::unique_lock<std::mutex> lck(mutex_);
  ServerWeakup(barrier_type);
  WaitServerWeakup(barrier_type);

  valid_ = true;
  release_ = true;

  if (barrier_type == BarrierType::kSendBarrier) {
    barrier_type = BarrierType::kRecvBarrier;
    send_barrier_queue->Clear();
  } else {
    barrier_type = BarrierType::kSendBarrier;
    recv_barrier_queue->Clear();
  }

  workder_cv_.notify_all();
}

void BarrierMonitor::Release() {
  std::unique_lock<std::mutex> lck(mutex_);

  valid_ = true;
  release_ = true;
  barrier_type = BarrierType::kRecvBarrier;
  send_barrier_queue->Clear();
  recv_barrier_queue->Clear();
}

bool BarrierMonitor::Wait() {
  std::unique_lock<std::mutex> lk(mutex_);
  workder_cv_.wait(lk, [this] { return (release_); });
  return valid_;
}

void BarrierMonitor::WaitServerWeakup() {
  std::unique_lock<std::mutex> lk(mutex_);
  server_cv_.wait(lk);
}
void BarrierMonitor::ServerWeakup() {
  std::unique_lock<std::mutex> lk(mutex_);
  server_cv_.notify_all();
}

std::once_flag BarrierMonitor::init_flag_;
std::unique_ptr<BarrierMonitor> BarrierMonitor::monitor_(nullptr);

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
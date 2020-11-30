/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <ThreadPool.h>
#include <stdint.h>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "paddle/fluid/distributed/communicator_common.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"

#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/service/ps_client.h"

DECLARE_bool(communicator_is_sgd_optimizer);

namespace paddle {
namespace distributed {

using Scope = framework::Scope;
using Variable = framework::Variable;

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(size_t capacity) : capacity_(capacity) {
    PADDLE_ENFORCE_GT(capacity_, 0, "The capacity must be greater than 0.");
  }

  bool Push(const T &elem) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return queue_.size() < capacity_; });
      PADDLE_ENFORCE_LT(queue_.size(), capacity_);
      queue_.push_back(elem);
    }
    cv_.notify_one();
    return true;
  }

  bool Push(T &&elem) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return queue_.size() < capacity_; });
      PADDLE_ENFORCE_LT(queue_.size(), capacity_);
      queue_.emplace_back(std::move(elem));
    }
    cv_.notify_one();
    return true;
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [=] { return !queue_.empty(); });
    T rc(std::move(queue_.front()));
    queue_.pop_front();
    cv_.notify_one();
    return rc;
  }

  size_t Cap() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  const size_t capacity_;
  std::deque<T> queue_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
inline void MergeVars(const std::string &var_name,
                      const std::vector<std::shared_ptr<Variable>> &vars,
                      Scope *scope, bool merge_add = true) {
  PADDLE_ENFORCE_NE(vars.empty(), true, platform::errors::InvalidArgument(
                                            "vector vars are empty."));
  auto cpu_place = platform::CPUPlace();
  auto &var0 = vars[0];
  auto *out_var = scope->Var(var_name);

  if (var0->IsType<framework::LoDTensor>()) {
    auto dims = var0->Get<framework::LoDTensor>().dims();
    VLOG(3) << "merge " << var_name << " LoDTensor dims " << dims
            << "; merge add: " << merge_add;
    // init output tensor
    auto *out_t = out_var->GetMutable<framework::LoDTensor>();
    out_t->mutable_data<T>(dims, cpu_place);
    // check the input dims
    for (auto &var : vars) {
      auto &var_t = var->Get<framework::LoDTensor>();
      PADDLE_ENFORCE_EQ(
          var_t.dims(), dims,
          platform::errors::InvalidArgument("vars should have the same dims"));
    }

    // set output tensor to 0.
    auto cpu_ctx = paddle::platform::CPUDeviceContext();
    paddle::operators::math::SetConstant<paddle::platform::CPUDeviceContext, T>
        constant_functor;
    constant_functor(cpu_ctx, out_t, static_cast<T>(0));
    // sum all vars to out
    auto result = EigenVector<T>::Flatten(*out_t);
    for (auto &var : vars) {
      auto &in_t = var->Get<framework::LoDTensor>();
      auto in = EigenVector<T>::Flatten(in_t);
      result.device(*cpu_ctx.eigen_device()) = result + in;
    }
    if (!merge_add) {
      result.device(*cpu_ctx.eigen_device()) =
          result / static_cast<T>(vars.size());
    }
  } else if (var0->IsType<framework::SelectedRows>()) {
    auto &slr0 = var0->Get<framework::SelectedRows>();
    auto *out_slr = out_var->GetMutable<framework::SelectedRows>();
    out_slr->mutable_rows()->clear();
    out_slr->mutable_value()->mutable_data<T>({{}}, cpu_place);
    std::vector<const paddle::framework::SelectedRows *> inputs;
    inputs.reserve(vars.size());
    for (auto &var : vars) {
      inputs.push_back(&var->Get<framework::SelectedRows>());
    }
    auto dev_ctx = paddle::platform::CPUDeviceContext();
    if (merge_add) {
      paddle::operators::math::scatter::MergeAdd<
          paddle::platform::CPUDeviceContext, T>
          merge_add;
      merge_add(dev_ctx, inputs, out_slr);
    } else {
      paddle::operators::math::scatter::MergeAverage<
          paddle::platform::CPUDeviceContext, T>
          merge_average;
      merge_average(dev_ctx, inputs, out_slr);
    }

    VLOG(3) << "merge " << var_name << " SelectedRows height: " << slr0.height()
            << " dims: " << slr0.value().dims() << "; merge add: " << merge_add;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument("unsupported var type: %s!",
                                                   var0->Type()));
  }
}

using RpcCtxMap = std::unordered_map<std::string, CommContext>;
using RecvCtxMap = std::unordered_map<uint64_t, std::vector<std::string>>;
using SparseValue = std::unordered_map<int64_t, std::vector<float>>;

class Communicator {
 public:
  Communicator();

  explicit Communicator(const std::map<std::string, std::string> &envs_) {
    VLOG(0) << "Communicator Init Envs";
    for (auto &iter : envs_) {
      envs[iter.first] = iter.second;
      VLOG(0) << iter.first << ": " << iter.second;
    }
    barrier_table_id_ = std::stoi(envs.at("barrier_table_id"));
    trainer_id_ = std::stoi(envs.at("trainer_id"));
  }

  virtual ~Communicator() {}

  virtual void Start() = 0;

  virtual void Stop() = 0;

  virtual bool IsRunning() { return running_; }

  virtual void Clean() {}

  virtual bool Check(const int table_id) = 0;
  virtual bool Check(const std::vector<std::string> &var_tables) = 0;

  virtual void Send(const std::vector<std::string> &var_names,
                    const framework::Scope &scope) = 0;

  virtual void RecvNoBarrier() {}

  virtual void Barrier() {}

  virtual void BarrierWithTable(uint32_t barrier_type) {
    auto rets = _worker_ptr->barrier(barrier_table_id_, barrier_type);
    rets.wait();
  }

  virtual void BarrierTriggerDecrement() {}

  virtual void BarrierTriggerReset(int init_counter) {}

  virtual void InitEnvs() = 0;

  virtual void InitImpl(const RpcCtxMap &send_varname_to_ctx,
                        const RecvCtxMap &recv_varname_to_ctx,
                        const std::string &dist_desc,
                        const std::vector<uint64_t> &host_sign_list,
                        Scope *recv_scope) {}

  static Communicator *GetInstance() { return communicator_.get(); }

  static std::shared_ptr<Communicator> GetInstantcePtr() {
    return communicator_;
  }

  template <typename T>
  static Communicator *InitInstance(
      const RpcCtxMap &send_ctx, const RecvCtxMap &recv_ctx,
      const std::string &dist_desc, const std::vector<uint64_t> &host_sign_list,
      Scope *recv_scope, const std::map<std::string, std::string> &envs) {
    std::call_once(init_flag_, &Communicator::InitWithRpcCtx<T>, send_ctx,
                   recv_ctx, dist_desc, host_sign_list, recv_scope,
                   std::ref(envs));
    return communicator_.get();
  }

  // Init is called by InitInstance.
  template <typename T>
  static void InitWithRpcCtx(const RpcCtxMap &send_ctx,
                             const RecvCtxMap &recv_ctx,
                             const std::string &dist_desc,
                             const std::vector<uint64_t> &host_sign_list,
                             Scope *recv_scope,
                             const std::map<std::string, std::string> &envs) {
    if (communicator_.get() == nullptr) {
      communicator_.reset(new T(std::ref(envs)));
      communicator_->InitEnvs();
      communicator_->InitImpl(send_ctx, recv_ctx, dist_desc, host_sign_list,
                              recv_scope);
    }
  }

  PSClient *GetPsClient() { return _worker_ptr.get(); }

  std::shared_ptr<paddle::distributed::PSClient> GetPsClientPtr() {
    return _worker_ptr;
  }

  std::shared_ptr<PSClient> _worker_ptr;  // pointer to worker

 protected:
  bool running_ = false;
  bool waiting_ = true;
  bool flushing_ = false;
  static std::shared_ptr<Communicator> communicator_;
  static std::once_flag init_flag_;

  int barrier_table_id_ = 0;
  int trainer_id_ = 0;
  std::unordered_map<std::string, std::string> envs;

  //计算每个shard 对 dense的存储量
  inline uint32_t dense_dim_per_shard(uint32_t dense_dim_total,
                                      uint32_t shard_num) {
    return dense_dim_total / shard_num + 1;
  }

  void init_gflag(const std::string &gflags);
  paddle::distributed::PSParameter _ps_param;
  paddle::distributed::PaddlePSEnvironment _ps_env;
};

class AsyncCommunicator : public Communicator {
 public:
  AsyncCommunicator() : Communicator() {}

  explicit AsyncCommunicator(const std::map<std::string, std::string> &envs)
      : Communicator(envs) {}

  ~AsyncCommunicator();

  void InitEnvs() {
    VLOG(0) << "AsyncCommunicator InitEnvs Begin";
    min_send_grad_num_before_recv_ =
        std::stoi(envs.at("communicator_min_send_grad_num_before_recv"));
    thread_pool_size_ = std::stoi(envs.at("communicator_thread_pool_size"));
    max_merge_var_num_ = std::stoi(envs.at("communicator_max_merge_var_num"));
    send_wait_times_ = std::stoi(envs.at("communicator_send_wait_times"));
    send_queue_size_ = std::stoi(envs.at("communicator_send_queue_size"));
    need_global_step_ =
        static_cast<bool>(std::stoi(envs.at("need_global_step")));

    // trainers_ = std::stoi(envs.at("trainers"));
    pserver_push_sparse_merge_limit_ =
        std::stoi(envs.at("pserver_push_sparse_merge_limit"));
    pserver_push_dense_merge_limit_ =
        std::stoi(envs.at("pserver_push_dense_merge_limit"));
    pserver_pull_dense_limit_ = std::stoi(envs.at("pserver_pull_dense_limit"));
    pserver_async_push_sparse_interval_ms_ =
        std::stoi(envs.at("pserver_async_push_sparse_interval_ms"));
    pserver_async_push_dense_interval_ms_ =
        std::stoi(envs.at("pserver_async_push_dense_interval_ms"));
    pserver_communicate_compress_type_ =
        std::stoi(envs.at("pserver_communicate_compress_type"));
    pserver_scale_gradient_by_merge_ =
        std::stoi(envs.at("pserver_scale_gradient_by_merge"));
    pserver_max_async_call_num_ =
        std::stoi(envs.at("pserver_max_async_call_num"));
    pserver_timeout_ms_ = std::stoi(envs.at("pserver_timeout_ms"));
    pserver_connect_timeout_ms_ =
        std::stoi(envs.at("pserver_connect_timeout_ms"));
    pserver_sparse_merge_thread_ =
        std::stoi(envs.at("pserver_sparse_merge_thread"));
    pserver_sparse_table_shard_num_ =
        std::stoi(envs.at("pserver_sparse_table_shard_num"));
    VLOG(0) << "AsyncCommunicator InitEnvs End";
  }

  void Start() override;

  void Stop() override;

  void InitImpl(const RpcCtxMap &send_varname_to_ctx,
                const RecvCtxMap &recv_varname_to_ctx,
                const std::string &dist_desc,
                const std::vector<uint64_t> &host_sign_list,
                Scope *recv_scope) override;

  void InitParams();

  virtual void MainThread();

  virtual void SendSparse(const std::string &var_name, int table_id);
  virtual void SendDense(const std::vector<std::string> &var_names,
                         int table_id);

  virtual bool Check(const int table_id);
  virtual bool Check(const std::vector<std::string> &var_tables);

  void Send(const std::vector<std::string> &var_names,
            const framework::Scope &scope) override;

  virtual void SendByCommunicator();

  virtual void SendGlobalStep(int batches) {}

  virtual void RecvByCommunicator();

  virtual void RecvNoBarrier();

  virtual int BatchesCounter() { return 1; }

  virtual void BarrierSend() {}

  virtual void BarrierRecv() {}

  virtual void BarrierWeakUp() {}

 protected:
  std::map<uint64_t, std::vector<paddle::distributed::Region>>
      _dense_pull_regions;
  std::unordered_map<std::string,
                     std::shared_ptr<BlockingQueue<std::shared_ptr<Variable>>>>
      send_varname_to_queue_;

  int min_send_grad_num_before_recv_;
  int thread_pool_size_;
  int max_merge_var_num_;
  int send_wait_times_;
  int send_queue_size_;
  bool need_global_step_ = false;
  int trainers_;

  int pserver_push_sparse_merge_limit_;
  int pserver_push_dense_merge_limit_;
  int pserver_sparse_merge_thread_;
  int pserver_async_push_sparse_interval_ms_;
  int pserver_async_push_dense_interval_ms_;
  int pserver_pull_dense_limit_;
  int pserver_scale_gradient_by_merge_;
  int pserver_max_async_call_num_;
  int pserver_communicate_compress_type_;
  int pserver_timeout_ms_;
  int pserver_connect_timeout_ms_;
  int pserver_sparse_table_shard_num_;

  RpcCtxMap send_varname_to_ctx_;
  RecvCtxMap recv_varname_to_ctx_;

  std::unique_ptr<std::thread> main_thread_{nullptr};
  std::unique_ptr<std::thread> recv_thread_{nullptr};

  Scope *recv_scope_;                  // should be global scope
  std::unique_ptr<Scope> send_scope_;  // an independent scope
  int server_nums;

  std::atomic<uint32_t> _async_call_num{0};
};

class HalfAsyncCommunicator : public AsyncCommunicator {
 public:
  HalfAsyncCommunicator() {}

  explicit HalfAsyncCommunicator(const std::map<std::string, std::string> &envs)
      : AsyncCommunicator(envs) {}

  void InitEnvs() {
    min_send_grad_num_before_recv_ = 0;

    max_merge_var_num_ = std::stoi(envs.at("communicator_max_merge_var_num"));
    send_wait_times_ = std::stoi(envs.at("communicator_send_wait_times"));
    thread_pool_size_ = std::stoi(envs.at("communicator_thread_pool_size"));
    send_queue_size_ = std::stoi(envs.at("communicator_send_queue_size"));
    need_global_step_ =
        static_cast<bool>(std::stoi(envs.at("need_global_step")));
    VLOG(0) << "HalfAsyncCommunicator Initialized";
  }

  void SendByCommunicator() override;

  void Clean() override;

  void Barrier() override;

  void BarrierTriggerDecrement() override;

  void BarrierTriggerReset(int initial_val) override;

  int BatchesCounter();

  void BarrierWeakUp();

 protected:
  // mutex for Wait for barrier
  std::mutex barrier_mutex_;
  std::condition_variable barrier_cond_;
  std::atomic<int64_t> barrier_trigger_{0};
  std::atomic<int64_t> barrier_counter_{0};
};

class SyncCommunicator : public HalfAsyncCommunicator {
 public:
  SyncCommunicator() : HalfAsyncCommunicator() {}

  explicit SyncCommunicator(const std::map<std::string, std::string> &envs)
      : HalfAsyncCommunicator(envs) {}

  void InitEnvs() {
    min_send_grad_num_before_recv_ = 0;

    max_merge_var_num_ = std::stoi(envs.at("communicator_max_merge_var_num"));
    send_wait_times_ = std::stoi(envs.at("communicator_send_wait_times"));
    thread_pool_size_ = std::stoi(envs.at("communicator_thread_pool_size"));
    send_queue_size_ = std::stoi(envs.at("communicator_send_queue_size"));
    need_global_step_ =
        static_cast<bool>(std::stoi(envs.at("need_global_step")));

    auto pserver_strings = envs.at("pserver_endpoints");
    pserver_endpoints_ = paddle::string::Split(pserver_strings, ',');
    VLOG(0) << "SyncCommunicator Initialized";
  }

  void BarrierSend();

  void BarrierRecv();

 private:
  std::vector<std::string> pserver_endpoints_{};
};

}  // namespace distributed
}  // namespace paddle

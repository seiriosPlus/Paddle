// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <ThreadPool.h>
#include <assert.h>
#include <pthread.h>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include "Eigen/Dense"
#include "paddle/fluid/distributed/table/accessor.h"
#include "paddle/fluid/distributed/table/common_table.h"
#include "paddle/fluid/distributed/table/depends/initializers.h"
#include "paddle/fluid/distributed/table/depends/large_scale_kv.h"
#include "paddle/fluid/distributed/table/depends/sparse.h"
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

class CommonSparseTable : public SparseTable {
 public:
  explicit CommonSparseTable() { rwlock_.reset(new framework::RWLock); }
  virtual ~CommonSparseTable() {}

  // unused method begin
  virtual int32_t pull_dense(float* pull_values, size_t num) { return 0; };
  virtual int32_t push_dense_param(const float* values, size_t num) {
    return 0;
  };
  virtual int32_t push_dense(const float* values, size_t num) { return 0; };
  // unused method end

  virtual int32_t initialize() override;
  virtual int32_t initialize_shard() override { return 0; }
  virtual void create_initializer(const std::string& attr,
                                  const std::string& name);
  virtual int32_t initialize_value();
  virtual int32_t initialize_optimizer();

  int32_t load(const std::string& path, const std::string& param) override;

  int32_t save(const std::string& path, const std::string& param) override;

  virtual std::pair<int64_t, int64_t> print_table_stat() override;
  virtual int32_t pull_sparse(float* pull_values, const uint64_t* keys,
                              size_t num) override;

  virtual int32_t push_sparse(const uint64_t* keys, const float* values,
                              size_t num) override;

  // only for sparse geo table
  virtual int32_t push_sparse_param(const uint64_t* keys, const float* values,
                                    size_t num) override;

  virtual int32_t pour() override;
  virtual int32_t flush() override;
  virtual int32_t shrink() override;
  virtual void clear() override;

 protected:
  virtual int32_t _push_sparse(const uint64_t* keys, const float* values,
                               size_t num);

 private:
  const int task_pool_size_ = 10;
  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;

  bool sync = false;
  int param_dim_ = 0;
  std::shared_ptr<SparseOptimizer> optimizer_;
  std::unordered_map<std::string, Initializer*> initializers_;
  std::vector<std::shared_ptr<ValueBlock>> shard_values_;
  std::unordered_map<uint64_t, ReservoirValue<float>> pull_reservoir_;
  std::unique_ptr<framework::RWLock> rwlock_{nullptr};
};

}  // namespace distributed
}  // namespace paddle
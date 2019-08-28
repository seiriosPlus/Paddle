//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {
namespace distributed {

using LoDTensor = framework::LoDTensor;
constexpr int64_t kNoPadding = -1;

void prefetchs(const std::vector<const LoDTensor&>& id_tensors,
               std::vector<LoDTensor*>* out_tensors,
               LoDTensor* persistable_tensor, const bool backfill,
               const std::vector<std::string>& table_names,
               const std::vector<std::string>& endpoints,
               const std::vector<int64_t>& height_sections,
               const framework::ExecutionContext& context,
               const framework::Scope& scope);

void prefetch(const LoDTensor& id_tensor, LoDTensor* out_tensor,
              LoDTensor* persistable_tensor, const bool backfill,
              const std::vector<std::string>& table_names,
              const std::vector<std::string>& endpoints,
              const std::vector<int64_t>& height_sections,
              const framework::ExecutionContext& context,
              const framework::Scope& scope);

};  // namespace distributed
};  // namespace operators
};  // namespace paddle

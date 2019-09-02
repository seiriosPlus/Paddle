/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"

namespace paddle {
namespace operators {

inline bool NeedSend(const framework::Scope& scope,
                     const std::string& varname) {
  // dummy variable is only used in parallel executor to represent
  // some dependency relationship, we don't need to send/recv it.
  // TODO(paddle-dev): Why would parallel executor logic leaked into here?
  if (varname.find(framework::ir::Node::kControlDepVarName) !=
      std::string::npos)
    return false;
  auto* var = scope.FindVar(varname);
  PADDLE_ENFORCE_NOT_NULL(var, "Can not find variable '%s' in the send side.",
                          varname);
  if (var->IsType<framework::LoDTensor>()) {
    return var->Get<framework::LoDTensor>().IsInitialized();
  } else if (var->IsType<framework::SelectedRows>()) {
    return var->Get<framework::SelectedRows>().rows().size() > 0UL;
  } else {
    PADDLE_THROW(
        "Variable type in send side should be in "
        "[LodTensor, SelectedRows]");
  }
  return false;
}

inline std::vector<int64_t> ToAbsoluteSection(
    const std::vector<int64_t>& height_sections) {
  std::vector<int64_t> abs_sections;
  abs_sections.resize(height_sections.size());
  abs_sections[0] = 0;
  for (size_t i = 1; i < height_sections.size(); ++i) {
    abs_sections[i] = height_sections[i - 1] + abs_sections[i - 1];
  }
  return abs_sections;
}

inline size_t GetSectionIndex(int64_t id,
                              const std::vector<int64_t>& abs_sections) {
  for (size_t i = 1; i < abs_sections.size(); ++i) {
    if (id < abs_sections[i]) {
      return i - 1;
    }
  }
  return abs_sections.size() - 1;
}

template <typename T>
inline void debug_tensor(const framework::Scope& scope,
                         const std::string& var_name) {
  auto* var = scope.FindVar(var_name);

  if (var->IsType<framework::LoDTensor>()) {
    return;
  } else if (var->IsType<framework::SelectedRows>()) {
    auto& slr = var->Get<framework::SelectedRows>();

    std::vector<int64_t> cpu_rows(slr.rows().begin(), slr.rows().end());
    auto row_n = slr.value().numel() / slr.rows().size();

    std::stringstream ss;
    ss << "\n" << var_name << ":\n";

    for (auto& cpu_row : cpu_rows) {
      ss << cpu_row << " ";
      std::stringstream ss_t;
      for (int x = 0; x < row_n; x++) {
        ss_t << slr.value().data<T>()[cpu_row * row_n + x] << " ";
      }
      ss << ss_t.str() << "\n";
    }
    ss << "\n";
    VLOG(1) << ss.str();
  } else {
    PADDLE_THROW("unsupported var type to send!");
  }
}

}  // namespace operators
}  // namespace paddle

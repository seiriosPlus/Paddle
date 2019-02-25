/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/benchmark/op_tester_config.h"

namespace paddle {
namespace operators {
namespace benchmark {

class OpTester {
 public:
  OpTester() {}

  void Init(const std::string &filename);
  void Init(const OpTesterConfig &config);

  void Run();

  std::string DebugString();

 private:
  std::vector<std::string> GetOpProtoInputNames();
  std::vector<std::string> GetOpProtoOutputNames();

  void CreateInputVarDesc();
  void CreateOutputVarDesc();

  framework::VarDesc *Var(const std::string &name);
  void CreateVariables(framework::Scope *scope);

  template <typename T>
  void SetupTensor(framework::LoDTensor *input,
                   const std::vector<int64_t> &shape, T lower, T upper);

  void RunImpl();

 private:
  OpTesterConfig config_;
  std::string type_;
  framework::OpDesc op_desc_;
  std::unordered_map<std::string, std::unique_ptr<framework::VarDesc>> vars_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::unique_ptr<framework::OperatorBase> op_;
  platform::Place place_;
  std::unique_ptr<framework::Scope> scope_;
};

}  // namespace benchmark
}  // namespace operators
}  // namespace paddle

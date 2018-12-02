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

#include "paddle/fluid/imperative/layer.h"
#include <deque>
#include <limits>
#include <map>
#include <random>
#include <utility>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace imperative {

using framework::Variable;

void AddTo(Variable* src, Variable* dst) {
  framework::LoDTensor* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  framework::LoDTensor* src_tensor = src->GetMutable<framework::LoDTensor>();
  PADDLE_ENFORCE(dst_tensor->numel() == src_tensor->numel(), "%lld vs %lld",
                 dst_tensor->numel(), src_tensor->numel());
  float* dst_data = dst_tensor->mutable_data<float>(platform::CPUPlace());
  const float* src_data = src_tensor->data<float>();
  for (size_t i = 0; i < src_tensor->numel(); ++i) {
    dst_data[i] += src_data[i];
  }
}

class Autograd {
 public:
  explicit Autograd(framework::Scope* scope) : scope_(scope) {}

  void RunBackward(VarBase* var, framework::Variable* grad) {
    if (!var->pre_op_) {
      var->ApplyGrad(scope_, grad);
      return;
    }
    PADDLE_ENFORCE(var->pre_op_->op_desc_);
    // TODO(panyx0718): Only create vars that "require_grad"
    std::vector<Variable*> op_grads =
        CreateOpGrads(var->pre_op_->output_vars_->size());
    op_grads[var->pre_op_out_idx_] = grad;

    std::deque<std::pair<OpBase*, std::vector<Variable*>>> ready;
    ready.push_back(std::make_pair(var->pre_op_, op_grads));

    std::map<OpBase*, int> dep_counts = ComputeDepCounts(var->pre_op_);
    std::map<OpBase*, std::vector<Variable*>> visited;

    while (!ready.empty()) {
      OpBase* ready_op = ready.front().first;
      std::vector<Variable*> ready_op_grads = ready.front().second;
      ready.pop_front();
      std::vector<Variable*> input_grads = ready_op->ApplyGrad(scope_);

      for (size_t i = 0; i < input_grads.size(); ++i) {
        if (!input_grads[i]) continue;
        OpBase* pre_op = ready_op->pre_ops_->at(i);
        if (!pre_op) continue;
        int pre_op_out_idx = ready_op->pre_ops_out_idx_->at(i);

        dep_counts[pre_op] -= 1;
        PADDLE_ENFORCE(dep_counts[pre_op] >= 0);
        bool pre_op_ready = dep_counts[pre_op] == 0;

        if (pre_op_ready) {
          if (visited.find(pre_op) == visited.end()) {
            PADDLE_ENFORCE(pre_op->output_vars_->size() == 1);
            visited[pre_op] = {input_grads[i]};
          } else {
            std::vector<Variable*>& pre_op_grads = visited[pre_op];
            AccumGrads(pre_op_out_idx, input_grads[i], &pre_op_grads);
          }
          ready.push_back(std::make_pair(pre_op, visited[pre_op]));
        } else {
          if (visited.find(pre_op) == visited.end()) {
            // TODO(panyx0718): Only create vars that "require_grad"
            visited[pre_op] = CreateOpGrads(var->pre_op_->output_vars_->size());
          } else {
          }
          std::vector<Variable*>& pre_op_grads = visited[pre_op];
          AccumGrads(pre_op_out_idx, input_grads[i], &pre_op_grads);
        }
      }
    }
  }

 private:
  void AccumGrads(int grad_idx, Variable* grad,
                  std::vector<Variable*>* op_grads) {
    if (!(*op_grads)[grad_idx]) {
      // FIXME(panyx0718): This should be a deep copy.
      (*op_grads)[grad_idx] = grad;
      return;
    }
    AddTo(grad, (*op_grads)[grad_idx]);
  }

  std::map<OpBase*, int> ComputeDepCounts(OpBase* op) {
    std::map<OpBase*, int> ret;

    std::deque<OpBase*> queue;
    queue.push_back(op);
    std::unordered_set<OpBase*> visited;
    visited.insert(op);
    while (!queue.empty()) {
      OpBase* candidate = queue.front();
      queue.pop_front();
      for (OpBase* pre_op : *(candidate->pre_ops_)) {
        if (!pre_op) continue;
        if (visited.find(pre_op) == visited.end()) {
          visited.insert(pre_op);
          queue.push_back(pre_op);
        }
        ret[pre_op] += 1;
      }
    }

    return ret;
  }

  std::vector<Variable*> CreateOpGrads(size_t count) {
    std::vector<Variable*> op_grads;
    for (size_t i = 0; i < count; ++i) {
      op_grads.push_back(nullptr);
    }
    return op_grads;
  }

  framework::Scope* scope_;
};

framework::Variable* CreateVariable(const std::string& name,
                                    const framework::DDim& dim, float val,
                                    framework::Scope* scope,
                                    bool random_name = true) {
  std::string varname = name;
  if (random_name) {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(
        1, std::numeric_limits<int>::max());
    int id = dist6(rng);
    varname = string::Sprintf("%s@%d", varname, id);
  }

  LOG(ERROR) << "creating var " << varname;
  framework::Variable* var = scope->Var(varname);
  framework::LoDTensor* tensor = var->GetMutable<framework::LoDTensor>();

  float* data = tensor->mutable_data<float>(dim, platform::CPUPlace());
  std::fill(data, data + tensor->numel(), val);
  return var;
}

framework::LoDTensor& VarBase::Grad() {
  VLOG(3) << "get var grad " << var_desc_->Name();
  return *grads_->GetMutable<framework::LoDTensor>();
}

void VarBase::ApplyGrad(framework::Scope* scope, Variable* grad) {
  VLOG(3) << "apply var grad " << var_desc_->Name() << " "
          << grad->Get<framework::LoDTensor>().data<float>()[0];
  if (!grads_) {
    grads_ =
        CreateVariable(string::Sprintf("%s@IGrad", var_desc_->Name()),
                       var_->Get<framework::LoDTensor>().dims(), 0.0, scope);
  }
  AddTo(grad, grads_);
  VLOG(3) << "grad_ after apply var grad " << var_desc_->Name() << " "
          << grads_->Get<framework::LoDTensor>().data<float>()[0];
}

std::vector<Variable*> OpBase::ApplyGrad(framework::Scope* scope) {
  VLOG(3) << "op grad " << grad_op_desc_->Type();

  for (const std::string& invar : grad_op_desc_->InputArgumentNames()) {
    block_->FindRecursiveOrCreateVar(invar);
    framework::Variable* var = scope->Var(invar);
    LOG(ERROR) << "op grad in var " << invar;
    if (!var->IsInitialized()) {
      framework::VarDesc* var_desc = block_->FindVar(invar);
      if (var_desc->GetType() == framework::proto::VarType::LOD_TENSOR) {
        LOG(ERROR) << "grad op invar init " << invar;
        var->GetMutable<framework::LoDTensor>();
      } else {
        LOG(ERROR) << "tracer doesn't support yet";
      }
    } else {
      var->GetMutable<framework::LoDTensor>()->type();
    }
  }

  std::vector<Variable*> ret;
  for (size_t i = 0; i < input_vars_->size(); ++i) {
    ret.push_back(nullptr);
  }
  for (const std::string& outvar : grad_op_desc_->OutputArgumentNames()) {
    LOG(ERROR) << "grad outvar " << outvar;
    block_->FindRecursiveOrCreateVar(outvar);
    framework::Variable* var = scope->Var(outvar);
    if (!var->IsInitialized()) {
      framework::VarDesc* var_desc = block_->FindVar(outvar);
      if (var_desc->GetType() == framework::proto::VarType::LOD_TENSOR) {
        var->GetMutable<framework::LoDTensor>();
      } else {
        LOG(ERROR) << "tracer doesn't support yet";
      }
    }
  }
  grad_op_desc_->InferShape(*block_);
  grad_op_desc_->InferVarType(block_);
  std::unique_ptr<framework::OperatorBase> opbase =
      framework::OpRegistry::CreateOp(*grad_op_desc_);

  opbase->Run(*scope, platform::CPUPlace());

  for (const std::string& outvar : grad_op_desc_->OutputArgumentNames()) {
    if (grad_to_var_->find(outvar) != grad_to_var_->end()) {
      std::string origin_var = (*grad_to_var_)[outvar];
      for (size_t i = 0; i < input_vars_->size(); ++i) {
        VarBase* origin_in_var = (*input_vars_)[i];
        if (origin_in_var->var_desc_->Name() == origin_var) {
          framework::Variable* var = scope->FindVar(outvar);
          LOG(ERROR) << "apply grad " << outvar << " with origin "
                     << origin_var;
          // TODO(panyx0718): Accumulate.
          // origin_in_var->grads_ = var;
          origin_in_var->ApplyGrad(scope, var);
          ret[i] = var;
          // TODO(panyx0718): There might be 2 var with the same name. We
          // currently assume the are the same Variable*. So it doesn't matter
          // which one is used.
          break;
        }
      }
    }
  }
  return ret;
}

void VarBase::RunBackward(framework::Scope* scope) {
  // TODO(panyx0718): Might not be 0th, need to detect.
  grads_ = CreateVariable(pre_op_->grad_op_desc_->InputArgumentNames()[0],
                          var_->Get<framework::LoDTensor>().dims(), 1.0, scope,
                          false);
  framework::Variable* grad =
      CreateVariable("init@imperative_grad",
                     var_->Get<framework::LoDTensor>().dims(), 1.0, scope);

  Autograd(scope).RunBackward(this, grad);
}

}  // namespace imperative
}  // namespace paddle

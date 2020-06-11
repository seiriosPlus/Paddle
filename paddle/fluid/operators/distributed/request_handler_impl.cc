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

#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include <iostream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"
#include "paddle/fluid/string/piece.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/split.h"

#include "paddle/fluid/operators/distributed/async_sparse_param_update_recorder.h"
#include "paddle/fluid/operators/distributed/barrier_monitor.h"
#include "paddle/fluid/operators/distributed/heart_beat_monitor.h"

namespace paddle {
namespace operators {
namespace distributed {

// define LOOKUP_TABLE_PATH for checkpoint notify to save lookup table variables
// to directory specified.
constexpr char LOOKUP_TABLE_PATH[] = "kLookupTablePath";

bool RequestSendHandler::Handle(const std::string &varname,
                                framework::Scope *scope,
                                framework::Variable *invar,
                                framework::Variable **outvar,
                                const int trainer_id,
                                const std::string &out_var_name,
                                const std::string &table_name) {
  VLOG(4) << "RequestSendHandler:" << varname;

  if (invar == nullptr) {
    LOG(FATAL) << "sync: Can not find server side var: " << varname;
    return false;
  }

  if (distributed_mode_ == DistributedMode::kSync) {
    return true;
  }

  HeartBeatMonitor::GetInstance()->Update(trainer_id, varname, RUNNING);

  std::string run_varname = varname;
  string::Piece part_piece("@PIECE");
  string::Piece var_name_piece = string::Piece(varname);

  if (string::Contains(var_name_piece, part_piece)) {
    auto varname_splits = paddle::string::Split(varname, '@');
    run_varname = varname_splits[0];
    scope->Rename(varname, run_varname);
  }

  if (distributed_mode_ == DistributedMode::kGeo &&
      AsyncSparseParamUpdateRecorder::GetInstance()->HasGrad(run_varname)) {
    auto &grad_slr =
        scope->FindVar(run_varname)->Get<framework::SelectedRows>();
    AsyncSparseParamUpdateRecorder::GetInstance()->Update(run_varname,
                                                          grad_slr.rows());
  }

  executor_->RunPreparedContext((*grad_to_prepared_ctx_)[run_varname].get(),
                                scope);

  return true;
}

bool RequestGetHandler::Handle(const std::string &varname,
                               framework::Scope *scope,
                               framework::Variable *invar,
                               framework::Variable **outvar,
                               const int trainer_id,
                               const std::string &out_var_name,
                               const std::string &table_name) {
  VLOG(3) << "RequestGetHandler:" << varname
          << " out_var_name: " << out_var_name << " trainer_id: " << trainer_id
          << " table_name: " << table_name;

  if (distributed_mode_ == DistributedMode::kSync) {
    *outvar = scope_->FindVar(varname);
  } else {
    if (enable_dc_asgd_) {
      // NOTE: the format is determined by distribute_transpiler.py
      std::string param_bak_name =
          string::Sprintf("%s.trainer_%d_bak", varname, trainer_id);
      VLOG(3) << "getting " << param_bak_name << " trainer_id " << trainer_id;
      auto var = scope_->FindVar(varname);
      auto t_orig = var->Get<framework::LoDTensor>();
      auto param_bak = scope_->Var(param_bak_name);
      auto t = param_bak->GetMutable<framework::LoDTensor>();
      t->mutable_data(dev_ctx_->GetPlace(), t_orig.type());
      VLOG(3) << "copying " << varname << " to " << param_bak_name;
      framework::TensorCopy(t_orig, dev_ctx_->GetPlace(), t);
    }

    if (distributed_mode_ == DistributedMode::kGeo &&
        AsyncSparseParamUpdateRecorder::GetInstance()->HasParam(varname) &&
        !table_name.empty()) {
      std::vector<int64_t> updated_rows;
      AsyncSparseParamUpdateRecorder::GetInstance()->GetAndClear(
          varname, trainer_id, &updated_rows);
      if (VLOG_IS_ON(3)) {
        std::ostringstream sstream;
        sstream << "[";
        for (auto &row_id : updated_rows) {
          sstream << row_id << ", ";
        }
        sstream << "]";
        VLOG(3) << "updated_rows size: " << updated_rows.size() << " "
                << sstream.str();
      }
      auto &origin_tensor =
          scope_->FindVar(varname)->Get<framework::LoDTensor>();
      auto *origin_tensor_data = origin_tensor.data<float>();
      auto &dims = origin_tensor.dims();
      *outvar = scope->Var();
      auto *out_slr = (*outvar)->GetMutable<framework::SelectedRows>();
      out_slr->set_rows(updated_rows);
      out_slr->set_height(dims[0]);
      auto out_dims = framework::make_ddim(
          {static_cast<int64_t>(updated_rows.size()), dims[1]});
      auto *data = out_slr->mutable_value()->mutable_data<float>(
          out_dims, origin_tensor.place());
      auto width = dims[1];
      for (size_t i = 0; i < updated_rows.size(); ++i) {
        PADDLE_ENFORCE_LT(updated_rows[i], dims[0],
                          platform::errors::OutOfRange(
                              "expected >= 0 and < %ld, but got %ld.", dims[0],
                              updated_rows[i]));
        memcpy(data + i * width, origin_tensor_data + updated_rows[i] * width,
               sizeof(float) * width);
      }
    } else {
      *outvar = scope_->FindVar(varname);
    }
  }
  return true;
}

bool RequestGetNoBarrierHandler::Handle(const std::string &varname,
                                        framework::Scope *scope,
                                        framework::Variable *invar,
                                        framework::Variable **outvar,
                                        const int trainer_id,
                                        const std::string &out_var_name,
                                        const std::string &table_name) {
  VLOG(4) << "RequestGetNoBarrierHandler:" << varname
          << " out_var_name: " << out_var_name;

  // get var from pserver immediately without barriers
  string::Piece without_barrier_piece(WITHOUT_BARRIER_MESSAGE);
  string::Piece var_name_piece = string::Piece(varname);

  if (string::Contains(var_name_piece, without_barrier_piece)) {
    var_name_piece = string::TrimSuffix(var_name_piece, without_barrier_piece);
    VLOG(4) << "Get var " << var_name_piece << " with "
            << WITHOUT_BARRIER_MESSAGE;
    *outvar = scope_->FindVar(var_name_piece.ToString());
    return true;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "GetNoBarrier must contain %s", WITHOUT_BARRIER_MESSAGE));
  }
  return true;
}

bool RequestPrefetchHandler::Handle(const std::string &varname,
                                    framework::Scope *scope,
                                    framework::Variable *invar,
                                    framework::Variable **outvar,
                                    const int trainer_id,
                                    const std::string &out_var_name,
                                    const std::string &table_name) {
  VLOG(4) << "RequestPrefetchHandler " << varname;

  if (table_name.empty()) {
    auto var_desc = program_->Block(0).FindVar(out_var_name);
    InitializeVariable(*outvar, var_desc->GetType());
    executor_->RunPreparedContext(
        (*prefetch_var_name_to_prepared_ctx_)[varname].get(), scope);
  } else {
    (*outvar)->GetMutable<framework::LoDTensor>();
    auto lookup_table_op =
        BuildLookupTableOp(table_name, varname, out_var_name);
    paddle::platform::CPUPlace cpu_place;
    lookup_table_op->Run(*scope, cpu_place);
  }
  return true;
}

bool RequestCheckpointHandler::Handle(const std::string &varname,
                                      framework::Scope *scope,
                                      framework::Variable *invar,
                                      framework::Variable **outvar,
                                      const int trainer_id,
                                      const std::string &out_var_name,
                                      const std::string &table_name) {
  PADDLE_ENFORCE_NE(
      checkpoint_notify_id, -1,
      platform::errors::Unavailable(
          "when checkpoint_notify_id = -1, there should be no RPC invoke."));

  // TODO(tangwei12): find out why scope will be error.
  auto *lt_var = scope_->FindVar(LOOKUP_TABLE_PATH)->GetMutable<std::string>();
  lt_var->clear();
  lt_var->append(out_var_name);
  VLOG(4) << "RequestCheckpointHandler update var kLookupTablePath to: "
          << out_var_name;
  executor_->RunPreparedContext(checkpoint_prepared_ctx_.get(), scope_);
  return true;
}

bool RequestNotifyHandler::Handle(const std::string &varname,
                                  framework::Scope *scope,
                                  framework::Variable *invar,
                                  framework::Variable **outvar,
                                  const int trainer_id,
                                  const std::string &out_var_name,
                                  const std::string &table_name) {
  VLOG(3) << "async process var: " << varname << ", trainer_id: " << trainer_id;

  string::Piece decay_piece(LEARNING_RATE_DECAY_COUNTER);
  string::Piece batch_piece(BATCH_BARRIER_MESSAGE);
  string::Piece fetch_piece(FETCH_BARRIER_MESSAGE);
  string::Piece complete_piece(COMPLETE_MESSAGE);

  string::Piece var_name_piece = string::Piece(varname);

  if (string::Contains(var_name_piece, batch_piece)) {
    return BarrierMonitor::GetInstance()->IncreaseBarrier(
        trainer_id, BATCH_BARRIER_MESSAGE);
  } else if (string::Contains(var_name_piece, fetch_piece)) {
    return BarrierMonitor::GetInstance()->IncreaseBarrier(
        trainer_id, FETCH_BARRIER_MESSAGE);
  } else if (string::Contains(var_name_piece, complete_piece)) {
    if (HeartBeatMonitor::GetInstance() != nullptr) {
      HeartBeatMonitor::GetInstance()->Update(trainer_id, "", COMPLETED);
    }
    rpc_server_->Complete();
    BarrierMonitor::GetInstance()->DecreaseWorker();
    return true;
  } else if (string::Contains(var_name_piece, decay_piece)) {
    VLOG(3) << "LearningRate Decay Counter Update";
    PADDLE_ENFORCE_NE(
        lr_decay_block_id, -1,
        platform::errors::InvalidArgument(
            "when lr_decay_block_id = -1, there should be no RPC invoke."));
    auto *origin_var = scope_->FindVar(varname);
    auto origin_var_tensor = origin_var->Get<framework::LoDTensor>();
    auto *send_var = scope->FindVar(varname);
    auto send_var_tensor = send_var->Get<framework::LoDTensor>();
    int64_t *origin_value =
        origin_var_tensor.mutable_data<int64_t>(origin_var_tensor.place());
    int64_t *send_value =
        send_var_tensor.mutable_data<int64_t>(send_var_tensor.place());
    origin_value[0] += send_value[0];
    executor_->RunPreparedContext(lr_decay_prepared_ctx_.get(), scope_);

    return true;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "unkown varname %s with RequestNotifyHandler", varname));
  }
  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

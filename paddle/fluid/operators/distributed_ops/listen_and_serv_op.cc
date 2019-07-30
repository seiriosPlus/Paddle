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

#include "paddle/fluid/operators/distributed_ops/listen_and_serv_op.h"

#include <stdio.h>  // for removing the port file
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <thread>  // NOLINT
#include <vector>

#include "gflags/gflags.h"

#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/math/math_function.h"

#include "paddle/fluid/operators/distributed/handlers/checkpoint_handler.h"
#include "paddle/fluid/operators/distributed/handlers/get_handler.h"
#include "paddle/fluid/operators/distributed/handlers/get_no_barrier_handler.h"
#include "paddle/fluid/operators/distributed/handlers/prefetch_handler.h"
#include "paddle/fluid/operators/distributed/handlers/send_handler.h"

#include "paddle/fluid/platform/profiler.h"

DEFINE_int32(rpc_send_thread_num, 12, "number of threads for rpc send");
DEFINE_int32(rpc_get_thread_num, 12, "number of threads for rpc get");
DEFINE_int32(rpc_prefetch_thread_num, 12, "number of threads for rpc prefetch");

namespace paddle {
namespace operators {

void RunServer(std::shared_ptr<distributed::RPCServer> service) {
  service->StartServer();
  VLOG(4) << "RunServer thread end";
}
static void split(const std::string &str, char sep,
                  std::vector<std::string> *pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

static void ParallelExecuteBlocks(
    const std::vector<size_t> &parallel_blkids, framework::Executor *executor,
    const std::vector<std::shared_ptr<framework::ExecutorPrepareContext>>
        &prepared,
    framework::ProgramDesc *program, framework::Scope *scope) {
  std::vector<std::future<void>> fs;
  for (size_t idx : parallel_blkids) {
    fs.push_back(framework::Async([&executor, &prepared, &scope, idx]() {
      int run_block = idx;  // thread local
      try {
        VLOG(3) << "running server block: " << run_block
                << "pointer: " << prepared[run_block].get();
        executor->RunPreparedContext(prepared[run_block].get(), scope);
      } catch (const std::exception &e) {
        LOG(FATAL) << "run sub program:" << idx << " error " << e.what();
      }
    }));
  }
  for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
}

ListenAndServOp::ListenAndServOp(const std::string &type,
                                 const framework::VariableNameMap &inputs,
                                 const framework::VariableNameMap &outputs,
                                 const framework::AttributeMap &attrs)
    : OperatorBase(type, inputs, outputs, attrs) {}

ListenAndServOp::~ListenAndServOp() { Stop(); }

void ListenAndServOp::Stop() {
  rpc_service_->ShutDown();
  server_thread_->join();
  auto file_path = string::Sprintf("/tmp/paddle.%d.port", ::getpid());
  remove(file_path.c_str());
}

void ListenAndServOp::SavePort() const {
  // NOTE: default write file to /tmp/paddle.selected_port
  rpc_service_->SavePort();
}

static int64_t GetTimestamp() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

void ListenAndServOp::RunOptimizeLoop(framework::Executor *executor,
                                      framework::ProgramDesc *program,
                                      framework::Scope *recv_scope) const {
  VLOG(4) << "RunOptimizeLoop for async and sync";
  auto grad_to_block_id_str =
      Attr<std::vector<std::string>>("grad_to_block_id");
  DoubleFindMap<std::string, int32_t> grad_to_block_id;

  auto append_block_maps = [](DoubleFindMap<std::string, int32_t> *out_map,
                              const std::string &grad_and_id) {
    std::vector<std::string> pieces;
    split(grad_and_id, ':', &pieces);
    VLOG(3) << "after split, key = " << pieces[0] << ", id=" << pieces[1];
    PADDLE_ENFORCE_EQ(pieces.size(), 2);
    PADDLE_ENFORCE_EQ(out_map->count(pieces[0]), 0);

    int block_id = std::stoi(pieces[1]);
    (*out_map)[pieces[0]] = block_id;
  };

  for (const auto &grad_and_id : grad_to_block_id_str) {
    append_block_maps(&grad_to_block_id, grad_and_id);
  }

  size_t num_blocks = program->Size();
  PADDLE_ENFORCE_GE(num_blocks, 2,
                    "server program should have at least 2 blocks");

  std::vector<int> block_list;
  for (size_t blkid = 1; blkid < num_blocks; ++blkid) {
    block_list.push_back(blkid);
  }
  auto optimize_prepared = executor->Prepare(*program, block_list);
  // execute global block if needed, block id 1 in the program is global
  // block if it's not bind to a grad var for it's update.
  if (block_list[0] == 1 &&
      grad_to_block_id.find_value(static_cast<int32_t>(1)) ==
          grad_to_block_id.end()) {
    executor->RunPreparedContext(optimize_prepared[0].get(), recv_scope);
  }
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      grad_to_prepared_ctx, param_to_prepared_ctx;
  for (size_t i = 0; i < block_list.size(); ++i) {
    auto blkid = block_list[i];
    auto it = grad_to_block_id.find_value(blkid);
    if (it != grad_to_block_id.end()) {
      grad_to_prepared_ctx[it->first] = optimize_prepared[i];
    }
  }

  send_handler_->SetGradToPreparedCtx(&grad_to_prepared_ctx);
  get_handler_->SetGradToPreparedCtx(&grad_to_prepared_ctx);
  prefetch_handler_->SetGradToPreparedCtx(&grad_to_prepared_ctx);
}

static void FillRequestCtx(distributed::RequestHandler *h,
                           framework::Scope *scope,
                           platform::DeviceContext *dev_ctx,
                           framework::Executor *executor,
                           framework::ProgramDesc *program,
                           distributed::RPCServer *rpc_server) {
  h->SetScope(scope);
  h->SetDevCtx(dev_ctx);
  h->SetExecutor(executor);
  h->SetProgram(program);
  h->SetRPCServer(rpc_server);
}

void ListenAndServOp::CacheVarsType(const std::vector<std::string> &varnames,
                                    const framework::Scope &scope) const {
  for (const auto &varname : varnames) {
    auto var = scope.FindVar(varname);
    PADDLE_ENFORCE(var != nullptr,
                   "Received var should be initialized in the received scope.");
    if (var->IsType<framework::SelectedRows>()) {
      sparse_vars_.push_back(varname);
    } else if (var->IsType<framework::LoDTensor>() ||
               var->IsType<framework::Tensor>()) {
      dense_vars_.push_back(varname);
    } else {
      PADDLE_THROW(
          "The type of received var should be in [SelectedRows, LoDTensor, "
          "Tensor].");
    }
  }
}

void ListenAndServOp::RunImpl(const framework::Scope &scope,
                              const platform::Place &dev_place) const {
  // Mark this as PS that it should decide profiling by listening from trainer.
  platform::SetProfileListener();
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(dev_place);
  framework::Scope &recv_scope = scope.NewScope();

  bool sync_mode = Attr<bool>("sync_mode");
  bool dc_asgd = Attr<bool>("dc_asgd");
  auto fan_in = Attr<int>("Fanin");
  auto inputs = Inputs("X");

  PADDLE_ENFORCE(!rpc_service_);
  std::string endpoint = Attr<std::string>("endpoint");
  int checkpoint_block_id = Attr<int>(kCheckpointBlockId);

  VLOG(4) << "sync_mode:" << sync_mode << ", num_workers:" << fan_in
          << ", endpoint:" << endpoint
          << ", checkpoint_block_id: " << checkpoint_block_id;

  rpc_service_.reset(new RPCSERVER_T(endpoint, fan_in));

  // ----------------------------------------------------------------------
  // Prepare request handlers

  if (sync_mode) {
    VLOG(1) << "continue";
  } else {
    VLOG(1) << "continue";
  }

  send_handler_.reset(new distributed::SendHandlerAsync());
  get_handler_.reset(new distributed::GetHandlerAsync());
  prefetch_handler_.reset(new distributed::PrefetchHandler());
  notify_handler_.reset(new distributed::CheckpointHandler());

  static_cast<distributed::CheckpointHandler *>(notify_handler_.get())
      ->SetId(checkpoint_block_id);

  rpc_service_->RegisterRPC(distributed::RequestType::SEND, send_handler_.get(),
                            FLAGS_rpc_send_thread_num);
  rpc_service_->RegisterRPC(distributed::RequestType::RECV, get_handler_.get(),
                            FLAGS_rpc_get_thread_num);
  rpc_service_->RegisterRPC(distributed::RequestType::PREFETCH,
                            prefetch_handler_.get(),
                            FLAGS_rpc_prefetch_thread_num);
  rpc_service_->RegisterRPC(distributed::RequestType::CHECKPOINT,
                            notify_handler_.get());

  // ----------------------------------------------------------------------
  // Set handler contexts
  auto optimize_blocks =
      Attr<std::vector<framework::BlockDesc *>>(kOptimizeBlocks);
  PADDLE_ENFORCE(optimize_blocks.size() >= 1,
                 "optimize blocks should be 1 at least on the pserver side.");
  auto *program = optimize_blocks[0]->Program();
  framework::Executor executor(dev_place);

  auto f = std::bind(FillRequestCtx, std::placeholders::_1, &recv_scope,
                     &dev_ctx, &executor, program, rpc_service_.get());

  f(send_handler_.get());
  f(get_handler_.get());
  f(prefetch_handler_.get());
  f(notify_handler_.get());

  // ----------------------------------------------------------------------
  // Prepare checkpoint block to handler
  std::shared_ptr<framework::ExecutorPrepareContext> ckpt_pre_context = nullptr;
  if (checkpoint_block_id != -1) {
    auto ctx = executor.Prepare(*program, checkpoint_block_id);
    ckpt_pre_context = std::move(ctx);
  }
  static_cast<distributed::CheckpointHandler *>(notify_handler_.get())
      ->SetCheckpointNotifyPreparedCtx(ckpt_pre_context);

  // ----------------------------------------------------------------------
  // Prepare prefetch block to handler
  std::vector<int> prefetch_block_id_list;
  std::unordered_map<int, std::string> block_id_to_prefetch_var_name;

  auto prefetch_var_name_to_block_id_str =
      Attr<std::vector<std::string>>(kPrefetchVarNameToBlockId);
  for (const auto &prefetch_var_name_and_id :
       prefetch_var_name_to_block_id_str) {
    std::vector<std::string> pieces;
    split(prefetch_var_name_and_id, ':', &pieces);
    VLOG(3) << "after split, prefetch_var = " << pieces[0]
            << ", id=" << pieces[1];
    PADDLE_ENFORCE_EQ(pieces.size(), 2);

    int block_id = std::stoi(pieces[1]);
    prefetch_block_id_list.push_back(block_id);
    block_id_to_prefetch_var_name[block_id] = pieces[0];
  }

  auto prefetch_prepared = executor.Prepare(*program, prefetch_block_id_list);

  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      prefetch_var_name_to_prepared_ctx;
  for (size_t i = 0; i < prefetch_block_id_list.size(); ++i) {
    auto block_id = prefetch_block_id_list[i];
    auto prefetch_var_name = block_id_to_prefetch_var_name[block_id];
    prefetch_var_name_to_prepared_ctx[prefetch_var_name] = prefetch_prepared[i];
  }
  static_cast<distributed::PrefetchHandler *>(prefetch_handler_.get())
      ->SetPrefetchPreparedCtx(&prefetch_var_name_to_prepared_ctx);
  // ----------------------------------------------------------------------

  // start the server listening after all member initialized.
  server_thread_.reset(new std::thread(RunServer, rpc_service_));
  VLOG(3) << "wait server thread to become ready...";
  rpc_service_->WaitServerReady();

  signal(SIGINT, SignalHandler::StopAndExit);
  signal(SIGTERM, SignalHandler::StopAndExit);

  // Cache the type of the received vars as `sparse_vars_` and `dense_vars_`
  // so that we can reset them at the end of each iteration.
  // NOTE: only used in sync update
  CacheVarsType(inputs, recv_scope);

  // Write to a file of server selected port for python use.
  SavePort();

  RunOptimizeLoop(&executor, program, &recv_scope);

  while (true) {
    if (rpc_service_->IsExit()) {
      VLOG(4) << "get exit! rpc_processor break!";
      break;
    }
    sleep(1);
  }  // while(true)
}

class ListenAndServOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Variables that server recv.").AsDuplicable();
    AddComment(R"DOC(" + "ListenAndServ operator" + "\n" + "This operator" +
" will start a RPC server which can receive variables from send_op and send" +
"back variables to recv_op.)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
    AddAttr<std::vector<std::string>>(
        "grad_to_block_id",
        "['param1@GRAD.block0:1', 'param2@GRAD.blockn:2'] "
        "a map from grad name to it's optimize block id")
        .SetDefault({});
    AddAttr<bool>("sync_mode", "if works at sync_mode or not").SetDefault(true);
    AddAttr<bool>("dc_asgd", "set to true will enable DC-ASGD training.")
        .SetDefault(false);
    AddAttr<std::vector<framework::BlockDesc *>>(
        kOptimizeBlocks, "Optimize blocks to run on server side.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>(kPrefetchVarNameToBlockId,
                                      "prefetch blocks to run on server side.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>(
        kSparseGradToParam,
        "sparse grad name to param name. like: 'emb@Grad:emb'")
        .SetDefault({});
    AddAttr<int>("Fanin", "How many clients send to this server.")
        .SetDefault(1);
    AddAttr<int>(kCheckpointBlockId,
                 "BolckID to run save checkpoint on pserer.")
        .SetDefault(-1);
  }
};

void SignalHandler::StopAndExit(int signal_num) {
  // Do not use VLOG here for the device for printing maybe already released.
  // exit will release interal allocated resoureces.
  exit(0);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(listen_and_serv, ops::ListenAndServOp,
                  ops::ListenAndServOpMaker);

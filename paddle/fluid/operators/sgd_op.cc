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

#include "paddle/fluid/operators/sgd_op.h"

namespace paddle {
namespace operators {

class SGDOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of SGDOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "Learning rate should have 1 element");
    auto param_dim = ctx->GetInputDim("Param");
    // TODO(qijun): check dimensions of Param and Grad at complie
    // and run time.
    ctx->SetOutputDim("ParamOut", param_dim);
  }
};

class SGDOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SGDOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("LearningRate", "(Tensor) Learning rate of SGD");
    AddInput("Grad", "(Tensor) Input gradient");
    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddComment(R"DOC(

SGD operator

This operator implements one step of the stochastic gradient descent algorithm.

$$param\_out = param - learning\_rate * grad$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(sgd, ops::SGDOp, ops::SGDOpMaker);
REGISTER_OP_CPU_KERNEL(sgd, ops::SGDOpKernel<float>, ops::SGDOpKernel<double>);

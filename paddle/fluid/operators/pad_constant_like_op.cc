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

#include "paddle/fluid/operators/pad_constant_like_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class PadConstantLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of PadConstantLikeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of PadConstantLikeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of PadConstantLikeOp should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(x_dim.size(), y_dim.size(),
                      "The dimention of X and Y should be the same.");

    for (int i = 0; i < x_dim.size(); ++i) {
      PADDLE_ENFORCE_GE(x_dim[i], y_dim[i]);
    }
    ctx->SetOutputDim("Out", x_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Y")->type()),
        ctx.device_context());
  }
};

class PadConstantLikeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of pad_constant_like op. "
             "The input should be a k-D tensor(k > 0 and k < 7)");
    AddInput("Y",
             "The input of pad_constant_like op. "
             "The input should be a k-D tensor(k > 0 and k < 7)");
    AddOutput("Out",
              "The output of pad_constant_like op. "
              "A tensor with the same shape as X.");
    AddAttr<float>("pad_value",
                   "(float, default 0.0) "
                   "The value to fill the padded areas.")
        .SetDefault(0.0f);
    AddComment(R"DOC(
PadConstantLikeOp Operator.

Pad input(Y) with a pad_value, the number of values padded to the edges of each
axis is specified by the difference of the shape of X and Y.
((0, shape_x_0 - shape_y_0), ... (0, shape_x_n - shape_y_n)) unique pad widths for
each axis.
The input should be a k-D tensor(k > 0 and k < 7). As an example:

case1:
    Given:
        X = [[1, 2],
             [3, 4],
             [1, 2],
             [3, 4]]],
        X.shape = (4, 2)

        Y = [[5, 6],
            [7, 8]],
        Y.shape = (2, 2)

    And
        pad_value = 0,

    Return:
        Out = [[5, 6],
               [7, 8],
               [0, 0],
               [0, 0]]
        Out.shape = (4, 2)

case2:
    Given:
        X = [[[[ 0,  1,  2],
               [ 3,  4,  5]],
              [[ 6,  7,  8],
               [ 9, 10, 11]],
              [[12, 13, 14],
               [15, 16, 17]]],
             [[[18, 19, 20],
               [21, 22, 23]],
              [[24, 25, 26],
               [27, 28, 29]],
              [[30, 31, 32],
               [33, 34, 35]]]]
        X.shape = (2, 3, 2, 3)

        Y = [[[[35, 36, 37]],
              [[38, 39, 40]],
              [[41, 42, 43]]]]
        Y.shape = (1, 3, 1, 3)

    And
        pad_value = -1,

    Return:

        Out = [[[[35, 36, 37],
                 [-1, -1, -1]],
                [[38, 39, 40],
                 [-1, -1, -1]],
                [[41, 42, 43],
                 [-1, -1, -1]]],
               [[[-1, -1, -1],
                 [-1, -1, -1]],
                [[-1, -1, -1],
                 [-1, -1, -1]],
                [[-1, -1, -1],
                 [-1, -1, -1]]]]
        Out.shape = (2, 3, 2, 3)
)DOC");
  }
};

class PadConstantLikeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto y_dim = ctx->GetInputDim("Y");
    auto dout_dim = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_EQ(dout_dim.size(), y_dim.size(),
                      "The dimention of X and Y should be the same.");

    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dim);
      ctx->ShareLoD("Y", /*->*/ y_grad_name);

      for (int i = 0; i < y_dim.size(); ++i) {
        PADDLE_ENFORCE_GE(dout_dim[i], y_dim[i]);
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Y")->type()),
        ctx.device_context());
  }
};

class PadConstantLikeOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *bind = new framework::OpDesc();
    bind->SetType("pad_constant_like_grad");
    bind->SetInput("Y", Input("Y"));
    bind->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    bind->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    bind->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(bind);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(pad_constant_like, ops::PadConstantLikeOp,
                  ops::PadConstantLikeOpMaker, ops::PadConstantLikeOpGradMaker);
REGISTER_OPERATOR(pad_constant_like_grad, ops::PadConstantLikeOpGrad);

REGISTER_OP_CPU_KERNEL(
    pad_constant_like,
    ops::PadConstantLikeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PadConstantLikeKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    pad_constant_like_grad,
    ops::PadConstantLikeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PadConstantLikeGradKernel<paddle::platform::CPUDeviceContext, double>);

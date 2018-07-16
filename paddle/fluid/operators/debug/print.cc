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

#include "paddle/fluid/operators/debug/print.h"

namespace paddle {
namespace operators {
namespace debug {

void PrintVariableLod(const framework::Scope& scope,
                      const std::string& var_name, const std::string& message,
                      const bool print_tensor_type,
                      const bool print_tensor_shape,
                      const bool print_tensor_lod, const int summarize) {
  in_var_ptr = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(in_var_ptr);
  auto& in_tensor = in_var_ptr->Get<framework::LoDTensor>();

  framework::LoDTensor printed_tensor;
  printed_tensor.set_lod(in_tensor.lod());
  printed_tensor.Resize(in_tensor.dims());

  if (platform::is_cpu_place(in_tensor.place())) {
    printed_tensor.ShareDataWith(in_tensor);
  } else {
    // copy data to cpu to print
    platform::CPUPlace place;
    framework::TensorCopy(in_tensor, place, &printed_tensor);
  }

  Formater formater;
  formater.message = message;
  if (print_tensor_name) {
    formater.name = printed_var_name;
  }
  if (print_tensor_type) {
    formater.dtype = printed_tensor.type();
  }
  if (print_tensor_shape) {
    auto& dims = printed_tensor.dims();
    formater.dims.resize(dims.size());
    for (int i = 0; i < dims.size(); ++i) formater.dims[i] = dims[i];
  }
  if (print_tensor_lod) {
    formater.lod = printed_tensor.lod();
  }
  formater.summarize = summarize;
  formater.data = reinterpret_cast<void*>(printed_tensor.data<void>());
  formater(printed_tensor.numel());
}
};  // namespace debug

}  // namespace operators
}  // namespace paddle
}  // namespace paddle

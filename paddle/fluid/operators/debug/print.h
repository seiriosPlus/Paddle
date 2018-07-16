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
#include <algorithm>
#include <ctime>

#include <string>
#include <vector>

#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace operators {
namespace debug {

#define CLOG std::cout

struct Formater {
  std::string message;
  std::string name;
  std::vector<int> dims;
  std::type_index dtype{typeid(const char)};
  framework::LoD lod;
  int summarize;
  void* data{nullptr};

  void operator()(size_t size) {
    PrintMessage();
    PrintName();
    PrintDims();
    PrintDtype();
    PrintLod();
    PrintData(size);
  }

 private:
  void PrintMessage() { CLOG << std::time(nullptr) << "\t" << message << "\t"; }
  void PrintName() {
    if (!name.empty()) {
      CLOG << "Tensor[" << name << "]" << std::endl;
    }
  }
  void PrintDims() {
    if (!dims.empty()) {
      CLOG << "\tshape: [";
      for (auto i : dims) {
        CLOG << i << ",";
      }
      CLOG << "]" << std::endl;
    }
  }
  void PrintDtype() {
    if (!framework::IsType<const char>(dtype)) {
      CLOG << "\tdtype: " << dtype.name() << std::endl;
    }
  }
  void PrintLod() {
    if (!lod.empty()) {
      CLOG << "\tLoD: [";
      for (auto level : lod) {
        CLOG << "[ ";
        for (auto i : level) {
          CLOG << i << ",";
        }
        CLOG << " ]";
      }
      CLOG << "]" << std::endl;
    }
  }

  void PrintData(size_t size) {
    PADDLE_ENFORCE_NOT_NULL(data);
    // print float
    if (framework::IsType<const float>(dtype)) {
      Display<float>(size);
    } else if (framework::IsType<const double>(dtype)) {
      Display<double>(size);
    } else if (framework::IsType<const int>(dtype)) {
      Display<int>(size);
    } else if (framework::IsType<const int64_t>(dtype)) {
      Display<int64_t>(size);
    } else if (framework::IsType<const bool>(dtype)) {
      Display<bool>(size);
    } else {
      CLOG << "\tdata: unprintable type: " << dtype.name() << std::endl;
    }
  }

  template <typename T>
  void Display(size_t size) {
    auto* d = reinterpret_cast<T*>(data);
    CLOG << "\tdata: ";
    if (summarize != -1) {
      summarize = std::min(size, (size_t)summarize);
      for (int i = 0; i < summarize; i++) {
        CLOG << d[i] << ",";
      }
    } else {
      for (size_t i = 0; i < size; i++) {
        CLOG << d[i] << ",";
      }
    }
    CLOG << std::endl;
  }
};

void PrintVariableLod(const framework::Scope& scope,
                      const std::string& var_name, const std::string& message,
                      const bool print_tensor_name,
                      const bool print_tensor_type,
                      const bool print_tensor_shape,
                      const bool print_tensor_lod, const int summarize) {
  auto* in_var_ptr = scope.FindVar(var_name);
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
    formater.name = var_name;
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

}  // namespace debug
}  // namespace operators
}  // namespace paddle

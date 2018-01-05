/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <time.h>
#include <iostream>
#include "gflags/gflags.h"
#include "paddle/inference/inference.h"

DEFINE_string(dirname, "", "Directory of the inference model.");
DEFINE_string(feed_var_names, "", "Names of feeding variables");
DEFINE_string(fetch_var_names, "", "Names of fetching variables");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_dirname.empty() || FLAGS_feed_var_names.empty() ||
      FLAGS_fetch_var_names.empty()) {
    // Example:
    //   ./example --dirname=recognize_digits_mlp.inference.model
    //             --feed_var_names="x"
    //             --fetch_var_names="fc_2.tmp_2"
    std::cout << "Usage: ./example --dirname=path/to/your/model "
                 "--feed_var_names=x --fetch_var_names=y"
              << std::endl;
    exit(1);
  }

  std::cout << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::cout << "FLAGS_feed_var_names: " << FLAGS_feed_var_names << std::endl;
  std::cout << "FLAGS_fetch_var_names: " << FLAGS_fetch_var_names << std::endl;

  std::string dirname = FLAGS_dirname;
  std::vector<std::string> feed_var_names = {FLAGS_feed_var_names};
  std::vector<std::string> fetch_var_names = {FLAGS_fetch_var_names};

  paddle::InferenceEngine* engine = new paddle::InferenceEngine();
  engine->LoadInferenceModel(dirname, feed_var_names, fetch_var_names);

  paddle::framework::LoDTensor input;
  srand(time(0));
  float* input_ptr =
      input.mutable_data<float>({1, 784}, paddle::platform::CPUPlace());
  for (int i = 0; i < 784; ++i) {
    input_ptr[i] = rand() / (static_cast<float>(RAND_MAX));
  }

  std::vector<paddle::framework::LoDTensor> feeds;
  feeds.push_back(input);
  std::vector<paddle::framework::LoDTensor> fetchs;
  engine->Execute(feeds, fetchs);

  for (size_t i = 0; i < fetchs.size(); ++i) {
    auto dims_i = fetchs[i].dims();
    std::cout << "dims_i:";
    for (int j = 0; j < dims_i.size(); ++j) {
      std::cout << " " << dims_i[j];
    }
    std::cout << std::endl;
    std::cout << "result:";
    float* output_ptr = fetchs[i].data<float>();
    for (int j = 0; j < paddle::framework::product(dims_i); ++j) {
      std::cout << " " << output_ptr[j];
    }
    std::cout << std::endl;
  }

  delete engine;
  return 0;
}

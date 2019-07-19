#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AUC Utils, Calc AUC from predicts and labels"""

from __future__ import print_function

import os
import sys

import numpy as np
from paddle.fluid import metrics


def get_auc_files(dirname):
    if not os.path.isdir(dirname):
        raise IOError("{} do not exist or not directory".format(dirname))

    aucs = []
    for f in os.listdir(dirname):
        aucs.append(os.path.join(dirname, f))
    return aucs


def calc_auc(auc_files, num_thresholds):
    auc = metrics.Auc(name="auc", curve='ROC', num_thresholds=num_thresholds)

    for f in auc_files:
        with open(f, "r") as rb:
            for l in rb.readlines():
                l = l.strip()

                if l is None:
                    continue

                vals = l.split(" ")

                p_np = np.array([[float(vals[0]), float(vals[1])]])
                l_np = np.array([int(vals[2])])

                auc.update(p_np, l_np)
    return auc.eval()


if __name__ == "__main__":
    print(
        "Calc AUC with pridict/label files, each line of the file should be 'predict[0] predict[1] label'"
    )

    num_thresholds = 2**12

    if len(sys.argv) != 2:
        raise ValueError("python -u auc.py with auc_log dirname")

    auc_dirname = sys.argv[1]

    print("AUC log dir name: {}, num_thresholds: {}".format(sys.argv[1],
                                                            num_thresholds))

    auc = calc_auc(get_auc_files(auc_dirname), num_thresholds)

    print("AUC: {}".format(auc))

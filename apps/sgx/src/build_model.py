#!/usr/bin/python3

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Creates a simple TVM modules."""

import time
import argparse
import numpy as np
import tvm
from tvm import relay
import tvm.contrib.graph_executor as executor
import tvm.testing
import sys



def timer(thunk, repeat=1, number=10, dryrun=3, min_repeat_ms=1000):
    """Helper function to time a function"""
    for i in range(dryrun):
        thunk()
    ret = []
    for _ in range(repeat):
        while True:
            beg = time.time()
            for _ in range(number):
                thunk()
            end = time.time()
            lat = (end - beg) * 1e3
            if lat >= min_repeat_ms:
                break
            number = int(max(min_repeat_ms / (lat / number) + 1, number * 1.618))
        ret.append(lat / number)
    return ret


def bert_whole():
    import inspect
    import torch
    import torch.utils.dlpack
    import os
    import transformers
    from transformers import BertModel, BertTokenizer, BertConfig
    target = "llvm -mcpu=core-avx2"
    #target = "llvm --system-lib"
    ctx = tvm.cpu()

    config = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers = 12)

    
    model = BertModel(config)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(model.graph.inputs())[1:]]
    mod, params = tvm.relay.frontend.pytorch.from_pytorch(model, shape_list, default_dtype="float32")

    tvm.relay.backend.te_compiler.get().clear()

    # with tvm.autotvm.apply_history_best(log_filename):
    with tvm.transform.PassContext(opt_level=3, required_pass=["FastMath"]):
        graph, lib, params = tvm.relay.build(mod,
                                     target=target,
                                     params=params)

    # Save model
    print('saving model...')
    import os
    from os import path as osp
    build_dir = osp.abspath(sys.argv[1])
    if not osp.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)
    lib.save(osp.join(build_dir, "model.o"))
    with open(osp.join(build_dir, "graph.json"), "w") as fo:
        fo.write(graph)
    with open(osp.join(build_dir, "params.bin"), "wb") as fo:
        fo.write(relay.save_param_dict(cparams))

def main():
    
    bert_whole()
    


if __name__ == "__main__":
    main()

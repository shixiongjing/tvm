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
import mxnet as mx
import gluonnlp as nlp
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




def resnet18():
    dshape = (1, 28, 28)
    net, params = relay.testing.mlp.get_workload(batch_size=dshape[0], dtype="float32")

    dshape = (1, 3, 224, 224)
    net, params = relay.testing.resnet.get_workload(
        layers=18, batch_size=dshape[0], image_shape=dshape[1:]
    )

    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(net, "llvm --system-lib", params=params)

    build_dir = osp.abspath(sys.argv[1])
    if not osp.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)

def main():
    print('start model loading')
    batch = 1
    seq_length = 128
    #print('PATH:'+sys.path)

    # Instantiate a BERT classifier using GluonNLP
    model_name = 'bert_12_768_12'
    dataset = 'book_corpus_wiki_en_uncased'
    mx_ctx = mx.cpu()
    bert, _ = nlp.model.get_model(
        name=model_name,
        ctx=mx_ctx,
        dataset_name=dataset,
        pretrained=False,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False)
    model = nlp.model.BERTClassifier(bert, dropout=0.1, num_classes=2)
    model.initialize(ctx=mx_ctx)
    model.hybridize(static_alloc=True)

    print('Preparing input data...')
    # Prepare input data
    dtype = "float32"
    inputs = np.random.randint(0, 2000, size=(batch, seq_length)).astype(dtype)
    token_types = np.random.uniform(size=(batch, seq_length)).astype(dtype)
    valid_length = np.asarray([seq_length] * batch).astype(dtype)

    # Convert to MXNet NDArray and run the MXNet model
    inputs_nd = mx.nd.array(inputs, ctx=mx_ctx)
    token_types_nd = mx.nd.array(token_types, ctx=mx_ctx)
    valid_length_nd = mx.nd.array(valid_length, ctx=mx_ctx)
    mx_out = model(inputs_nd, token_types_nd, valid_length_nd)
    mx_out.wait_to_read()

    # Benchmark the MXNet latency
    res = timer(lambda: model(inputs_nd, token_types_nd, valid_length_nd).wait_to_read(),
                repeat=3,
                dryrun=5,
                min_repeat_ms=1000)
    print(f"MXNet latency for batch {batch} and seq length {seq_length}: {np.mean(res):.2f} ms")


    # Optimize the BERT model using TVM

    # First, Convert the MXNet model into TVM Relay format
    shape_dict = {
        'data0': (batch, seq_length),
        'data1': (batch, seq_length),
        'data2': (batch,)
    }
    mod, params = relay.frontend.from_mxnet(model, shape_dict)

    # Compile the imported model
    # target = "llvm -mcpu=core-avx2"
    target = "llvm --system-lib"
    with relay.build_config(opt_level=3, required_pass=["FastMath"]):
        graph, lib, cparams = relay.build(mod, target, params=params)

    # Save model
    print('saving model...')
    import os
    from os import path as osp
    build_dir = osp.abspath(sys.argv[1])
    if not osp.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)

    # Create the executor and set the parameters and inputs
    ctx = tvm.cpu()
    rt = executor.create(graph, lib, ctx)
    rt.set_input(**cparams)
    rt.set_input(data0=inputs, data1=token_types, data2=valid_length)

    # Run the executor and validate the correctness
    rt.run()
    out = rt.get_output(0)
    tvm.testing.assert_allclose(out.asnumpy(), mx_out.asnumpy(), rtol=1e-3, atol=1e-3)

    # Benchmark the TVM latency
    ftimer = rt.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=1000)
    prof_res = np.array(ftimer().results) * 1000
    print(f"TVM latency for batch {batch} and seq length {seq_length}: {np.mean(prof_res):.2f} ms")

    #lib.export_library(path_lib)
    lib.save(osp.join(build_dir, "model.o"))
    with open(osp.join(build_dir, "graph.json"), "w") as fo:
        fo.write(graph)
    with open(osp.join(build_dir, "params.bin"), "wb") as fo:
        fo.write(relay.save_param_dict(cparams))
    print(temp)

if __name__ == "__main__":
    main()

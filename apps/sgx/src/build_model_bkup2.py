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

mode = 0
# 0 for original bert
# 1 for bert from traced
# 2 for bert cropped layer

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

def bert_whole_origin():
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


def bert_ly:
    import inspect
    import torch
    import torch.utils.dlpack
    import os
    import transformers
    from transformers import BertModel, BertTokenizer, BertConfig
    target = "llvm --system-lib"
    ctx = tvm.cpu()

    enc = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenizing input text
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)

    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    dummy_input = [tokens_tensor, segments_tensors]

    # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
    model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

    model.cpu()
    model.eval()
    model.float()
    for p in model.parameters():
        p.requires_grad_(False)

    class DebugWrap(torch.nn.Module):
        def __init__(self, root, target_qn):
            super().__init__()
            self.root = (root,) # Hide from PyTorch
            parent, = self.root
            target_qn = target_qn.split('.')
            self.target_basename = target_qn[-1]
            for nc in target_qn[:-1]:
                parent = getattr(parent, nc)
            self.parent = (parent,)
            target = getattr(parent, self.target_basename)
            self.wrapped = target
            setattr(parent, self.target_basename, self)
        def remove(self):
            parent, = self.parent
            setattr(parent, self.target_basename, self.wrapped)
            self.root = None
        def forward(self, *inp, **kwinp):
            assert self.root is not None
            self.DEBUG_INP = inp
            self.DEBUG_KWINP = kwinp
            out = self.wrapped(*inp, **kwinp)
            self.DEBUG_OUT = out
            return out

    try:
        debug_wrap = DebugWrap(model, "encoder.layer.0.attention.self")
        tt = tokens_tensor.cpu()
        st = segments_tensors.cpu()
        model(tt, st)
    finally:
        debug_wrap.remove() 

    inp = debug_wrap.DEBUG_INP[:2]
    traced_module = torch.jit.trace(debug_wrap.wrapped, inp)
    shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in list(traced_module.graph.inputs())[1:]]
    mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_module, shape_list, default_dtype="float32")

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
    # Uncomment to test module
    # compiled_module = tvm.contrib.graph_executor.create(graph, lib, ctx)

    # inp_tvm = [tvm.nd.array(i.numpy(), ctx) for i in inp[:2]]
    # for (n, _), i in zip(shape_list, inp_tvm):
    #     compiled_module.set_input(n, i)
    # compiled_module.set_input(**params)
    # compiled_module.run()
    # traced_module.cpu()
    # x0 = numpy.abs(compiled_module.get_output(0).asnumpy()-traced_module(*inp[:2])[0].numpy()).max()
    
def bert_whole:
    import inspect
    import torch
    import torch.utils.dlpack
    import os
    import transformers
    from transformers import BertModel, BertTokenizer, BertConfig
    target = "llvm --system-lib"
    ctx = tvm.cpu()

    enc = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenizing input text
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)

    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    dummy_input = [tokens_tensor, segments_tensors]

    # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
    model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
    traced_model.eval()
    for p in traced_model.parameters():
        p.requires_grad_(False)

    shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_module.graph.inputs())[1:]]
    mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_module, shape_list, default_dtype="float32")

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
    if mode == 0:
        bert_whole_origin()
    elif mode == 1:
        bert_whole()
    elif mode == 2:
        bert_ly()


if __name__ == "__main__":
    main()

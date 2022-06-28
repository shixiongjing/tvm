from collections import namedtuple
import inspect
import torch
import torch.utils.dlpack
import os
import transformers
import tvm
from tvm import relay
import tvm.contrib.graph_executor as executor
import tvm.testing
import time
import numpy as np
from transformers import BertForPreTraining, BertModel, BertTokenizer, BertConfig
target = "llvm -mcpu=core-avx2"
    #target = "llvm --system-lib"
ctx = tvm.cpu()

config = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers = 3, return_dict=False)


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



model = BertModel(config)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

#with torch.no_grad():
#    torch_preds = model(tokens_tensor, segments_tensors)
#traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
#traced_model.eval()
#for p in traced_model.parameters():
#    p.requires_grad_(False)
#with torch.no_grad():
#    torch_preds = model(tokens_tensor, segments_tensors)
traced_model = torch.jit.trace(model, (tokens_tensor, segments_tensors)).eval()
input_1 = 'input_ids'
input_2 = 'input.2'
shape_list = [(input_1, list(tokens_tensor.shape)), 
              (input_2, list(segments_tensors.shape))]

mod, params = relay.frontend.from_pytorch(traced_model, shape_list)
#shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in tuple(traced_model.graph.inputs())[1:]]
#mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")

tvm.relay.backend.te_compiler.get().clear()

    # with tvm.autotvm.apply_history_best(log_filename):
with tvm.transform.PassContext(opt_level=3, required_pass=["FastMath"]):
    graph, lib, params = tvm.relay.build(mod, target=target, params=params)
# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this mode
# Create the executor and set the parameters and inputs
# Prepare input data
dtype = "float32"
inputs = np.random.randint(0, 2000, size=(batch, seq_length)).astype(dtype)
token_types = np.random.uniform(size=(batch, seq_length)).astype(dtype)
valid_length = np.asarray([seq_length] * batch).astype(dtype)

ctx = tvm.cpu()
rt = executor.create(graph, lib, ctx)
rt.set_input(**params)
rt.set_input(data0=inputs, data1=token_types, data2=valid_length)

# Run the executor and validate the correctness
rt.run()
out = rt.get_output(0)
tvm.testing.assert_allclose(out.asnumpy(), mx_out.asnumpy(), rtol=1e-3, atol=1e-3)

# Benchmark the TVM latency
ftimer = rt.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=1000)
prof_res = np.array(ftimer().results) * 1000
print(f"TVM latency for batch {batch} and seq length {seq_length}: {np.mean(prof_res):.2f} ms")
# Run the executor and validate the correctness
rt.run()
out = rt.get_output(0)
tvm.testing.assert_allclose(out.asnumpy(), mx_out.asnumpy(), rtol=1e-3, atol=1e-3)

# Benchmark the TVM latency
ftimer = rt.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=1000)
prof_res = np.array(ftimer().results) * 1000
print(f"TVM latency for batch {batch} and seq length {seq_length}: {np.mean(prof_res):.2f} ms")

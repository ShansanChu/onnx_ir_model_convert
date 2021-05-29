# to fix bi-sru time-sequence error and add reverse node using onnx graphsurgeon by Nvidia
#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#

import onnx_graphsurgeon as gs
import numpy as np
import onnx

graph = gs.import_onnx(onnx.load("wlg_tf2onnx_ocr_recog.onnx"))

#new_node to add
import onnx
import numpy as np
from onnx import TensorProto
import onnxruntime as rt
#bs_loc=np.array([1],dtype=np.int64)
#timeloc=np.array([0],dtype=np.int64)
#bs=onnx.helper.make_node('Constant',inputs=[],
#        outputs=['bs'],
#        value=onnx.helper.make_tensor(
#        name='bs_loc',
#        data_type=onnx.TensorProto.INT64,
#        dims=bs_loc.shape,
#        vals=bs_loc,),)
#length=onnx.helper.make_node(
#        'Constant',
#        inputs=[],
#        outputs=['length'],
#        value=onnx.helper.make_tensor(
#        name='time_loc',
#        data_type=onnx.TensorProto.INT64,
#        dims=timeloc.shape,
#        vals=timeloc,),)
#node_shape=onnx.helper.make_node('Shape',inputs=['X'],outputs=['shape'])
#node_bs=onnx.helper.make_node('Gather',inputs=['shape','bs'],outputs=['batchsize'])
#node_len=onnx.helper.make_node('Gather',inputs=['shape','length'],outputs=['time_len'])
#node_reverseShape=onnx.helper.make_node('Expand',inputs=['time_len','batchsize'],outputs=['len'])

#node = onnx.helper.make_node('ReverseSequence',
#        inputs=['X', 'len'],
#        outputs=['Y'],
#        time_axis=0,
#        batch_axis=1,)

# 1. Remove the `b` input of the add node
first_loop = [node for node in graph.nodes if node.op == "Concat"][-1]
first_loop.inputs = [inp for inp in first_loop.inputs]
first_loop.outputs=[out for out in first_loop.outputs]
#print(first_loop)
#help(graph.nodes)
print(first_loop.inputs)
print(first_loop.outputs)
print('####',first_loop.inputs[-1])
# 2. Change the Add to a LeakyRelu
#first_add.op = "LeakyRelu"
#first_add.attrs["alpha"] = 0.02

# 3. Add an identity after the add node
shape_out = gs.Variable(name="shape_out", dtype=np.int64)
shape=gs.Node(op='Shape',inputs=[first_loop.inputs[-1]],outputs=[shape_out])
bs=gs.Constant('bs',values=np.array([1],dtype=np.int64))
length=gs.Constant('length',values=np.array([0],dtype=np.int64))
bs_out=gs.Variable(name='bs_out',dtype=np.int64,shape=(1))
time_len=gs.Variable(name='time_len',dtype=np.int64,shape=(1))
gather_1=gs.Node(op='Gather',inputs=[shape_out,bs],outputs=[bs_out])
gather_2=gs.Node(op='Gather',inputs=[shape_out,length],outputs=[time_len])
reverse_len=gs.Variable(name='reverse_len',dtype=np.int64)
expand_len=gs.Node(op='Expand',inputs=[time_len,bs_out],outputs=[reverse_len])
reverse_out=gs.Variable(name='reverse_out',dtype=np.float32)
reverse_seq=gs.Node(op='ReverseSequence',inputs=[first_loop.inputs[-1],reverse_len],outputs=[reverse_out],attrs={'time_axis':0,'batch_axis':1})
first_loop.inputs[-1]=reverse_out
#identity_out = gs.Variable("identity_out", dtype=np.float32)
#identity = gs.Node(op="Identity", inputs=first_add.outputs, outputs=[identity_out])
graph.nodes.extend([shape,bs,length,gather_1,gather_2,expand_len,reverse_seq])

# 4. Modify the graph output to be the identity output
graph.outputs = first_loop.outputs

# 5. Remove unused nodes/tensors, and topologically sort the graph
# ONNX requires nodes to be topologically sorted to be considered valid.
# Therefore, you should only need to sort the graph when you have added new nodes out-of-order.
# In this case, the identity node is already in the correct spot (it is the last node,
# and was appended to the end of the list), but to be on the safer side, we can sort anyway.
graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "modified.onnx")

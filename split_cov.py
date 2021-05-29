# Modified by Shan shan.zhu@enflame-tech.com based on the source code from onnxconvertCommon
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

import itertools
import numpy as np
import onnx
from onnx import helper
from onnx import onnx_pb as onnx_proto


def _npfloat16_to_int(np_list):
    '''
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    '''
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in np_list]


def convert_tensor_float_to_float16(tensor):
    '''
    Convert tensor float to float16.

    :param tensor: TensorProto object
    :return tensor_float16: converted TensorProto object

    Example:

    ::

        from onnxmltools.utils.float16_converter import convert_tensor_float_to_float16
        new_tensor = convert_tensor_float_to_float16(tensor)

    '''
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError('Expected input type is an ONNX TensorProto but got %s' % type(tensor))

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            int_list = _npfloat16_to_int(np.float16(tensor.float_data))
            print(np.all(np.isfinite(np.float16(tensor.float_data))))
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.fromstring(tensor.raw_data, dtype='float32')
            # convert float to float16
            float16_list = np.float16(float32_list)
            print(np.all(np.isfinite(float16_list)))
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tostring()
    return tensor


def convert_float_to_float16(model,float_list=[]):
    '''
    Convert tensor float type in the ONNX ModelProto input to tensor float16.

    :param model: ONNX ModelProto object
    :param float_list: custom variable name which keep variable in float
    :return: converted ONNX ModelProto object

    Examples:

    ::

        Example 1: Convert ONNX ModelProto object:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        new_onnx_model = convert_float_to_float16(onnx_model)

        Example 2: Convert ONNX model binary file:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        from onnxmltools.utils import load_model, save_model
        onnx_model = load_model('model.onnx')
        new_onnx_model = convert_float_to_float16(onnx_model)
        save_model(new_onnx_model, 'new_model.onnx')

    '''
    func_infer_shape = None
    if onnx.__version__ >= '1.2':
        try:
            from onnx.shape_inference import infer_shapes
            func_infer_shape = infer_shapes
        finally:
            pass

    domain_flag = 0
    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError('Expected model type is an ONNX ModelProto but got %s' % type(model))

    # create black list
    op_black_list = ['ArrayFeatureExtractor', 'Binarizer', 'CastMap', 'CategoryMapper', 'DictVectorizer',
                     'FeatureVectorizer', 'Imputer', 'LabelEncoder', 'LinearClassifier', 'LinearRegressor',
                     'Normalizer','ArgMax','Softmax','Loop','Sigmoid','Add', #accumulate sum #'Tanh',
                     'OneHotEncoder', 'SVMClassifier', 'SVMRegressor', 'Scaler', 'TreeEnsembleClassifier',
                     'TreeEnsembleRegressor', 'ZipMap', 'NonMaxSuppression', 'TopK', 'RoiAlign', 'Resize', 'Range']
    # create a queue for BFS
    queue = []
    value_info_list = []
    float32_value_info_list=dict() #to store all output and node calculation is done with float32
    float16_value_info_list=dict() # to store all output and node calculation done with float16
    #float_list=[model.graph.input[-1].name]
    print('float list',float_list)
    node_list = []
    cust_list=[]
    cust_float=['softmax/transpose'] #node name from custom to done in FP32
    all_constants_names=set()
    # type inference on input model
    if func_infer_shape is not None:
        model = func_infer_shape(model)
    queue.append(model)
    #all_bias=set()
    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph (GraphProto)
            if isinstance(q, onnx_proto.ModelProto):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute (AttributeProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.initializer:  # TensorProto type
                    all_constants_names.add(n.name)
                    #print(n)
                for n in q.node:
                    #add by Shan for conv bias split
                    #conv_add,outputName,bias_name=False,None,None
                    if n.op_type=="Conv":
                    #    print('----',n.input)
                        if len(n.input)<3:continue
                        #all_bias.add(n.input[-1])
                        float_list.append(n.input[-1])
                    #add by shan for custom user list
                    if n.name=='1d/transpose':
                        cust_list.append(n)
                    # if n is in the black list (doesn't support float16), no conversion for the node,
                    # and save the node for further processing
                    if n.op_type in op_black_list or n.name in cust_float:
                        node_list.append(n)
                        if n.op_type=='Loop':
                            float_list+=n.input
                            loop_node=n.attribute[0].g
                            print(loop_node.value_info)
                            #    print('##DEBUG',n.attribute[0].g.node)
                        float_list+=n.output
                        #include all intializer constants to float type
                        for inp in n.input:
                            if inp in all_constants_names:
                                float_list.append(inp)
                    else:
                        if n.op_type == 'Cast':
                            for attr in n.attribute:
                                if attr.name == 'to' and attr.i == 1:
                                    attr.i = 10
                                    break
                        if n.op_type=='Loop':
                            loop_node=n.attribute[0].g
                            print(loop_node.value_info)
                        for attr in n.attribute:
                            next_level.append(attr)
            # if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
            # and process node.attribute.t and node.attribute.tensors (TensorProto)
            if isinstance(q, onnx_proto.AttributeProto):
                next_level.append(q.g)
                for n in q.graphs:
                    next_level.append(n)
                q.t.CopyFrom(convert_tensor_float_to_float16(q.t))
                for n in q.tensors:
                    n = convert_tensor_float_to_float16(n)
            # if q is graph, process graph.initializer(TensorProto), input, output and value_info (ValueInfoProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.initializer:  # TensorProto type
                    #print(n.name,'##initializer')
                    if n.name in float_list or '1d/while' in n.name:continue
                    n = convert_tensor_float_to_float16(n)
                # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                for n in itertools.chain(q.input, q.output, q.value_info):
                    #print('===',n.name)
                    if n.name in float_list:
                        float32_value_info_list[n.name]=n
                        continue
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                        float16_value_info_list[n.name]=n
                        value_info_list.append(n)
        queue = next_level

    #print('@@##',loop_node.value_info)
    #print('##@@',model.graph.value_info)
    #print('@@@@',value_info_list)
    print('FLOAT LIST ARE ',float_list)
    print('FLOAT CUST LIST IS ',cust_list)
    print('####',float32_value_info_list.keys())
    print('####',float16_value_info_list.keys())
    # process the nodes in black list that doesn't support tensor(float16)
    help(model.graph.value_info)
    node_list+=cust_list
    for node in cust_list:
        if node.name==cust_list[0].name:
            for i in range(len(node.output)):
                output = node.output[i]
                print('***'*10,node.op_type)
                for value_info in value_info_list:
                    if output == value_info.name:
                        # create new value_info for current node's new output
                        if node.op_type!='Loop':
                            new_value_info = model.graph.value_info.add()
                        else:
                            new_value_info = loop_node.value_info.add()
                        value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                        del float16_value_info_list[output]
                        float32_value_info_list[output]=node
                        domain_flag = 1
                        continue
    queue.append(model)
    visit=set()
    fp_nod_info=dict()#to avoid two consective cast cast node
    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph (GraphProto)
            if isinstance(q, onnx_proto.ModelProto):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute (AttributeProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.node:
                    conv_add,outputName,bias_name=False,None,None
                    if n.op_type=="Conv":
                        print('----',n.input)
                        if len(n.input)<3:continue
                        #all_bias.add(n.input[-1])
                        bias_name=n.input[-1]
                        inputs=n.input
                        del n.input[-1]
                        for inp in n.input:                                                                                                            print('&&&&&',inp)
                        outputName=n.output[0]
                        n.output[0]=outputName[:-2]+'NoAdd:0'
                        conv_add=True
                        print('----',n.output)
                    if conv_add and outputName in float16_value_info_list:
                        nodes=[onnx.helper.make_node('Cast',inputs=n.output,outputs=['Cast_'+outputName],to=int(onnx_proto.TensorProto.FLOAT),name='Cast_'+outputName),
                                onnx.helper.make_node('Transpose',inputs=['Cast_'+outputName],outputs=['x_tr_'+outputName],perm=[0,2,3,1],name='x_tr_'+outputName),
                                onnx.helper.make_node('Add',inputs=['x_tr_'+outputName,bias_name],outputs=['Add_'+outputName],name='Add_'+outputName),
                                onnx.helper.make_node('Transpose',inputs=['Add_'+outputName],outputs=['y_tr_'+outputName],perm=[0,3,1,2],name='y_tr_'+outputName),
                                onnx.helper.make_node('Cast',inputs=['y_tr_'+outputName],outputs=[outputName],to=int(onnx_proto.TensorProto.FLOAT16),name=outputName+'_cast_y_tr')]
                        model.graph.node.extend(nodes)
                        value_info=float16_value_info_list[outputName]
                        print('====$$$',value_info.type.tensor_type.shape.dim)
                        dim_info=value_info.type.tensor_type.shape.dim
                        new_dim_info=[dim_info[0],dim_info[2],dim_info[3],dim_info[1]]
                        print('====###',type(dim_info[1]))
                        new_value_info = model.graph.value_info.add()
                        new_value_info.CopyFrom(value_info)
                        new_value_info.name = n.output[0]
                        new_value_info_1=model.graph.value_info.add()
                        new_value_info_1.CopyFrom(value_info)
                        new_value_info_1.name='Cast_'+outputName
                        new_value_info_1.type.tensor_type.elem_type = 1
                        new_value_info_2=model.graph.value_info.add()
                        new_value_info_2.CopyFrom(value_info)
                        new_value_info_2.name='Add_'+outputName
                        new_value_info_2.type.tensor_type.elem_type = 1
                        new_value_info_2.type.tensor_type.shape.dim[1].CopyFrom(new_dim_info[1])
                        new_value_info_2.type.tensor_type.shape.dim[2].CopyFrom(new_dim_info[2])
                        new_value_info_2.type.tensor_type.shape.dim[3].CopyFrom(new_dim_info[3])
                        new_value_info_3=model.graph.value_info.add()
                        new_value_info_3.CopyFrom(new_value_info_2)
                        new_value_info_3.name='x_tr_'+outputName
                        new_value_info_4=model.graph.value_info.add()
                        new_value_info_4.CopyFrom(new_value_info_1)
                        new_value_info_4.name='y_tr_'+outputName
                        fp_nod_info[outputName]=nodes[-1] #to avoid two concentive two cast 
                    elif conv_add and outputName in float32_value_info_list:
                        nodes=[onnx.helper.make_node('Cast',inputs=n.output,outputs=['Cast_'+outputName],to=int(onnx_proto.TensorProto.FLOAT)),
                                onnx.helper.make_node('Transpose',inputs=['Cast_'+outputName],outputs=['x_tr_'+outputName],perm=[0,2,3,1]),
                                onnx.helper.make_node('Add',inputs=['x_tr_'+outputName,bias_name],outputs=['Add_'+outputName]),
                                helper.make_node('Transpose',inputs=['Add_'+outputName],outputs=[outputName],perm=[0,3,1,2])]
                        model.graph.node.extend(nodes)
                        value_info=float16_value_info_list[outputName]
                        dim_info=value_info.type.tensor_type.shape.dim
                        new_dim_info=[dim_info[0],dim_info[2],dim_info[3],dim_info[1]]
                        print('====$$$',value_info.type.tensor_type.shape)
                        new_value_info = model.graph.value_info.add()
                        new_value_info.CopyFrom(value_info)
                        new_value_info.name = n.output[0]
                        new_value_info_1=model.graph.value_info.add()
                        new_value_info_1.CopyFrom(value_info)
                        new_value_info_1.name='Cast_'+outputName
                        new_value_info_1.type.tensor_type.elem_type = 1
                        new_value_info_2=model.graph.value_info.add()
                        new_value_info_2.CopyFrom(value_info)
                        new_value_info_2.name='Add_'+outputName
                        new_value_info_2.type.tensor_type.elem_type = 1
                        new_value_info_2.type.tensor_type.shape.dim[1].CopyFrom(new_dim_info[1])
                        new_value_info_2.type.tensor_type.shape.dim[2].CopyFrom(new_dim_info[2])
                        new_value_info_2.type.tensor_type.shape.dim[3].CopyFrom(new_dim_info[3])
                        new_value_info_3=model.graph.value_info.add()
                        new_value_info_3.CopyFrom(new_value_info_2)
                        new_value_info_3.name='x_tr_'+outputName
                    Dtype=0
                    if n.output[0] in float32_value_info_list:
                        Dtype=1
                    elif n.output[0] in float16_value_info_list:
                        Dtype=2
                    if Dtype==0:continue#assert(0), "output data type error not in [FP16,float]"
                    for idx in range(len(n.input)):
                        Ntype=0
                        i=n.input[idx]
                        if i in float32_value_info_list:
                            Ntype=1
                            value_info=float32_value_info_list[i]
                        elif i in float16_value_info_list:
                            Ntype=2
                            value_info=float16_value_info_list[i]
                        if Ntype==0:continue#assert(0),'input  data type error not in [FP16,float]'
                        if Ntype==Dtype:continue
                        if i in visit:
                            n.input[idx]=i+'_casted'
                            continue
                        if Ntype==2 and i in fp_nod_info:#one cast is added from FP32 to FP16,avoid duplicate cast nodes
                            to_remove=fp_nod_info[i]
                            n.input[idx]=to_remove.input[0]
                            model.graph.node.remove(to_remove)
                            model.graph.value_info.remove(value_info)
                            continue
                        if n.op_type!='Loop':
                            new_value_info = model.graph.value_info.add()
                        else:
                            new_value_info = loop_node.value_info.add()
                        new_value_info.CopyFrom(value_info)
                        new_value_info.name = i + '_casted'
                        new_value_info.type.tensor_type.elem_type = 1 if Dtype==1 else 10
                        # add Cast node (from tensor(float) to tensor(float16) after current node
                        attrs = {'name': i + 'Cast'}
                        attrs['to'] = 1 if Dtype==1 else 10
                        nodes = [helper.make_node('Cast', [i], [i+'_casted'], to=int(attrs['to']),name=attrs['name'])]#kwarg=attrs)]
                        if n.op_type=='Loop':
                            loop_node.node.extend(nodes)
                        else:
                            model.graph.node.extend(nodes)
                        # change current node's input name
                        n.input[idx] = i + '_casted'
                        domain_flag = 1
                        visit.add(i)
            if isinstance(q, onnx_proto.AttributeProto):
                next_level.append(q.g)
                for n in q.graphs:
                    next_level.append(n)
        queue = next_level


    if domain_flag:
        # Create operator set for cast node
        op_set = model.opset_import.add()
        op_set.domain = ""
        op_set.version = 13#7
    return model

# Fusion and graph optimization
# Author shan.zhu@enflame-tech.com
import onnxruntime as ort
EP_list = ['CPUExecutionProvider']
sess_option=ort.SessionOptions()
sess_option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
dir_path='/home/shan/OCR/'
model_path='ocr3_0_recog_v.onnx' #ocr3_0_recog_h.onnx ocr3_0_recog_v.onnx ocr3_3_detect.onnx
sess_option.optimized_model_filepath='opt_'+model_path
session = ort.InferenceSession(dir_path+model_path, sess_option,providers=EP_list)
import onnx
from onnx import shape_inference
#perform shape inference
onnx_model=onnx.load('opt_'+model_path)
inferred_model=shape_inference.infer_shapes(onnx_model)
onnx.save(inferred_model,'opt_'+model_path)





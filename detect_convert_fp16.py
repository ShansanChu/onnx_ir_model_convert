#Author shan.zhu@enflame-tech.com
#from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxconverter_common import split_cov as fp16
from onnxmltools.utils import load_model, save_model
import onnx
org_model='opt_ocr3_3_detect.onnx'
#org_model='/home/devdata/shan/CV/tf2onnx_ocr_recog.onnx'
onnx.checker.check_model(org_model)
print('Done org model check')
#model_path="tf2onnx_ocr_recog_to_transpose_optimized.onnx"
onnx_model = load_model(org_model)
onnx.checker.check_model(onnx_model)
print('BEFORE CONVERT')
new_onnx_model = fp16.convert_float_to_float16(onnx_model)
#onnx.checker.check_model(new_onnx_model)
save_model(new_onnx_model, 'FP16_fuse_opt_'+'ocr_3_3_detect.onnx')

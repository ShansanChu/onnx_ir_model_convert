# First version part
This is the codes used to convert fp32 ocr model as well as to modify the reverse bi-sru codes.

Note for the recognition model, Please keep sru, softmax, argmax in calculation float32, used **float16_shan.py**.

To fix bi-sru reverse part time sequence error, check the code in **modify.py** which adds reverse node to fix the error.

Onnx model is modified with onnx graphsurgeon from Nvidia.

# OCR models updates

**Update May 13,2020 for OCR update which seperate bias add from conv calculation**

1. Test fp32 onnx model accuracy:
    - [x] detection model accuracy test with detect_pb_shan.py
    - [x] recog model accuracy test with recog_fp32_shan.py
2. conv bn fusion and graph optimization:
    - [x] with opt_onnx.py (need to change model path)
3. Convert fp32 onnx model to fp16 mixed precision one:
    - [x] put the split_cov.py to onnxconvert_common directory
    - [x] for detection model, use detect_convert_fp16.py
    - [x] for recognition model, use convert_fp16.py
4. mixed-precison fp16 model and fp32 model accuracy compare
    - [x] for detection model fp16 mixed precision onnx and fp2 onnx model compare,batch_detect_onnx_shan.py
    - [x] for recog, recog_fp32_fp16_shan.py

For details of onnx op ir refer to - https://github.com/onnx/onnx/blob/master/docs/Operators.md
import numpy as np
import onnxruntime as rt
import os
import mymodule

import torch


options = rt.SessionOptions()

print("Current process id is ", os.getpid())
print("Context info before running python script: context_info_1: ", mymodule.context_info_1, ", context_info_2: ", mymodule.context_info_2)
import pdb; pdb.set_trace()
print("creating inferencing session")
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cuda_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]
mymodule.context_info_1=["ccc", "ddd"]
mymodule.context_info_2=["333", "444"]

sess = rt.InferenceSession("sample_interop.onnx")
print("done inferencing session, process id is:", os.getpid())
input_name = sess.get_inputs()[0].name
x = np.array([1.0, 2.0, 3.0, 4.0], np.float32).reshape(2,2)
print("running inferencing session")
pred_onx = sess.run(["Y"], {input_name: x})[0]
print("done inferencing session")
print(pred_onx)

########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, types, json, math, time
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
with open(f"{current_path}/../misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

########################################################################################################

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20230109-ctx4096'
MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/RWKV-4-World-0.1B-v1-20230520-ctx4096-converted-fp32.pth'
# MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'

PAD_SEQ = []

########################################################################################################

print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Tune these below (test True/False for all of them) to find the fastest setting:
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

print(f'Loading model - {MODEL_NAME}')
model = RWKV(model=MODEL_NAME, strategy='cuda fp32')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

print('Check LAMBADA...')
xsum = 0
xcnt = 0
xacc = 0
for d in todo:
    src = PAD_SEQ + pipeline.encode(d[0])
    dst = pipeline.encode(d[1])

    logits = 0
    correct = True
    out, model_state = model.forward(src+dst, None, full_output=True)

    for i in range(len(dst)):
        probs = F.softmax(out[len(src)-1+i,:], dim=-1)
        logits += math.log(probs[dst[i]])
        if torch.argmax(probs).item() != dst[i]:
            correct = False

    xcnt += 1
    xsum += logits
    xacc += 1 if correct else 0
    if xcnt % 100 == 0 or xcnt == len(todo):
        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))


# Check LAMBADA...
# 100 ppl 42.41 acc 34.0
# 200 ppl 29.33 acc 37.0
# 300 ppl 25.95 acc 39.0
# 400 ppl 27.29 acc 36.75
# 500 ppl 28.3 acc 35.4
# 600 ppl 27.04 acc 35.83
# 700 ppl 27.38 acc 34.57
# 800 ppl 26.95 acc 34.75
# 900 ppl 26.69 acc 35.33
# 1000 ppl 26.51 acc 35.9
# 1100 ppl 26.15 acc 36.18
# 1200 ppl 26.95 acc 35.25
# 1300 ppl 27.38 acc 35.0
# 1400 ppl 27.49 acc 34.86
# 1500 ppl 27.52 acc 35.2
# 1600 ppl 27.12 acc 35.06
# 1700 ppl 26.95 acc 35.41
# 1800 ppl 27.18 acc 35.28
# 1900 ppl 27.45 acc 34.63
# 2000 ppl 26.83 acc 35.25
# 2100 ppl 26.96 acc 35.1
# 2200 ppl 27.49 acc 35.0
# 2300 ppl 27.6 acc 34.96
# 2400 ppl 27.68 acc 34.62
# 2500 ppl 27.15 acc 34.96
# 2600 ppl 26.87 acc 35.0
# 2700 ppl 27.12 acc 35.15
# 2800 ppl 27.29 acc 35.07
# 2900 ppl 27.65 acc 35.1
# 3000 ppl 27.69 acc 35.23
# 3100 ppl 27.61 acc 35.39
# 3200 ppl 27.64 acc 35.19
# 3300 ppl 27.61 acc 34.91
# 3400 ppl 27.55 acc 35.09
# 3500 ppl 27.49 acc 35.23
# 3600 ppl 27.37 acc 35.25
# 3700 ppl 27.13 acc 35.22
# 3800 ppl 27.0 acc 35.32
# 3900 ppl 26.87 acc 35.59
# 4000 ppl 26.94 acc 35.62
# 4100 ppl 26.78 acc 35.66
# 4200 ppl 26.51 acc 35.81
# 4300 ppl 26.55 acc 35.81
# 4400 ppl 26.6 acc 35.8
# 4500 ppl 26.6 acc 35.78
# 4600 ppl 26.43 acc 35.89
# 4700 ppl 26.39 acc 35.91
# 4800 ppl 26.35 acc 35.92
# 4900 ppl 26.4 acc 35.76
# 5000 ppl 26.19 acc 35.84
# 5100 ppl 26.17 acc 35.88
# 5153 ppl 26.16 acc 35.88

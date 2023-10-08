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

PAD_SEQ = []

########################################################################################################


import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("BBuf/RWKV-4-World-169M", torch_dtype=torch.float32).to(0)
tokenizer = AutoTokenizer.from_pretrained("BBuf/RWKV-4-World-169M", trust_remote_code=True)

print('Check LAMBADA...')
xsum = 0
xcnt = 0
xacc = 0

for d in todo:
    # 使用tokenizer对数据进行编码
    src = tokenizer(d[0], return_tensors="pt").to(0)["input_ids"].squeeze(0).tolist()
    dst = tokenizer(d[1], return_tensors="pt").to(0)["input_ids"].squeeze(0).tolist()

    logits = 0
    correct = True
    # 使用模型进行前向传播
    input_tensor = torch.tensor(src + dst).unsqueeze(0).to(0)
    out = model(input_tensor).logits

    out = out.squeeze(0)
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
# 100 ppl 42.4 acc 34.0
# 200 ppl 29.3 acc 37.0
# 300 ppl 25.94 acc 39.0
# 400 ppl 27.27 acc 36.75
# 500 ppl 28.28 acc 35.4
# 600 ppl 27.02 acc 35.83
# 700 ppl 27.36 acc 34.57
# 800 ppl 26.93 acc 34.75
# 900 ppl 26.68 acc 35.33
# 1000 ppl 26.49 acc 35.9
# 1100 ppl 26.13 acc 36.18
# 1200 ppl 26.94 acc 35.25
# 1300 ppl 27.37 acc 35.0
# 1400 ppl 27.48 acc 34.86
# 1500 ppl 27.51 acc 35.13
# 1600 ppl 27.11 acc 35.0
# 1700 ppl 26.94 acc 35.35
# 1800 ppl 27.16 acc 35.22
# 1900 ppl 27.44 acc 34.58
# 2000 ppl 26.82 acc 35.2
# 2100 ppl 26.95 acc 35.05
# 2200 ppl 27.48 acc 34.95
# 2300 ppl 27.59 acc 34.91
# 2400 ppl 27.67 acc 34.58
# 2500 ppl 27.14 acc 34.88
# 2600 ppl 26.85 acc 34.92
# 2700 ppl 27.11 acc 35.07
# 2800 ppl 27.28 acc 35.0
# 2900 ppl 27.64 acc 35.07
# 3000 ppl 27.67 acc 35.2
# 3100 ppl 27.59 acc 35.35
# 3200 ppl 27.63 acc 35.16
# 3300 ppl 27.59 acc 34.88
# 3400 ppl 27.53 acc 35.03
# 3500 ppl 27.48 acc 35.17
# 3600 ppl 27.36 acc 35.19
# 3700 ppl 27.12 acc 35.16
# 3800 ppl 26.98 acc 35.26
# 3900 ppl 26.85 acc 35.54
# 4000 ppl 26.92 acc 35.58
# 4100 ppl 26.76 acc 35.61
# 4200 ppl 26.5 acc 35.79
# 4300 ppl 26.53 acc 35.81
# 4400 ppl 26.59 acc 35.8
# 4500 ppl 26.58 acc 35.78
# 4600 ppl 26.41 acc 35.89
# 4700 ppl 26.37 acc 35.89
# 4800 ppl 26.34 acc 35.9
# 4900 ppl 26.39 acc 35.73
# 5000 ppl 26.17 acc 35.82
# 5100 ppl 26.15 acc 35.86
# 5153 ppl 26.14 acc 35.86

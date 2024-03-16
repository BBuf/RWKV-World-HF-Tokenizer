## RWKV World Model HuggingFace Tokenizer

### Huggingface Project With This Repo

> When uploading the converted model to Huggingface, if the bin file is too large, you need to use this command `huggingface-cli lfs-enable-largefiles .` to lift the size limit.

- [RWKV/rwkv-5-world-1b5](https://huggingface.co/RWKV/rwkv-5-world-1b5)
- [RWKV/rwkv-5-world-3b](https://huggingface.co/RWKV/rwkv-5-world-3b)
- [RWKV/rwkv-5-world-169m](https://huggingface.co/RWKV/rwkv-5-world-169m)
- [RWKV/rwkv-4-world-169m](https://huggingface.co/RWKV/rwkv-4-world-169m)
- [RWKV/rwkv-4-world-430m](https://huggingface.co/RWKV/rwkv-4-world-430m)
- [RWKV/rwkv-4-world-1b5](https://huggingface.co/RWKV/rwkv-4-world-1b5)
- [RWKV/rwkv-4-world-3b](https://huggingface.co/RWKV/rwkv-4-world-3b)
- [RWKV/rwkv-4-world-7b](https://huggingface.co/RWKV/rwkv-4-world-7b)


### Huggingface RWKV World Model Convert


Using the script `scripts/convert_batch_rwkv5_world_model_to_hf.sh` , convert the PyTorch format model from the huggingface `BlinkDL/rwkv-5-world` project to the Huggingface format. Here, we take 3B as an example.

```shell
#!/bin/bash
set -x

cd scripts
python convert_rwkv5_checkpoint_to_hf.py --repo_id BlinkDL/rwkv-5-world \
 --checkpoint_file RWKV-5-World-3B-v2-20231118-ctx16k.pth \
 --output_dir ../../rwkv5-v2-world-3b-model/ \
 --tokenizer_file ../rwkv5_model \
 --size 3B \
 --is_world_tokenizer True

cp ../rwkv5_model/vocab.txt ../../rwkv5-v2-world-3b-model/
cp ../rwkv5_model/tokenization_rwkv5.py ../../rwkv5-v2-world-3b-model/
cp ../rwkv5_model/tokenizer_config.json ../../rwkv5-v2-world-3b-model/
cp ../rwkv5_model/configuration_rwkv5.py ../../rwkv5-v2-world-3b-model/
cp ../rwkv5_model/modeling_rwkv5.py ../../rwkv5-v2-world-3b-model/
cp ../rwkv5_model/generation_config.json ../../rwkv5-v2-world-3b-model/

```

Additionally, **you need to add the following lines at the beginning of the `config.json` file in the generated folder** :

```json
"architectures": [
    "Rwkv5ForCausalLM"
  ],
  "auto_map": {
      "AutoConfig": "configuration_rwkv5.Rwkv5Config",
      "AutoModelForCausalLM": "modeling_rwkv5.Rwkv5ForCausalLM"
  },
```

### Run Huggingface RWKV World Model

The `run_hf_world_model_xxx.py` demonstrates how to load the converted model using Huggingface's `AutoModelForCausalLM` and how to perform model inference using the custom `RWKV World Model's HuggingFace Version Tokenizer` loaded through `AutoTokenizer`:

#### CPU

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n', '\n').replace('\n\n', '\n')
    input = input.strip().replace('\r\n', '\n').replace('\n\n', '\n')
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""

model = AutoModelForCausalLM.from_pretrained("BBuf/rwkv-5-world-1b5", trust_remote_code=True).to(torch.float32)
tokenizer = AutoTokenizer.from_pretrained("BBuf/rwkv-5-world-1b5", trust_remote_code=True, padding_side='left')

texts = ["请介绍北京的旅游景点", "介绍一下大熊猫", "乌兰察布"]
prompts = [generate_prompt(text) for text in texts]

inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(inputs["input_ids"], max_new_tokens=128, do_sample=True, temperature=1.0, top_p=0.3, top_k=0, )

for output in outputs:
    print(tokenizer.decode(output.tolist(), skip_special_tokens=True))
```

output:

```shell
User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: 请介绍北京的旅游景点

Assistant: 北京是中国的首都，拥有丰富的旅游资源和文化遗产。以下是一些值得一游的景点：
1. 故宫：位于北京市中心，是明清两代的皇宫，是中国最大的古代宫殿建筑群之一，内有众多珍贵的文物和艺术品。
2. 天安门广场：位于北京市中心，是中国最著名的广场之一，是中国现代化建
User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: 介绍一下大熊猫

Assistant: 大熊猫是一种生活在中国中部地区的哺乳动物，是熊科的一种。它们的外貌特征包括黑白相间的毛皮、圆圆的身体和圆形的耳朵。大熊猫的食物主要是竹子，它们会在竹子上挖洞或者爬树来寻找食物。大熊猫是濒危物种，目前只有约1,200只存活在野外。由于栖息地的破坏和人类活动的
User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: 乌兰察布

Assistant: 乌兰察布是中国内蒙古自治区的一个县级市，位于该省中部，距离首府呼和浩特约120公里。乌兰察布市是中国著名的牧业和畜牧业基地，是中国著名的畜牧业大县。该市拥有丰富的自然资源和人文资源，是一个充满活力和创新的城市。
```

#### GPU

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n', '\n').replace('\n\n', '\n')
    input = input.strip().replace('\r\n', '\n').replace('\n\n', '\n')
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""

model = AutoModelForCausalLM.from_pretrained("BBuf/rwkv-5-world-1b5", trust_remote_code=True).to(torch.float32).to(0)
tokenizer = AutoTokenizer.from_pretrained("BBuf/rwkv-5-world-1b5", trust_remote_code=True, padding_side='left')

texts = ["请介绍北京的旅游景点", "介绍一下大熊猫", "乌兰察布"]
prompts = [generate_prompt(text) for text in texts]

inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(0)
outputs = model.generate(inputs["input_ids"], max_new_tokens=128, do_sample=True, temperature=1.0, top_p=0.3, top_k=0, )

for output in outputs:
    print(tokenizer.decode(output.tolist(), skip_special_tokens=True))
```

output:

```shell
User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: 请介绍北京的旅游景点

Assistant: 北京是中国的首都，拥有丰富的旅游资源和文化遗产。以下是一些值得一游的景点：
1. 故宫：位于北京市中心，是明清两代的皇宫，是中国最大的古代宫殿建筑群之一，内有众多珍贵的文物和艺术品。
2. 天安门广场：位于北京市中心，是中国最著名的广场之一，是中国现代化建
User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: 介绍一下大熊猫

Assistant: 大熊猫是一种生活在中国中部地区的哺乳动物，是熊科的一种。它们的外貌特征包括黑白相间的毛皮、圆圆的身体和圆形的耳朵。大熊猫的食物主要是竹子，它们会在竹子上挖洞或者爬树来寻找食物。大熊猫是濒危物种，目前只有约1,200只存活在野外。由于栖息地的破坏和人类活动的
User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: 乌兰察布

Assistant: 乌兰察布是中国内蒙古自治区的一个县级市，位于该省中部，距离首府呼和浩特约120公里。乌兰察布市是中国著名的牧业和畜牧业基地，是中国著名的畜牧业大县。该市拥有丰富的自然资源和人文资源，是一个充满活力和创新的城市。
```

### Check Lambda


The `lambda_pt.py` and `lambda_hf.py` files under the `check_lambda` folder evaluate the lambda dataset using the original PyTorch model of RWKV4 World 169M and the HuggingFace model, respectively. From the logs, it can be observed that the evaluation results they obtained are essentially the same.

#### lambda_pt.py lambda evaluate log

```shell
# Check LAMBADA...
# 100 ppl 42.41 acc 34.0
# 200 ppl 29.33 acc 37.0
# 300 ppl 25.95 acc 39.0
# 400 ppl 27.29 acc 36.75
# 500 ppl 28.3 acc 35.4
# 600 ppl 27.04 acc 35.83
...
# 5000 ppl 26.19 acc 35.84
# 5100 ppl 26.17 acc 35.88
# 5153 ppl 26.16 acc 35.88
```

#### lambda_hf.py lambda evaluate log

```shell
# Check LAMBADA...
# 100 ppl 42.4 acc 34.0
# 200 ppl 29.3 acc 37.0
# 300 ppl 25.94 acc 39.0
# 400 ppl 27.27 acc 36.75
# 500 ppl 28.28 acc 35.4
# 600 ppl 27.02 acc 35.83
...
# 5000 ppl 26.17 acc 35.82
# 5100 ppl 26.15 acc 35.86
# 5153 ppl 26.14 acc 35.86
```

### Plan

- [x] Support RWKV5.0 model.
- [x] Support RWKV5.2 model.
- [x] Support Batch Infer For RWKV5.2 model.
- [ ] Support RWKV6.0 model.


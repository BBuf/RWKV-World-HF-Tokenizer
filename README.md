## RWKV World Model HuggingFace Tokenizer

### Huggingface Project With This Repo

- [BBuf/RWKV-4-World-169M](https://huggingface.co/BBuf/RWKV-4-World-169M)
### RWKV World Model's HuggingFace Version Tokenizer

The reference program below compares the encoding and decoding results of the original tokenizer and the HuggingFace version tokenizer for different sentences.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rwkv_tokenizer import TRIE_TOKENIZER
token_path = "/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer"

origin_tokenizer = TRIE_TOKENIZER('/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_vocab_v20230424.txt')

from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)

# test encoder
assert hf_tokenizer("Hello")['input_ids'] == origin_tokenizer.encode('Hello')
assert hf_tokenizer("S:2")['input_ids'] == origin_tokenizer.encode('S:2')
assert hf_tokenizer("Made in China")['input_ids'] == origin_tokenizer.encode('Made in China')
assert hf_tokenizer("今天天气不错")['input_ids'] == origin_tokenizer.encode('今天天气不错')
assert hf_tokenizer("男：听说你们公司要派你去南方工作?")['input_ids'] == origin_tokenizer.encode('男：听说你们公司要派你去南方工作?')

# test decoder
assert hf_tokenizer.decode(hf_tokenizer("Hello")['input_ids']) == 'Hello'
assert hf_tokenizer.decode(hf_tokenizer("S:2")['input_ids']) == 'S:2'
assert hf_tokenizer.decode(hf_tokenizer("Made in China")['input_ids']) == 'Made in China'
assert hf_tokenizer.decode(hf_tokenizer("今天天气不错")['input_ids']) == '今天天气不错'
assert hf_tokenizer.decode(hf_tokenizer("男：听说你们公司要派你去南方工作?")['input_ids']) == '男：听说你们公司要派你去南方工作?'
```


### Huggingface RWKV World Model Convert

Using the script `scripts/convert_rwkv_world_model_to_hf.sh` , convert the PyTorch format model from the huggingface `BlinkDL/rwkv-4-world` project to the Huggingface format. Here, we take 0.1B as an example.

```shell
#!/bin/bash
set -x

cd scripts
python convert_rwkv_checkpoint_to_hf.py --repo_id BlinkDL/rwkv-4-world \
 --checkpoint_file RWKV-4-World-0.1B-v1-20230520-ctx4096.pth \
 --output_dir ../rwkv4-world4-0.1b-model/ \
 --tokenizer_file /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer \
 --size 169M \
 --is_world_tokenizer True
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer/rwkv_vocab_v20230424.json ../rwkv4-world4-0.1b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer/tokenization_rwkv_world.py ../rwkv4-world4-0.1b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer/tokenizer_config.json ../rwkv4-world4-0.1b-model/
```


### Run Huggingface RWKV World Model

The `run_hf_world_model_xxx.py` demonstrates how to load the converted model using Huggingface's `AutoModelForCausalLM` and how to perform model inference using the custom `RWKV World Model's HuggingFace Version Tokenizer` loaded through `AutoTokenizer`:

#### CPU

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv4-world4-0.1b-model/")
tokenizer = AutoTokenizer.from_pretrained("/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv4-world4-0.1b-model/", trust_remote_code=True)

text = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
prompt = f'Question: {text.strip()}\n\nAnswer:'

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(inputs["input_ids"], max_new_tokens=256)
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
```

output:

```shell
Question: In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.

Answer: The researchers discovered a mysterious finding in a remote, undisclosed valley, in a remote, undisclosed valley.
```

#### GPU

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv4-world4-0.1b-model/", torch_dtype=torch.float16).to(0)
tokenizer = AutoTokenizer.from_pretrained("/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv4-world4-0.1b-model/", trust_remote_code=True)

text = "你叫什么名字？"
prompt = f'Question: {text.strip()}\n\nAnswer:'

inputs = tokenizer(prompt, return_tensors="pt").to(0)
output = model.generate(inputs["input_ids"], max_new_tokens=40)
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
```

output:

```shell
Question: 你叫什么名字？

Answer: 我是一个人工智能语言模型，没有具体的身份或者特征，也没有能力进行人类的任何任务
```

### Check Lambda


"The `lambda_pt.py` and `lambda_hf.py` files under the `check_lambda` folder evaluate the lambda dataset using the original PyTorch model of RWKV4 World 169M and the HuggingFace model, respectively. From the logs, it can be observed that the evaluation results they obtained are essentially the same."

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


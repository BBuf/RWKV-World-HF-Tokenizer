## RWKV World模型的HuggingFace Tokenizer

### 使用此仓库的Huggingface项目

> 上传转换后的模型到Huggingface上时，如果bin文件太大需要使用这个指令 `transformers-cli lfs-enable-largefiles` 解除大小限制.

- [RWKV/rwkv-5-world-169m](https://huggingface.co/RWKV/rwkv-5-world-169m)
- [RWKV/rwkv-4-world-169m](https://huggingface.co/RWKV/rwkv-4-world-169m)
- [RWKV/rwkv-4-world-430m](https://huggingface.co/RWKV/rwkv-4-world-430m)
- [RWKV/rwkv-4-world-1b5](https://huggingface.co/RWKV/rwkv-4-world-1b5)
- [RWKV/rwkv-4-world-3b](https://huggingface.co/RWKV/rwkv-4-world-3b)
- [RWKV/rwkv-4-world-7b](https://huggingface.co/RWKV/rwkv-4-world-7b)

### RWKV World模型的HuggingFace版本的Tokenizer

下面的参考程序比较了原始tokenizer和HuggingFace版本的tokenizer对不同句子的编码和解码结果。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rwkv_tokenizer import TRIE_TOKENIZER
token_path = "/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer"

origin_tokenizer = TRIE_TOKENIZER('/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_vocab_v20230424.txt')

from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)

# 测试编码器
assert hf_tokenizer("Hello")['input_ids'] == origin_tokenizer.encode('Hello')
assert hf_tokenizer("S:2")['input_ids'] == origin_tokenizer.encode('S:2')
assert hf_tokenizer("Made in China")['input_ids'] == origin_tokenizer.encode('Made in China')
assert hf_tokenizer("今天天气不错")['input_ids'] == origin_tokenizer.encode('今天天气不错')
assert hf_tokenizer("男：听说你们公司要派你去南方工作?")['input_ids'] == origin_tokenizer.encode('男：听说你们公司要派你去南方工作?')

# 测试解码器
assert hf_tokenizer.decode(hf_tokenizer("Hello")['input_ids']) == 'Hello'
assert hf_tokenizer.decode(hf_tokenizer("S:2")['input_ids']) == 'S:2'
assert hf_tokenizer.decode(hf_tokenizer("Made in China")['input_ids']) == 'Made in China'
assert hf_tokenizer.decode(hf_tokenizer("今天天气不错")['input_ids']) == '今天天气不错'
assert hf_tokenizer.decode(hf_tokenizer("男：听说你们公司要派你去南方工作?")['input_ids']) == '男：听说你们公司要派你去南方工作?'
```

### Huggingface RWKV World模型转换

使用脚本`scripts/convert_rwkv4_model_to_hf.sh`，将huggingface的`BlinkDL/rwkv-4-world`项目中的PyTorch格式模型转换为Huggingface格式。这里，我们以0.1B为例。

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

使用脚本 `scripts/convert_rwkv5_world_model_to_hf.sh`，将来自 huggingface `BlinkDL/rwkv-5-world` 项目的 PyTorch 格式模型转换为 Huggingface 格式。在这里，我们以 3B 为例。

```shell
#!/bin/bash
set -x

cd scripts
python convert_rwkv5_checkpoint_to_hf.py --repo_id BlinkDL/rwkv-5-world \
 --checkpoint_file RWKV-5-World-3B-v2-OnlyForTest_14%_trained-20231006-ctx4096.pth \
 --output_dir ../rwkv5-v2-world-3b-model/ \
 --tokenizer_file /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer \
 --size 3B \
 --is_world_tokenizer True \
 --model_version "5_2"

cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer/rwkv_vocab_v20230424.json ../rwkv5-v2-world-3b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer/tokenization_rwkv_world.py ../rwkv5-v2-world-3b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer/tokenizer_config.json ../rwkv5-v2-world-3b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5_model/configuration_rwkv5.py ../rwkv5-v2-world-3b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5_model/modeling_rwkv5.py ../rwkv5-v2-world-3b-model/
```

另外，您需要在生成文件夹中的 `config.json` 文件开头添加以下几行：

```json
"architectures": [
    "RwkvForCausalLM"
],
"auto_map": {
    "AutoConfig": "configuration_rwkv5.Rwkv5Config",
    "AutoModelForCausalLM": "modeling_rwkv5.RwkvForCausalLM"
},
```

### 运行Huggingface的RWKV World模型

`run_hf_world_model_xxx.py`演示了如何使用Huggingface的`AutoModelForCausalLM`加载转换后的模型，以及如何使用通过`AutoTokenizer`加载的自定义`RWKV World模型的HuggingFace版本的Tokenizer`进行模型推断：

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

输出：

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

输出：

```shell
Question: 你叫什么名字？

Answer: 我是一个人工智能语言模型，没有具体的身份或者特征，也没有能力进行人类的任何任务
```

### 检查Lambda



`check_lambda`文件夹下的`lambda_pt.py`和`lambda_hf.py`文件分别使用RWKV4 World 169M的原始PyTorch模型和HuggingFace模型对lambda数据集进行评估。从日志中可以看出，他们得到的评估结果基本上是一样的。

#### lambda_pt.py lambda评估日志

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

#### lambda_hf.py lambda评估日志

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

### 计划

- [x] 支持 RWKV5.0 模型。
- [ ] 支持 RWKV5.2 模型。



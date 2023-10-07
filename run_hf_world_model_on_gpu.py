import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv4-world4-0.1b-model/", torch_dtype=torch.float16).to(0)
tokenizer = AutoTokenizer.from_pretrained("/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv4-world4-0.1b-model/", trust_remote_code=True)

text = "你叫什么名字？"
prompt = f'Question: {text.strip()}\n\nAnswer:'

inputs = tokenizer(prompt, return_tensors="pt").to(0)
output = model.generate(inputs["input_ids"], max_new_tokens=40)
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))

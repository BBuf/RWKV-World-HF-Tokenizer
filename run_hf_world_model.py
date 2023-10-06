
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv4-world4-0.1b-model")
tokenizer = AutoTokenizer.from_pretrained("/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer", trust_remote_code=True)

prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(inputs["input_ids"], max_new_tokens=256)
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))

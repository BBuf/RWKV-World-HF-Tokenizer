from transformers import AutoModelForCausalLM, AutoTokenizer
token_path = "/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)

print(tokenizer("Hello"))
assert tokenizer("Hello")['input_ids'][0] == 33155

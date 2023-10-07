from transformers import AutoModelForCausalLM, AutoTokenizer
from rwkv_tokenizer import TRIE_TOKENIZER
tokenizer_file_path = "/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer"

origin_tokenizer = TRIE_TOKENIZER('/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_vocab_v20230424.txt')

from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path, trust_remote_code=True)

# test encoder
assert hf_tokenizer("Hello")['input_ids'] == origin_tokenizer.encode('Hello')
assert hf_tokenizer("S:2")['input_ids'] == origin_tokenizer.encode('S:2')
assert hf_tokenizer("Made in China")['input_ids'] == origin_tokenizer.encode('Made in China')
assert hf_tokenizer("今天天气不错")['input_ids'] == origin_tokenizer.encode('今天天气不错')
assert hf_tokenizer("男：听说你们公司要派你去南方工作?")['input_ids'] == origin_tokenizer.encode('男：听说你们公司要派你去南方工作?')
assert hf_tokenizer("Pré")['input_ids'] == origin_tokenizer.encode('Pré')

# test decoder
assert hf_tokenizer.decode(hf_tokenizer("Hello")['input_ids']) == 'Hello'
assert hf_tokenizer.decode(hf_tokenizer("S:2")['input_ids']) == 'S:2'
assert hf_tokenizer.decode(hf_tokenizer("Made in China")['input_ids']) == 'Made in China'
assert hf_tokenizer.decode(hf_tokenizer("今天天气不错")['input_ids']) == '今天天气不错'
assert hf_tokenizer.decode(hf_tokenizer("男：听说你们公司要派你去南方工作?")['input_ids']) == '男：听说你们公司要派你去南方工作?'


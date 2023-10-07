import re
import json

# 初始化一个空字典来存储转换后的数据
result = {}

# 使用正则表达式提取每行的数字和字符串
# 这个正则表达式可以匹配单引号或双引号括起来的字符串
pattern = r"(\d+)\s+(['\"])(.*?)\2\s+(\d+)"

# 读取文本文件
with open("/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_vocab_v20230424.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

for l in lines:
    idx = int(l[:l.index(' ')])
    x = eval(l[l.index(' '):l.rindex(' ')])
    x = x.encode("utf-8") if isinstance(x, str) else x
    assert isinstance(x, bytes)
    assert len(x) == int(l[l.rindex(' '):])
    result[x.decode('utf-8',  errors='ignore')] = idx  # 将bytes转换为str

print(result)
# 使用json模块保存字典为json文件
with open('/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer/rwkv_vocab_v20230424.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4)

print("Conversion completed!")
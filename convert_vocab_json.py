import re
import json

# 初始化一个空字典来存储转换后的数据
result = {}

# 使用正则表达式提取每行的数字和字符串
# 这个正则表达式可以匹配单引号或双引号括起来的字符串
pattern = r"(\d+)\s+(['\"])(.*?)\2\s+(\d+)"

# 读取文本文件
with open('/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_vocab_v20230424.txt', 'r', encoding='utf-8') as f:
    for line in f:
        match = re.match(pattern, line)
        if match:
            value, _, key, _ = match.groups()
            result[key] = int(value)
        else:
            parts = line.split(' ', 2)
            value = int(parts[0])
            key = str(parts[1])
            result[key] = int(value)

# 使用json模块保存字典为json文件
with open('/Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_vocab_v20230424.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("Conversion completed!")
#!/bin/bash
set -x

cd /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/scripts
python convert_rwkv_checkpoint_to_hf.py --repo_id https://huggingface.co/BlinkDL/rwkv-4-world \
 --checkpoint_file https://huggingface.co/BlinkDL/rwkv-4-world/blob/main/RWKV-4-World-0.1B-v1-20230520-ctx4096.pth \
 --output_dir /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/models/ \
 --tokenizer_file /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer \
 --is_world_tokenizer

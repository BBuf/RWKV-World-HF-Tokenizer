#!/bin/bash
set -x

cd scripts
python convert_rwkv_checkpoint_to_hf.py --repo_id BlinkDL/rwkv-4-world \
 --checkpoint_file RWKV-4-World-0.1B-v1-20230520-ctx4096.pth \
 --output_dir ../rwkv4-world4-0.1b-model/ \
 --tokenizer_file /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer \
 --size 169M \
 --is_world_tokenizer True

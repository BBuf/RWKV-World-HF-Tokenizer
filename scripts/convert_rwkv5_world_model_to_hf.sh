#!/bin/bash
set -x

cd scripts
python convert_rwkv5_checkpoint_to_hf.py --repo_id BlinkDL/rwkv-5-world \
 --checkpoint_file RWKV-5-World-0.1B-v1-20230803-ctx4096.pth \
 --output_dir ../rwkv5-world-169m-model/ \
 --tokenizer_file /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer \
 --size 169M \
 --is_world_tokenizer True

cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5.0_model/configuration_rwkv5.py ../rwkv5-world-169m-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5.0_model/modeling_rwkv5.py ../rwkv5-world-169m-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer/rwkv_vocab_v20230424.json ../rwkv5-world-169m-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer/tokenization_rwkv_world.py ../rwkv5-world-169m-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_tokenizer/tokenizer_config.json ../rwkv5-world-169m-model/

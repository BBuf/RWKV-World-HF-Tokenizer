#!/bin/bash
set -x

python -m scripts.convert_rwkv6_checkpoint_to_hf \
 --repo_id BlinkDL/rwkv-6-world \
 --checkpoint_file RWKV-x060-World-1B6-v2-20240208-ctx4096.pth \
 --output_dir ./rwkv6-world-1b6-v2-model/ \
 --tokenizer_file ./rwkv_world_tokenizer \
 --size 1B6 \
 --is_world_tokenizer True

cp ./rwkv_world_tokenizer/rwkv_vocab_v20230424.txt ./rwkv6-world-1b6-v2-model/
cp ./rwkv_world_tokenizer/tokenization_rwkv_world.py ./rwkv6-world-1b6-v2-model/
cp ./rwkv_world_tokenizer/tokenizer_config.json ./rwkv6-world-1b6-v2-model/
cp ./rwkv_world_v6_model_batch/configuration_rwkv6.py ./rwkv6-world-1b6-v2-model/
cp ./rwkv_world_v6_model_batch/modeling_rwkv6.py ./rwkv6-world-1b6-v2-model/
cp ./rwkv_world_v6_model_batch/generation_config.json ./rwkv6-world-1b6-v2-model/

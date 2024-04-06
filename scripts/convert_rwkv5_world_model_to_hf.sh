#!/bin/bash
set -x

cd scripts
python convert_rwkv5_checkpoint_to_hf.py --repo_id RWKV/v5-Eagle-7B \
 --checkpoint_file RWKV-v5-Eagle-World-7B-v2-20240128-ctx4096.pth \
 --output_dir ../../rwkv_model/v5-Eagle-7B/ \
 --tokenizer_file ../rwkv5_world_tokenizer \
 --size 7B \
 --is_world_tokenizer True

cp ../rwkv5_world_tokenizer/vocab.txt ../../rwkv_model/v5-Eagle-7B/
cp ../rwkv5_world_tokenizer/tokenization_rwkv5.py ../../rwkv_model/v5-Eagle-7B/
cp ../rwkv5_world_tokenizer/tokenizer_config.json ../../rwkv_model/v5-Eagle-7B/
cp ../rwkv5_model/configuration_rwkv5.py ../../rwkv_model/v5-Eagle-7B/
cp ../rwkv5_model/modeling_rwkv5.py ../../rwkv_model/v5-Eagle-7B/
cp ../rwkv5_model/generation_config.json ../../rwkv_model/v5-Eagle-7B/

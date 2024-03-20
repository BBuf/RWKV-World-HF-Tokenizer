#!/bin/bash
set -x

cd scripts
python convert_rwkv6_checkpoint_to_hf.py --repo_id BlinkDL/rwkv-6-world \
 --checkpoint_file RWKV-x060-World-1B6-v2-20240208-ctx4096.pth \
 --output_dir ../../rwkv_model/rwkv6-v2-world-1b6/ \
 --tokenizer_file ../rwkv6_model \
 --size 1B6 \
 --is_world_tokenizer True

cp ../rwkv6_model/vocab.txt ../../rwkv_model/rwkv6-v2-world-1b6/
cp ../rwkv6_model/tokenization_rwkv5.py ../../rwkv_model/rwkv6-v2-world-1b6/
cp ../rwkv6_model/tokenizer_config.json ../../rwkv_model/rwkv6-v2-world-1b6/
cp ../rwkv6_model/configuration_rwkv6.py ../../rwkv_model/rwkv6-v2-world-1b6/
cp ../rwkv6_model/modeling_rwkv6.py ../../rwkv_model/rwkv6-v2-world-1b6/
cp ../rwkv6_model/generation_config.json ../../rwkv_model/rwkv6-v2-world-1b6/

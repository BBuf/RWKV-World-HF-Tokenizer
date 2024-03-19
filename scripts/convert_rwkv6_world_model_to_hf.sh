#!/bin/bash
set -x

cd scripts
python convert_rwkv6_checkpoint_to_hf.py --repo_id BlinkDL/rwkv-5-world \
 --checkpoint_file RWKV-5-World-3B-v2-20231118-ctx16k.pth \
 --output_dir ../../rwkv6-v2-world-3b/ \
 --tokenizer_file ../rwkv6_model \
 --size 3B \
 --is_world_tokenizer True

cp ../rwkv6_model/vocab.txt ../../rwkv6-v2-world-3b/
cp ../rwkv6_model/tokenization_rwkv6.py ../../rwkv6-v2-world-3b/
cp ../rwkv6_model/tokenizer_config.json ../../rwkv6-v2-world-3b/
cp ../rwkv6_model/configuration_rwkv6.py ../../rwkv6-v2-world-3b/
cp ../rwkv6_model/modeling_rwkv6.py ../../rwkv6-v2-world-3b/
cp ../rwkv6_model/generation_config.json ../../rwkv6-v2-world-3b/

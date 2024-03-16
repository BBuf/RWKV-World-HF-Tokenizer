#!/bin/bash
set -x

cd scripts
# python convert_rwkv5_checkpoint_to_hf.py --repo_id BlinkDL/rwkv-5-world \
#  --checkpoint_file RWKV-5-World-3B-v2-20231118-ctx16k.pth \
#  --output_dir ../../rwkv5-v2-world-3b/ \
#  --tokenizer_file ../rwkv5_model \
#  --size 3B \
#  --is_world_tokenizer True

cp ../rwkv5_model/vocab.txt ../../rwkv5-v2-world-3b/
cp ../rwkv5_model/tokenization_rwkv5.py ../../rwkv5-v2-world-3b/
cp ../rwkv5_model/tokenizer_config.json ../../rwkv5-v2-world-3b/
cp ../rwkv5_model/configuration_rwkv5.py ../../rwkv5-v2-world-3b/
cp ../rwkv5_model/modeling_rwkv5.py ../../rwkv5-v2-world-3b/
cp ../rwkv5_model/generation_config.json ../../rwkv5-v2-world-3b/

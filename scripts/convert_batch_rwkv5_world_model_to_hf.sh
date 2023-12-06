#!/bin/bash
set -x

cd scripts
python convert_rwkv5_checkpoint_to_hf.py --repo_id BlinkDL/rwkv-5-world \
 --checkpoint_file RWKV-5-World-3B-v2-OnlyForTest_14%_trained-20231006-ctx4096.pth \
 --output_dir ../rwkv5-v2-world-3b-model/ \
 --tokenizer_file /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5_model_batch \
 --size 3B \
 --is_world_tokenizer True

cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5_model_batch/rwkv_vocab_v20230424.txt ../rwkv5-v2-world-3b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5_model_batch/tokenization_rwkv_world.py ../rwkv5-v2-world-3b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5_model_batch/tokenizer_config.json ../rwkv5-v2-world-3b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5_model_batch/configuration_rwkv5.py ../rwkv5-v2-world-3b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5_model_batch/modeling_rwkv5.py ../rwkv5-v2-world-3b-model/
cp /Users/bbuf/工作目录/RWKV/RWKV-World-HF-Tokenizer/rwkv_world_v5_model_batch/generation_config.json ../rwkv5-v2-world-3b-model/

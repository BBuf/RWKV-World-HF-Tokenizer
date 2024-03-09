import sys
import math
import torch
from collections import OrderedDict
import re

if len(sys.argv) != 3:
    print(f"Converts RWKV5.2 pth (non-huggingface) checkpoint to RWKV6.0")
    print("Usage: python convert5to6.py in_file out_file")
    exit()

model_path = sys.argv[1]

print("Loading file...")
state_dict = torch.load(model_path, map_location='cpu')

def convert_state_dict(state_dict):
    n_layer = 0
    n_embd = 0
    dim_att = 0

    state_dict_keys = list(state_dict.keys())
    for name in state_dict_keys:
        weight = state_dict.pop(name)

        # convert time_decay from (self.n_head, self.head_size) to (1,1,args.dim_att)
        if '.att.time_decay' in name:
            weight = weight.view(1,1,weight.size(0)*weight.size(1))
            n_embd = dim_att = weight.size(-1) 
        # convert time_mix_k, v, r, g into time_maa for both TimeMix and FFN
        if '.time_mix_' in name:
            name = name[:-5] + 'maa_' + name[-1:]
            weight = 1.0 - weight

        if name.startswith('blocks.'):
            layer_id_match = re.search(r"blocks\.(\d+)\.att", name)
            if layer_id_match is not None:
                n_layer = max(n_layer, int(layer_id_match.group(1)) + 1)

        state_dict[name] = weight

    # add in new params not in 5.2
    for layer_id in range(n_layer):
        layer_name = f'blocks.{layer_id}.att'

        ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
        ddd = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            ddd[0, 0, i] = i / n_embd

        state_dict[layer_name + '.time_maa_x'] = (1.0 - torch.pow(ddd, ratio_1_to_almost0))
        state_dict[layer_name + '.time_maa_w'] = (1.0 - torch.pow(ddd, ratio_1_to_almost0))

        TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
        state_dict[layer_name + '.time_maa_w1'] = (torch.zeros(n_embd, TIME_MIX_EXTRA_DIM*5).uniform_(-1e-4, 1e-4))
        state_dict[layer_name + '.time_maa_w2'] = (torch.zeros(5, TIME_MIX_EXTRA_DIM, n_embd).uniform_(-1e-4, 1e-4))

        TIME_DECAY_EXTRA_DIM = 64
        state_dict[layer_name + '.time_decay_w1'] = (torch.zeros(n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
        state_dict[layer_name + '.time_decay_w2'] = (torch.zeros(TIME_DECAY_EXTRA_DIM, dim_att).uniform_(-1e-4, 1e-4))

    print(f"n_layer: {n_layer}\nn_embd: {n_embd}")

    return state_dict

state_dict = convert_state_dict(state_dict)

torch.save(state_dict,sys.argv[2])
print("DONE. File written.")

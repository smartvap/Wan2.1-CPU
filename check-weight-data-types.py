#!/usr/bin/env python3
"""
检查模型权重的数据类型
"""

import torch
import os

model_dir = "/opt/models/Wan2.1-T2V-1.3B"

# 检查主要模型文件
files_to_check = [
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth"
]

for filename in files_to_check:
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        print(f"\n检查: {filename}")
        try:
            if filename.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(filepath)
            else:
                state_dict = torch.load(filepath, map_location='cpu')
            
            if isinstance(state_dict, dict):
                dtypes = {}
                for key, tensor in list(state_dict.items())[:10]:  # 只检查前10个
                    if isinstance(tensor, torch.Tensor):
                        dtypes.setdefault(str(tensor.dtype), 0)
                        dtypes[str(tensor.dtype)] += 1
                print(f"  数据类型分布: {dtypes}")
            else:
                print(f"  非字典类型: {type(state_dict)}")
        except Exception as e:
            print(f"  加载失败: {e}")

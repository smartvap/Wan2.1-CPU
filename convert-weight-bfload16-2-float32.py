#!/usr/bin/env python3
"""
转换模型权重为 float32
"""

import torch
import os
import glob

#model_dir = "/opt/models/Wan2.1-T2V-1.3B"
model_dir = "/opt/models/Wan2.1-T2V-14B"

# 查找所有模型文件
model_files = []
for ext in ['*.pth', '*.pt', '*.safetensors', '*.bin']:
    model_files.extend(glob.glob(os.path.join(model_dir, '**', ext), recursive=True))
    model_files.extend(glob.glob(os.path.join(model_dir, ext)))

print(f"找到 {len(model_files)} 个模型文件")

for filepath in model_files:
    print(f"处理: {filepath}")
    try:
        # 加载
        if filepath.endswith('.safetensors'):
            from safetensors.torch import load_file, save_file
            state_dict = load_file(filepath)
        else:
            state_dict = torch.load(filepath, map_location='cpu')
        
        # 转换
        converted = False
        if isinstance(state_dict, dict):
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], torch.Tensor):
                    if state_dict[key].dtype in (torch.bfloat16, torch.float16):
                        state_dict[key] = state_dict[key].to(torch.float32)
                        converted = True
        
        # 保存
        if converted:
            backup = filepath + '.bf16_backup'
            if not os.path.exists(backup):
                os.rename(filepath, backup)
            
            if filepath.endswith('.safetensors'):
                save_file(state_dict, filepath)
            else:
                torch.save(state_dict, filepath)
            
            print(f"  ✓ 已转换并备份到 {backup}")
        else:
            print(f"  ⏭️ 无需转换")
            
    except Exception as e:
        print(f"  ✗ 错误: {e}")

print("转换完成！")

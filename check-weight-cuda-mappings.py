#!/usr/bin/env python3
"""
转换模型权重为 float32 并修复设备映射
"""

import torch
import os
import glob

model_dir = "/opt/models/Wan2.1-T2V-1.3B"

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
            # 明确指定map_location='cpu'，确保加载到CPU
            state_dict = torch.load(filepath, map_location='cpu')

        # 转换
        converted = False
        if isinstance(state_dict, dict):
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], torch.Tensor):
                    # 1. 修复设备映射：确保张量在CPU上
                    if state_dict[key].is_cuda:
                        state_dict[key] = state_dict[key].cpu()
                    
                    # 2. 数据类型转换
                    if state_dict[key].dtype in (torch.bfloat16, torch.float16):
                        state_dict[key] = state_dict[key].to(torch.float32)
                        converted = True
                    
                    # 3. 移除可能存在的持久性CUDA引用
                    if hasattr(state_dict[key], '_torch'):
                        delattr(state_dict[key], '_torch')
        
        # 特殊情况：检查是否包含"module."前缀（多GPU训练）
        has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
        if has_module_prefix:
            print("  检测到 'module.' 前缀，移除中...")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # 移除 'module.'
                else:
                    new_key = key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
            converted = True

        # 保存
        if converted:
            backup = filepath + '.bf16_backup'
            if not os.path.exists(backup):
                os.rename(filepath, backup)
            
            if filepath.endswith('.safetensors'):
                save_file(state_dict, filepath)
            else:
                # 明确保存为CPU张量
                torch.save(state_dict, filepath, _use_new_zipfile_serialization=True)
                print(f"  ✓ 已转换并备份到 {backup}")
        else:
            print(f"  ⏭️ 无需转换")

    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()

print("转换完成！")

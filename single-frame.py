#!/usr/bin/env python3
"""
使用通义万相官方方法生成视频
"""

import os
import sys
import torch

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, '/opt/Wan2.1')

from wan.configs import get_config
from wan.text2video import WanT2V

print("=== 生成测试视频 ===")

# 1. 加载配置
print("1. 加载配置...")
config = get_config('t2v-1.3B')

# 2. 创建模型
print("2. 创建模型...")
try:
    wan_t2v = WanT2V(
        config=config,
        device='cpu',
        dtype=torch.float32
    )
    print("   模型创建成功")
except Exception as e:
    print(f"   模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 生成视频（少量帧用于测试）
print("3. 生成视频（测试用少量帧）...")
try:
    result = wan_t2v.generate(
        prompt="一只可爱的猫咪",  # 简单提示
        num_frames=16,            # 少量帧
        height=256,               # 小分辨率
        width=384,
        num_inference_steps=8,    # 少量步数
        guidance_scale=7.5
    )
    
    print(f"   生成完成，结果类型: {type(result)}")
    
    # 检查返回结果
    if isinstance(result, dict):
        print("   返回字典包含的键:", list(result.keys()))
        for key, value in result.items():
            if hasattr(value, 'shape'):
                print(f"     {key}: {value.shape}, {value.dtype}")
    
    # 假设视频帧在 'frames' 键中
    if isinstance(result, dict) and 'frames' in result:
        video = result['frames']
    elif hasattr(result, 'frames'):
        video = result.frames
    else:
        video = result  # 可能是直接返回的视频
    
    print(f"   视频数据: {type(video)}, 形状: {video.shape if hasattr(video, 'shape') else 'N/A'}")
    
    # 保存第一帧
    if hasattr(video, 'shape') and len(video.shape) >= 4:
        import numpy as np
        import cv2
        
        frame = video[0]  # 第一帧
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        
        print(f"   第一帧形状: {frame.shape}, dtype: {frame.dtype}")
        print(f"   值范围: [{frame.min():.3f}, {frame.max():.3f}]")
        
        # 转换到 [0, 255]
        if frame.max() <= 1.0 and frame.min() >= -1:
            frame = ((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        elif frame.max() <= 1.0 and frame.min() >= 0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        
        # 保存
        if frame.shape[-1] == 3:  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite('first_frame_test.png', frame)
        print("   ✅ 第一帧已保存为 first_frame_test.png")
        
except Exception as e:
    print(f"   生成失败: {e}")
    import traceback
    traceback.print_exc()

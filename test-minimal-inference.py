# test_minimal_inference.py
import torch
import sys
import os
sys.path.append('.')

def test_minimal_inference():
    """最小化推理测试"""
    print("=== 最小化推理测试 ===")
    
    # 设置环境
    torch.set_grad_enabled(False)
    
    # 1. 测试噪声采样
    print("\n1. 噪声采样测试...")
    batch_size = 1
    channels = 4
    height = 60
    width = 108
    
    # 在CPU上生成噪声
    torch.manual_seed(42)
    noise = torch.randn(batch_size, channels, height, width, device='cpu')
    print(f"噪声形状: {noise.shape}")
    print(f"噪声统计: min={noise.min():.4f}, max={noise.max():.4f}, mean={noise.mean():.4f}, std={noise.std():.4f}")
    
    # 2. 测试简单的变换
    print("\n2. 噪声变换测试...")
    # 模拟DiT的前向传播
    transformed = noise * 0.5  # 简单缩放
    print(f"变换后统计: min={transformed.min():.4f}, max={transformed.max():.4f}, mean={transformed.mean():.4f}")
    
    # 3. 测试VAE解码
    print("\n3. 尝试加载VAE并解码...")
    try:
        vae_path = "/opt/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
        vae_state = torch.load(vae_path, map_location='cpu')
        
        # 检查VAE结构
        decoder_keys = [k for k in vae_state.keys() if 'decoder' in k.lower()]
        print(f"VAE中包含{len(decoder_keys)}个解码器相关参数")
        if decoder_keys:
            print(f"前3个解码器参数: {decoder_keys[:3]}")
            
    except Exception as e:
        print(f"VAE测试失败: {e}")
    
    # 4. 测试最终输出
    print("\n4. 输出转换测试...")
    # 模拟模型输出（假设是[-1, 1]范围）
    model_output = torch.randn(batch_size, 3, 240, 432, device='cpu')  # 解码后的RGB
    
    # 转换为[0, 255]范围
    output_0_255 = ((model_output + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    print(f"原始输出范围: [{model_output.min():.4f}, {model_output.max():.4f}]")
    print(f"转换后范围: [{output_0_255.min()}, {output_0_255.max()}]")
    print(f"转换后数据类型: {output_0_255.dtype}")
    
    # 检查是否为纯色
    unique_colors = output_0_255.unique().shape[0]
    print(f"唯一颜色数量: {unique_colors}")
    
    if unique_colors < 10:
        print("⚠️ 警告: 输出颜色过于单一，可能是白屏/彩色雪花的征兆")
    else:
        print("✅ 输出颜色多样性正常")

if __name__ == "__main__":
    test_minimal_inference()

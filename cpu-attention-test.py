#!/usr/bin/env python3
"""
测试 CPU 注意力
"""

import torch
import sys
sys.path.insert(0, '/opt/Wan2.1')

# 导入我们的注意力模块
from wan.modules.attention import flash_attention, attention

print("测试 CPU 注意力...")

# 创建测试数据
batch_size = 2
seq_len_q = 10
seq_len_kv = 12
num_heads = 4
head_dim = 64

# 形状: [B, L, H, C]
q = torch.randn(batch_size, seq_len_q, num_heads, head_dim)
k = torch.randn(batch_size, seq_len_kv, num_heads, head_dim)
v = torch.randn(batch_size, seq_len_kv, num_heads, head_dim)

print(f"q 形状: {q.shape}")
print(f"k 形状: {k.shape}")
print(f"v 形状: {v.shape}")

# 测试 flash_attention
print("\n1. 测试 flash_attention (非因果):")
try:
    output = flash_attention(q, k, v, dtype=torch.float32)
    print(f"✓ 输出形状: {output.shape}")
    print(f"  输出 dtype: {output.dtype}")
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()

# 测试因果注意力
print("\n2. 测试 flash_attention (因果):")
try:
    output = flash_attention(q, k, v, causal=True, dtype=torch.float32)
    print(f"✓ 输出形状: {output.shape}")
except Exception as e:
    print(f"✗ 错误: {e}")

# 测试统一接口
print("\n3. 测试 attention 函数:")
try:
    output = attention(q, k, v, dtype=torch.float32)
    print(f"✓ 输出形状: {output.shape}")
except Exception as e:
    print(f"✗ 错误: {e}")

print("\n所有测试完成！")

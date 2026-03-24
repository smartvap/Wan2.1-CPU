# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn.functional as F
import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.float32,  # 改为 float32
    version=None,
):
    """
    CPU 版本的注意力实现
    替代原来的 flash_attn
    """
    # 参数检查
    half_dtypes = (torch.float16, torch.float32)  # 移除 bfloat16
    assert dtype in half_dtypes
    
    # 移除 CUDA 断言
    # assert q.device.type == 'cuda' and q.size(-1) <= 256
    
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    
    # 处理变长序列
    if q_lens is not None or k_lens is not None:
        warnings.warn('变长序列支持有限，使用简化实现')
    
    # 确保数据类型一致
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    
    if q_scale is not None:
        q = q * q_scale
    
    # 计算注意力分数
    # 形状: q: [B, Lq, H, C], k: [B, Lk, H, C]
    # 我们需要在最后两个维度计算点积
    
    # 转置以匹配维度
    q = q.transpose(1, 2)  # [B, H, Lq, C]
    k = k.transpose(1, 2)  # [B, H, Lk, C]
    v = v.transpose(1, 2)  # [B, H, Lk, C]
    
    # 计算缩放因子
    scale = softmax_scale or (q.size(-1) ** -0.5)
    q = q * scale
    
    # 计算注意力分数: [B, H, Lq, Lk]
    attn = torch.matmul(q, k.transpose(-2, -1))
    
    # 应用因果掩码
    if causal:
        mask = torch.triu(torch.ones(lq, lk, dtype=torch.bool, device=q.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Lq, Lk]
        attn = attn.masked_fill(mask, float('-inf'))
    
    # 应用滑动窗口掩码
    if window_size != (-1, -1):
        left, right = window_size
        if left > 0 or right > 0:
            mask = torch.ones(lq, lk, dtype=torch.bool, device=q.device)
            for i in range(lq):
                start = max(0, i - left)
                end = min(lk, i + right + 1)
                mask[i, start:end] = False
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Lq, Lk]
            attn = attn.masked_fill(mask, float('-inf'))
    
    # softmax
    attn = F.softmax(attn, dim=-1)
    
    # dropout
    if dropout_p > 0. and not deterministic:
        attn = F.dropout(attn, p=dropout_p)
    
    # 应用注意力
    output = torch.matmul(attn, v)  # [B, H, Lq, C]
    
    # 转置回原始形状
    output = output.transpose(1, 2)  # [B, Lq, H, C]
    
    # 恢复原始数据类型
    return output.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.float32,  # 改为 float32
    fa_version=None,
):
    """
    统一注意力接口
    总是使用我们的 CPU 版本
    """
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )

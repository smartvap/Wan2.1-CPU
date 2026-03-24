# test_model_loading.py
import torch
import sys
import os
sys.path.append('.')  # 添加当前路径

def test_model_loading():
    print("=== 测试模型加载 ===")
    
    # 1. 测试VAE模型
    print("\n1. 测试VAE模型加载...")
    vae_path = "/opt/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
    if os.path.exists(vae_path):
        try:
            vae_state = torch.load(vae_path, map_location='cpu')
            print(f"  VAE状态字典类型: {type(vae_state)}")
            if isinstance(vae_state, dict):
                print(f"  VAE键数量: {len(vae_state)}")
                for i, (k, v) in enumerate(list(vae_state.items())[:3]):
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        except Exception as e:
            print(f"  VAE加载失败: {e}")
    
    # 2. 测试DiT模型
    print("\n2. 测试DiT模型加载...")
    dit_path = "/opt/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    if os.path.exists(dit_path):
        try:
            from safetensors.torch import load_file
            dit_state = load_file(dit_path, device='cpu')
            print(f"  DiT状态字典类型: {type(dit_state)}")
            if isinstance(dit_state, dict):
                print(f"  DiT键数量: {len(dit_state)}")
                for i, (k, v) in enumerate(list(dit_state.items())[:3]):
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        except Exception as e:
            print(f"  DiT加载失败: {e}")
    
    # 3. 测试T5模型
    print("\n3. 测试T5模型加载...")
    t5_path = "/opt/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
    if os.path.exists(t5_path):
        try:
            t5_state = torch.load(t5_path, map_location='cpu')
            print(f"  T5状态字典类型: {type(t5_state)}")
            if isinstance(t5_state, dict):
                print(f"  T5键数量: {len(t5_state)}")
                for i, (k, v) in enumerate(list(t5_state.items())[:3]):
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        except Exception as e:
            print(f"  T5加载失败: {e}")
    
    # 4. 尝试导入Wan模块
    print("\n4. 测试Wan模块导入...")
    try:
        import wan
        print(f"  Wan模块版本: {wan.__version__ if hasattr(wan, '__version__') else '未知'}")
    except Exception as e:
        print(f"  Wan模块导入失败: {e}")

if __name__ == "__main__":
    test_model_loading()

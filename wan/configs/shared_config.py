# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = torch.float32
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = torch.float32

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
#wan_shared_cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
wan_shared_cfg.sample_neg_prompt = '过曝，静态画面，细节模糊，字幕，风格化，画作感，画面静止，整体发灰，低质量，JPEG压缩伪影，丑陋，残缺，多余的手指，畸形的手，畸形的脸，毁容，肢体变形，手指融合，静止，杂乱的背景，背景人群，逻辑错误，动作不连贯，帧率低，色彩失真，负片效果，色调异常，色差，褪色，色彩暗淡'

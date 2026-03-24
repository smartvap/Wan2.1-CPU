#!/bin/bash

CUDA_VISIBLE_DEVICES=-1 python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir /opt/models/Wan2.1-T2V-1.3B --offload_model True --t5_cpu --prompt "一只优雅的暹罗猫，拥有蓝色的眼睛和重点色的面部，在洒满冬日午后的阳光的窗台上放松，悠闲地舔着自己的前爪。背景是温暖的室内，有模糊的绿色植物和轻柔飘动的窗帘。光线柔和，充满生活气息。动态镜头缓慢平移，焦点集中在猫的面部表情和细腻的毛发纹理上。风格写实，4K高清，电影感，温馨的氛围。" --save_file mini-frames-wan2.1-cat.mp4 --sample_steps 15 --frame_num 15

CUDA_VISIBLE_DEVICES=-1 python generate.py --task t2v-14B --size 832*480 --ckpt_dir /opt/models/Wan2.1-T2V-14B --offload_model True --t5_cpu --prompt "两只可爱的鹧鸪，一雌一雄，雌鸟体羽灰褐，雄鸟拥有醒目的红眼圈和鲜艳的面颊。它们在林间覆着薄雪的苔石上相依偎，低头啄食着散落的草籽。背景是冬日的阔叶林，有模糊的光秃枝干和远处墨绿的松柏。柔和的晨光穿过稀疏的树冠，在林间投下长长的光柱。动态镜头缓缓推进，焦点集中在它们灵动的眼神、细腻的羽毛纹理以及啄食时轻微的颤动上。风格写实，4K高清，自然纪录片质感，宁静而充满生机。" --save_file wan2.1-cpu-鹧鸪-1.mp4 --sample_steps 20 --frame_num 60

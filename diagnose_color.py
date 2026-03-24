import cv2
import numpy as np
import os

def process_video_different_methods(input_path, output_dir):
    """
    尝试三种不同的色彩校正方法，生成三个对比视频。
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建三个输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_rgb = cv2.VideoWriter(os.path.join(output_dir, 'method1_rgb_invert.mp4'), fourcc, fps, (w, h))
    out_yuv = cv2.VideoWriter(os.path.join(output_dir, 'method2_yuv_invert.mp4'), fourcc, fps, (w, h))
    out_selective = cv2.VideoWriter(os.path.join(output_dir, 'method3_selective_invert.mp4'), fourcc, fps, (w, h))

    print("开始处理，尝试三种方法...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 方法1: 原始RGB全局反转 (就是你之前用的) ---
        result_rgb = 255 - frame

        # --- 方法2: 转换到YUV色彩空间，只反转Y（亮度）分量 ---
        # 这能修复“色调分离”问题
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(frame_yuv)
        y_inverted = 255 - y
        result_yuv_bgr = cv2.cvtColor(cv2.merge([y_inverted, u, v]), cv2.COLOR_YUV2BGR)

        # --- 方法3: 选择性通道反转 (适用于“红蓝颠倒”等) ---
        # 交换颜色通道，常用于修复“颜色反相但不完全”的问题
        b, g, r = cv2.split(frame)
        # 尝试交换红色和蓝色通道
        result_swap = cv2.merge([r, g, b])

        # 写入结果
        out_rgb.write(result_rgb)
        out_yuv.write(result_yuv_bgr)
        out_selective.write(result_swap)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"已处理 {frame_idx} 帧...")

    # 释放资源
    cap.release()
    out_rgb.release()
    out_yuv.release()
    out_selective.release()
    cv2.destroyAllWindows()

    print(f"\n处理完成！已生成三个测试视频在目录: {output_dir}")
    print("请按顺序检查，哪个效果最好：")
    print("1. 'method1_rgb_invert.mp4' - RGB全局反转 (原始方法)")
    print("2. 'method2_yuv_invert.mp4' - 仅反转Y亮度分量")
    print("3. 'method3_selective_invert.mp4' - 交换红蓝通道")

# --- 配置路径 ---
input_video = r"/opt/Wan2.1/mini-frames-wan2.1-cat.mp4"  # 你的原始负片视频
output_directory = r"/opt/Wan2.1/outputs"  # 输出目录，确保文件夹存在

# 如果输出目录不存在，则创建
os.makedirs(output_directory, exist_ok=True)

# 执行诊断
process_video_different_methods(input_video, output_directory)

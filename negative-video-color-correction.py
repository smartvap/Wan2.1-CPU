import cv2
import numpy as np
import os

def invert_video_colors(input_path, output_path):
    """
    将输入视频的色彩进行反转（负片效果校正为正片）。
    Args:
        input_path: 输入视频文件路径。
        output_path: 输出视频文件路径。
    """
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {input_path}")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器，也可用 'avc1'

    # 创建VideoWriter对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    frame_count = 0
    print(f"开始处理视频: {input_path}")
    print(f"视频信息: {width}x{height}, {fps} fps")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 核心步骤：色彩反转 (255 - 像素值)
        inverted_frame = 255 - frame

        # 写入处理后的帧
        out.write(inverted_frame)

        frame_count += 1
        if frame_count % 30 == 0:  # 每处理30帧打印一次进度
            print(f"已处理 {frame_count} 帧...")

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成! 总帧数: {frame_count}")
    print(f"校正后的视频已保存至: {output_path}")

# --- 使用方法：直接修改下面两个路径为你自己的 ---
input_video_path = r"/opt/Wan2.1/mini-frames-wan2.1-cat.mp4"  # 你的负片视频路径
output_video_path = r"/opt/Wan2.1/mini-frames-wan2.1-cat-correct.mp4"  # 输出视频路径

# 执行函数
invert_video_colors(input_video_path, output_video_path)

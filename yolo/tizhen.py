import cv2
import os

# 视频文件夹路径
video_folder = r"C:\Users\xunyi\Desktop\train_video1"
# 保存帧图像的文件夹路径
output_folder = r"C:\Users\xunyi\Desktop\video_tizhen1"
# 每秒提取的帧数（这里固定为6）
fps_target = 6

# 如果保存目录不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取文件夹中所有视频文件
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]

# 计数器，用来给每一帧命名
frame_count = 0

# 遍历每个视频文件
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    fps_video = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 每秒抽取3帧 → 每隔多少帧取一次
    frame_interval = int(fps_video / fps_target)

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 如果当前帧是目标帧，则保存
        if frame_index % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f'TRAIN{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        frame_index += 1

    cap.release()
    print(f'视频 "{video_file}" 已提取 {frame_index} 帧中的 {frame_count} 张图片')

print('所有视频处理完成！')

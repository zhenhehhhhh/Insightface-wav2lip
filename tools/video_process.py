import os
import cv2
import shutil
import numpy as np
import subprocess

in_video = 'D:\datasets\\005_8786.MXF'
cap = cv2.VideoCapture(in_video)  # 打开相机
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
# if fps != 25:
#     fps = 25  # ffmpeg -i xxx.mp4 -qscale 0 -r 25 -y xxx.mp4
#     command = 'ffmpeg -i {} -qscale 0 -r 25 -y {}'.format(in_video, in_video[:-4] + 'f25.mp4')
#     subprocess.call(command, shell=True)
#     cap = cv2.VideoCapture(in_video[:-4] + 'f25.mp4')  # 重新加载目标视频
count = 1
num = 1
while True:
    ret, frame = cap.read()  # 捕获一帧图像
    if not ret:
        break
    frame = np.rot90(frame, 3)
    frame_resize = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    # print(frame_resize.shape)
    # cv2.imshow('1', frame_resize)
    # cv2.waitKey(0)
    if count % 2 == 1:
        cv2.imwrite('D:\datasets\qijiang2ac6/' + str(num) + '.jpg', frame_resize)
        num += 1
    count += 1
    cv2.waitKey(1)
cap.release()

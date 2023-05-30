"""视频拼接"""
# -*- coding:utf-8 -*-
import numpy as np
import cv2, subprocess
import time

# dst_video_save = '4.mp4'
# command = 'ffmpeg -i {} -qscale 0 -r 25 -y {}'.format(dst_video_save, dst_video_save[:-4] + 'f25.mp4')
# subprocess.call(command, shell=True)

cap1 = cv2.VideoCapture('1306.mp4')
cap2 = cv2.VideoCapture('1675.mp4')
cap3 = cv2.VideoCapture('1919.mp4')
# cap4 = cv2.VideoCapture('1_3.mp4')
# cap5 = cv2.VideoCapture('4_h232.mp4')
fps = cap1.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
frame_w, frame_h = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('resultz.mp4', fourcc, fps, (int(frame_w * 3), frame_h))

while True:
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    ret, frame3 = cap3.read()
    # ret, frame4 = cap4.read()
    # ret, frame5 = cap5.read()
    if not ret:
        break
    merge = np.concatenate((frame1, frame2, frame3), 1)
    out.write(merge)
out.release()
audio_path = 'outputs/audio.wav'
command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 -vcodec libx264 {}'.format(audio_path, 'resultz.mp4', 'resultz_audio.mp4')
subprocess.call(command, shell=True)



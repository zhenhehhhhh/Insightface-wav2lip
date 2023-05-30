import os
import cv2
import numpy as np
from natsort import index_natsorted


vid_dir = 'peiac2k.mp4'
vid_dic = {}


cap = cv2.VideoCapture(vid_dir)  # 打开相机
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率

num = 0
while True:
    ret,frame = cap.read()  # 捕获一帧图像
    if ret:
        vid_dic[str(num)] = frame
        print(num)
        num = num + 1 
    else:
        break
cap.release()

for i in range(len(vid_dic)):
    img = vid_dic[str(i)]
    cv2.imshow('img', img)
    cv2.waitKey(1)
    print(i)

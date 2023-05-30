import os
import cv2
import numpy as np
from natsort import natsorted

# ---- 路径设置 ---- #
in_vid_dir = 'E:/0303-1/2R9A9704.MOV'
out_vid_dir = 'tools/pei1080.mp4'

# ---- 加载视频 ---- #
print(in_vid_dir)
cap = cv2.VideoCapture(in_vid_dir)  # 打开相机
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取视频帧率
totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
print('All-Frames:', totalFrame)

# ---- 视频保存 ---- #
H = 2212 # 帧高度
W = 1080  # 帧宽度
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(out_vid_dir, fourcc, fps, (W, H))

# ---- 循环加载 ---- #
count = -8
LW = 160
LH = 250
RW = 3740
RH = 2000


while True:
    ret, frame = cap.read()  # 捕获一帧图像
    if ret:
        if count>=0:
            print(str(count) + ':' + str(totalFrame))
            frame = frame[LH:RH, LW:RW]
            frame = cv2.resize(frame, (H, W))
            frame = np.rot90(frame, 1)
            cv2.imwrite('tools/temp_imgs/'+str(count)+'.jpg', frame)
            # out.write(frame)
            cv2.waitKey(1)
        count = count + 1
    else:
        break
cap.release()
# out.release()

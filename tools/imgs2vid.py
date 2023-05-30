import os
import cv2
from natsort import natsorted
import numpy as np

in_dir = 'chsy/1/'

flist = []
for filename in os.listdir(in_dir):
    if filename[-4:] == '.jpg':
        flist.append(filename)
flist = natsorted(flist)
flist = flist[0:2000]
print(flist)

frame = cv2.imread(in_dir + flist[0])
frame_w = frame.shape[1]
frame_h = frame.shape[0]
fps = 25
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(r'cqshangyou_1.mp4', fourcc, fps, (frame_w, frame_h))
for img in flist:
    frame = cv2.imread(in_dir + img)
    print(in_dir + img)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
for i in range(len(flist)-1, 0, -1):
    frame = cv2.imread(in_dir + flist[i])
    print(in_dir + flist[i])
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()



# in_dir = '4_frame/4_1c/'
#
# flist = []
# for filename in os.listdir(in_dir):
#     if filename[-4:] == '.jpg':
#         flist.append(filename)
# flist = natsorted(flist)
# flist = flist[921:1122]
# # flist = flist[18:78]
# print(flist)
#
# frame = cv2.imread(in_dir + flist[0])
# frame_w = frame.shape[1]
# frame_h = frame.shape[0]
# print(frame_w,frame_h)
# fps = 25
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('4_frame/biaoqing3_2.mp4', fourcc, fps, (frame_w, frame_h))
# for img in flist:
#     frame = cv2.imread(in_dir + img)
#     print(in_dir + img)
#     out.write(frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# out.release()

# # 视频循环
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out_video = cv2.VideoWriter('chq2_4.mp4', fourcc, 25, (1080, 2048))
#
# start = 298  # 85,210
# end = 400
# num = 14 # 循环次数
# root = 'chq2/2/'
# img = cv2.imread(root + str(start) + '.jpg')
# out_video.write(img)
# i = start + 1
# for _ in range(num):
#     while i != end:
#         img = cv2.imread(root + str(i) + '.jpg')
#         out_video.write(img)
#         i += 1
#     while i != start + 1:
#         img = cv2.imread(root + str(i) + '.jpg')
#         out_video.write(img)
#         i -= 1
# out_video.release()

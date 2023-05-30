# -*- coding:utf-8 -*-
import pygame
import sys
import cv2
import numpy as np
import os
import time

def video_show(vid_file, audio_file):
    pygame.init()  # 初始化pygame
    FPS = 25.0
    # 设置窗口位置
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (50, 70)
    FPSClock = pygame.time.Clock()
    # size = 1280, 720  # 设置窗口大小
    # screen = pygame.display.set_mode(size,pygame.FULLSCREEN)  # 显示窗口
    # screen = pygame.display.set_mode(size)  # 显示窗口
    # pygame.display.set_caption(u'音画同步播放')


    ogg = pygame.mixer.Sound(audio_file)
    # pygame.mixer.music.load("")

    videoCapture = cv2.VideoCapture(vid_file)
    
    # 真正时间
    realtime = int(1000 / FPS)

    # 阈值-要稍微留下空间
    jumpyu = 50

    # 跳帧计数器
    jumpnum = 0
    ogg.play()

    while True:
        a = pygame.time.get_ticks()

        if videoCapture.isOpened():
        #这里是跳帧工具
            if jumpnum != 0:
           #这个函数光读东西，比read少一个解码过程，所以能省点算点
                videoCapture.grab()
                jumpnum -= 1
                continue
            ret, frame = videoCapture.read()
            frame = cv2.resize(frame, (1280, 720))
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = pygame.surfarray.make_surface(frame)
            # frame = pygame.transform.flip(frame, False, True)
            # screen.blit(frame, (0, 0))


        for event in pygame.event.get():  # 遍历所有事件
            if event.type == pygame.QUIT:  # 如果单击关闭窗口，则退出
                sys.exit()
        cv2.imshow('imgs', frame)
        # cv2.waitKey(1)
        # time.sleep(0.01)
        # cv2.waitKey(1)
        # pygame.display.flip()  # 更新全部显示
        # pygame.display  # 更新全部显示
        time_next = FPSClock.tick(FPS)
        b = pygame.time.get_ticks()
        k = b - a
		#此处检测阈值，大于阈值做记录准备跳帧
        if k > jumpyu:
            print('超过阈值，跳帧')
            temp = (k - realtime)/realtime
            tempstr=str(temp)
            tempstr=tempstr[tempstr.find('.')+1:]
            if len(tempstr)>1 or tempstr[0]!='0':
                jumpnum+=(int(temp)+1)
                print('跳小数帧')
            else:
                print('跳整数帧')
                jumpnum+=int(temp)
            print('跳'+str(jumpnum)+'帧')

    pygame.quit()  # 退出pygame


if __name__ == '__main__':
    vid_file = 'AIactor-mini.mp4'
    audio_file = 'audio.wav'
    video_show(vid_file, audio_file)

# -*- coding:utf-8 -*-
import pygame
import sys
import cv2
import time
from pygame.locals import *



pygame.init() 
video_path = 'AIactor-mini.mp4'
ogg = pygame.mixer.Sound('audio.wav')
video = cv2.VideoCapture(video_path)
FPS = int(round(video.get(cv2.CAP_PROP_FPS)))
num = 0
ogg.play()

while True :
    if num == 0:
        T0 = time.time()

    if time.time()-T0 > num*(1./FPS):
        ret, frame = video.read()
        if ret == False:
            print('Total Time:', time.time()-T0)
            pygame.quit()
            sys.exit()
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        num += 1

    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()
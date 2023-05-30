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
 
FramePerSec = pygame.time.Clock()
 
Width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
pygame.init()
pygame.display.set_caption('OpenCV Video Player on Pygame')
 
screen = pygame.display.set_mode((Width, Height), 0, 32)
screen.fill([0,0,0])
num = 0

ogg.play()
 
while True :
 
    if num == 0:
        T0 = time.time()
 
    if time.time()-T0 > num*(1./FPS):
 
        ret, frame = video.read()
        TimeStamp = video.get(cv2.CAP_PROP_POS_MSEC)
 
        if ret == False:
            print('Total Time:', time.time()-T0)
            pygame.quit()
            sys.exit()
 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.transpose(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0,0))
 
        pygame.display.update()
        num += 1
 
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()
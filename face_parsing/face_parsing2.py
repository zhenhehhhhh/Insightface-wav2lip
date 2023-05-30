#!/usr/bin/python
# -*- encoding: utf-8 -*-

from .logger import setup_logger
from .model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2, time

n_classes = 19
net = BiSeNet(n_classes=n_classes)
#net.cuda()
net.load_state_dict(torch.load('./face_parsing/79999_iter.pth', map_location=torch.device('cpu')))
net.eval()
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def vis_parsing_maps(im, parsing_anno, stride):
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, (224, 224), fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    for pi in [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]:
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 255, 255])

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    return vis_parsing_anno_color


def face_parsing(img):
    with torch.no_grad():
        # img = Image.open(img)
        # image = img.resize((512, 512), Image.BILINEAR)
        image = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        #img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    return vis_parsing_maps(image, parsing, stride=1)


if __name__ == "__main__":
    t0 = time.time()
    img = cv2.imread('2.jpg')
    result = face_parsing(img)
    cv2.imwrite('result.jpg', result)
    print(time.time() - t0)

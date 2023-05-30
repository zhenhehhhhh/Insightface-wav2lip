# ---- 基于高清化人脸的图片序列向大屏视频输出（发布会开场专用） ---- #
import argparse
import cv2
import torch
import os
import shutil
import time
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from PIL import Image
from natsort import natsorted
from BcMatting.dataset import VideoDataset, ZipDataset
from BcMatting.dataset import augmentation as A
from BcMatting.model import MattingBase, MattingRefine
from BcMatting.inference_utils import HomographicAlignment
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 调整



def bcmatting(location, blocation, btgt_bgr_dir, bgr_dir, bbgr_dir, tgt_bgr_dir):
    # ---- step1 拼接基本参数设置 ---- #
    step = 2  # !!!
    save_dir = './temp/'

    # ---- step1 路径及参数计算 ---- #
    # bgr_dir = './temp/parbc.jpg'
    src_dir = './temp/head-imgs/'
    bbgr_img = cv2.imread(bbgr_dir)
    bWidth = bbgr_img.shape[1]
    bHeight = bbgr_img.shape[0]

    # tgt_bgr_dir = './temp/peiac4cu130s.mp4'
    # tgt_bgr_dir = btgt_bgr_dir
    Width = location[1][0] - location[0][0]
    Height = location[1][1] - location[0][1]
    imgs_save_dir = './temp/outputs1/'

    # ---- step2 拼接基本参数设置 ---- #
    # bbgr_dir = './temp/bbc.jpg'
    bsrc_dir = './temp/step1out.mp4'
    # parse_dir = './temp/partmask.jpg'
    blw = blocation[1][0] - blocation[0][0]  # 向大屏拼接图像的宽
    blh = blocation[1][1] - blocation[0][1]  # 向大屏拼接图像的高

    # ---- 模型加载 ---- #
    device = torch.device('cuda')  # 设置GPU
    model = MattingRefine(
        backbone='resnet50',
        backbone_scale=0.25,
        refine_threshold=0.1,
        refine_kernel_size=3)
    model = model.to(device).eval()  # 模型测试模式设置
    model.load_state_dict(torch.load('./checkpoints/pytorch_resnet50.pth', map_location=device), strict=False)  # 加载模型参数
    print('model loaded')


    # ---- step0 加载待抠图像序列名称 ---- #
    flist = []
    for filename in os.listdir(src_dir):
        if filename[-4:] == '.jpg':
            flist.append(filename)
    flist = natsorted(flist)
    totalFrame = len(flist)

    if step == 1:
        # ---- step1 加载背景图片 ---- #
        bgr_img = cv2.imread(bgr_dir)  # 加载背景图片
        bgr_img = cv2.resize(bgr_img, (500, 500))
        bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)  # BGR2RGB
        bgr_img = bgr_img.transpose(2, 0, 1)
        bgr_img = np.array([bgr_img])  
        bgr_img = torch.tensor(bgr_img/255).cuda()
        bgr_img = bgr_img.type(torch.FloatTensor)
        bgr_img = bgr_img.to(device, non_blocking=True)

        # ---- step1 加载目标背景视频 ---- #
        bgr_cap = cv2.VideoCapture(tgt_bgr_dir)
        fps = int(bgr_cap.get(cv2.CAP_PROP_FPS))  # 获取视频帧率
        frame_w = int(bgr_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽度
        frame_h = int(bgr_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 帧高度

        # ---- step1 输出视频保存设置 ---- #
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(save_dir + 'step1out.mp4', fourcc, 25, (bWidth, bHeight))

        # ---- step1 循环推理 ---- #
        count = 0
        with torch.no_grad():
            for item in flist[:]:
                # src_frame = cv2.imread(src_dir + flist[count])
                ret, tgt_img_bc = bgr_cap.read()
                if ret:
                    src_frame = cv2.imread(src_dir + flist[count])
                    print('step1: ' + str(count+1) + ' : ' + str(totalFrame))
                    # ---- src数据格式转换 ---- #
                    src_frame = cv2.resize(src_frame, (500, 500))
                    src_frame = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)  # BGR2RGB
                    src_frame = src_frame.transpose(2, 0, 1)
                    src_frame = np.array([src_frame])
                    src_frame = torch.tensor(src_frame/255).cuda()
                    src_frame = src_frame.type(torch.FloatTensor)
                    src_frame = src_frame.to(device, non_blocking=True)
                    pha, fgr, _, _, err, ref = model(src_frame, bgr_img)
                    
                    # ---- 原视频区域提取 ---- #
                    tgt_img_bc = cv2.resize(tgt_img_bc, (bWidth, bHeight))
                    tgt_bgr_img = tgt_img_bc[location[0][1]:location[1][1], location[0][0]:location[1][0]].copy()

                    # ---- 拼接合成 ---- #
                    tgt_bgr_img = cv2.resize(tgt_bgr_img, (500, 500))
                    fgr = fgr.cpu().numpy()[0].transpose(1,2,0)*255
                    fgr = cv2.cvtColor(fgr, cv2.COLOR_RGB2BGR)
                    pha = pha.cpu().numpy()[0].transpose(1,2,0)
                    pha[pha<0.7] = 0  # Mask阈值拼接策略
                    com = fgr * pha + tgt_bgr_img * (1 - pha)
                    com = cv2.resize(com, (Width, Height))
                    tgt_img_bc[location[0][1]:location[1][1], location[0][0]:location[1][0]] = com
                    cv2.imwrite(imgs_save_dir + str(count) + '.jpg', tgt_img_bc)
                    out.write(tgt_img_bc)
                    cv2.waitKey(1)
                    count = count + 1
                else:
                    # break
                    bgr_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bgr_cap.release()
            out.release()



    # ---- step2 加载需背景抠除的源视频 ---- #
    bsrc_cap = cv2.VideoCapture(bsrc_dir)

    # ---- step2 加载目标背景视频 ---- #
    print('btgt_bgr_dir', btgt_bgr_dir)
    if (btgt_bgr_dir[-3:] == 'mp4') or (btgt_bgr_dir[-3:] == 'MP4') or (btgt_bgr_dir[-3:] == 'avi') or (btgt_bgr_dir[-3:] == 'AVI'):
        bbgr_cap = cv2.VideoCapture(btgt_bgr_dir)
        bbHeight = int(bbgr_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 帧高度
        bbWidth = int(bbgr_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽度
        print('step2 video loaded!')
    elif (btgt_bgr_dir[-3:] == 'jpg') or (btgt_bgr_dir[-3:] == 'peg') or (btgt_bgr_dir[-3:] == 'png') or (btgt_bgr_dir[-3:] == 'gif'):
        btgt_img_bc = cv2.imread(btgt_bgr_dir)
        bbHeight = btgt_img_bc.shape[0]
        bbWidth = btgt_img_bc.shape[1]
        print('step2 image loaded!')

    # ---- 坐标超边界测试 ---- #
    fx1 = max([0, blocation[0][0]])
    fy1 = max([0, blocation[0][1]])
    fx2 = min([blocation[1][0], bbWidth])
    fy2 = min([blocation[1][1], bbHeight])

    cx1 = fx1 - blocation[0][0]
    cy1 = fy1 - blocation[0][1]
    cx2 = fx2 - blocation[0][0]
    cy2 = fy2 - blocation[0][1]

    dx = fx2 - fx1
    dy = fy2 - fy1
    ddx = int(dx/4)*4
    ddy = int(dy/4)*4

    # ---- step2 加载背景图片 ---- #
    bbgr_img = cv2.imread(bbgr_dir)  # 加载背景图片
    bbgr_img = cv2.resize(bbgr_img, (blw, blh))
    bbgr_img = bbgr_img[cy1:cy2,cx1:cx2].copy()  # 边缘裁剪
    bbgr_img = cv2.resize(bbgr_img, (ddx, ddy))  # 调整/4
    bbgr_img = cv2.cvtColor(bbgr_img, cv2.COLOR_BGR2RGB)  # BGR2RGB
    bbgr_img = bbgr_img.transpose(2, 0, 1)
    bbgr_img = np.array([bbgr_img])  
    bbgr_img = torch.tensor(bbgr_img/255).cuda()
    bbgr_img = bbgr_img.type(torch.FloatTensor)
    bbgr_img = bbgr_img.to(device, non_blocking=True)

    # ---- step2 加载区域parse图片 ---- #
    # parse_img = cv2.imread(parse_dir)
    # parse_img = cv2.resize(parse_img, (blw, blh))

    # ---- step2 输出视频保存设置 ---- #
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(save_dir + 'step2out.mp4', fourcc, 25, (bbWidth, bbHeight))

    # ---- step2 循环推理 ---- #
    count = 0
    with torch.no_grad():
        for item in flist[:]:
            ret = False
            if (btgt_bgr_dir[-3:] == 'mp4') or (btgt_bgr_dir[-3:] == 'avi'):
                ret, btgt_img_bc = bbgr_cap.read()
                print('btgt_img_bc.shape', btgt_img_bc.shape)
                print('bbgr_img', bbgr_img.shape)
                if ret:
                    print('????')
                    print('step2: ' + str(count+1) + ' : ' + str(totalFrame))
                    # ---- src数据格式转换 ---- #
                    _, bsrc_frame = bsrc_cap.read()  # 捕获一帧图像
                    bsrc_frame = cv2.resize(bsrc_frame, (blw, blh))
                    bsrc_frame = bsrc_frame[cy1:cy2,cx1:cx2].copy()  # 边缘裁剪
                    bsrc_frame = cv2.resize(bsrc_frame, (ddx, ddy))  # 调整/4
                    bsrc_frame = cv2.cvtColor(bsrc_frame, cv2.COLOR_BGR2RGB)  # BGR2RGB
                    bsrc_frame = bsrc_frame.transpose(2,0,1)
                    bsrc_frame = np.array([bsrc_frame])
                    bsrc_frame = torch.tensor(bsrc_frame/255).cuda()
                    bsrc_frame = bsrc_frame.type(torch.FloatTensor)
                    bsrc_frame = bsrc_frame.to(device, non_blocking=True)
                    print('bsrc_frame', bsrc_frame.shape)
                    pha, fgr, _, _, err, ref = model(bsrc_frame, bbgr_img)
                    
                    # ---- 原视频区域提取 ---- #
                    # btgt_img_bc = cv2.resize(btgt_img_bc, (bbWidth, bbHeight))
                    # btgt_bgr_img = btgt_img_bc[fy1:fy2, fx1:fx2].copy()

                    # ---- 拼接合成 ---- #
                    fgr = fgr.cpu().numpy()[0].transpose(1,2,0)*255
                    fgr = cv2.cvtColor(fgr, cv2.COLOR_RGB2BGR)
                    pha = pha.cpu().numpy()[0].transpose(1,2,0)
                    pha[pha<0.6] = pha[pha<0.6]/2

                    # ---- 区域掩膜置零 ---- #
                    # bc_part = (parse_img[:, :, 0] >= 250) & (parse_img[:, :, 1] == 0) & (parse_img[:, :, 2] == 0)  # 选取蓝色区域
                    # pha[bc_part] = 0  # 

                    btgt_bgr_img = btgt_img_bc[fy1:fy2, fx1:fx2].copy()
                    btgt_bgr_img = cv2.resize(btgt_bgr_img, (ddx, ddy))

                    com = fgr * pha + btgt_bgr_img * (1 - pha)
                    com = cv2.resize(com, (dx, dy))  # 还原至原来的图像尺寸
                    btgt_img_bc[fy1:fy2, fx1:fx2] = com
                    cv2.imwrite(save_dir + 'outputs2/' + str(count) + '.jpg', btgt_img_bc)
                    out.write(btgt_img_bc)
                    cv2.waitKey(1)
                    count = count + 1
                else:
                    bbgr_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                bbgr_cap.release()

            elif (btgt_bgr_dir[-3:] == 'peg') or (btgt_bgr_dir[-3:] == 'jpg') or (btgt_bgr_dir[-3:] == 'png'):
            # elif btgt_bgr_dir[-3:] == 'jpg' or 'peg' or 'png' or 'gif':
                print('step2: ' + str(count+1) + ' : ' + str(len(flist)))
                print('step2!!!!')
                # ---- src数据格式转换 ---- #
                _, bsrc_frame = bsrc_cap.read()  # 捕获一帧图像
                bsrc_frame = cv2.resize(bsrc_frame, (blw, blh))
                bsrc_frame = bsrc_frame[cy1:cy2,cx1:cx2].copy()  # 边缘裁剪
                bsrc_frame = cv2.resize(bsrc_frame, (ddx, ddy))  # 调整/4
                bsrc_frame = cv2.cvtColor(bsrc_frame, cv2.COLOR_BGR2RGB)  # BGR2RGB
                bsrc_frame = bsrc_frame.transpose(2,0,1)
                bsrc_frame = np.array([bsrc_frame])
                bsrc_frame = torch.tensor(bsrc_frame/255).cuda()
                bsrc_frame = bsrc_frame.type(torch.FloatTensor)
                bsrc_frame = bsrc_frame.to(device, non_blocking=True)
                pha, fgr, _, _, err, ref = model(bsrc_frame, bbgr_img)

                # ---- 拼接合成 ---- #
                fgr = fgr.cpu().numpy()[0].transpose(1,2,0)*255
                fgr = cv2.cvtColor(fgr, cv2.COLOR_RGB2BGR)
                pha = pha.cpu().numpy()[0].transpose(1,2,0)
                pha[pha<0.6] = pha[pha<0.6]/2

                btgt_img_bc2 = btgt_img_bc.copy()
                btgt_bgr_img = btgt_img_bc2[fy1:fy2, fx1:fx2].copy()
                btgt_bgr_img = cv2.resize(btgt_bgr_img, (ddx, ddy))

                com = fgr * pha + btgt_bgr_img * (1 - pha)
                com = cv2.resize(com, (dx, dy))  # 还原至原来的图像尺寸
                btgt_img_bc2[fy1:fy2, fx1:fx2] = com
                cv2.imwrite(save_dir + 'outputs2/' + str(count) + '.jpg', btgt_img_bc2)
                out.write(btgt_img_bc2)
                cv2.waitKey(1)
                count = count + 1
        
        bsrc_cap.release()
        out.release()
    print('finish!')

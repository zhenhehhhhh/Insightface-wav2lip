""""单张图片-图片换脸，不通过deepface训练"""
# -*- coding:utf-8 -*-
import numpy as np
import cv2
import time
from xlib import math as lib_math
from xlib.math import Affine2DMat, Affine2DUniMat
from xlib.image import ImageProcessor
from xlib.face import ELandmarks2D, FLandmarks2D
from util import YoloV5Face, InsightFace2D106, Face_Merge, Face_Merge2
import torch

print(torch.cuda.is_available())
import subprocess
from face_parsing.face_parsing2 import face_parsing


def face_process(frame):
    # ---- 人脸检测 ---- #
    frame_raw = frame.copy()
    h, w = frame.shape[0:2]
    rects = Yolov5.extract(frame, threshold=0.5, fixed_window=480)[0]
    if len(rects) == 0:
        return 0
    x1, y1, x2, y2 = int(rects[0][0]), int(rects[0][1]), int(rects[0][2]), int(rects[0][3])

    # ---- 人脸Marker检测 ---- #
    pts = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], ], dtype=np.float32)
    mat = Affine2DMat.umeyama(pts, uni_rect, True)
    g_p = mat.invert().transform_points([(0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0.5)])
    g_c = g_p[4]

    # ---- 全局空间中角点之间的对角向量 ---- #
    tb_diag_vec = lib_math.segment_to_vector(g_p[0], g_p[2]).astype(np.float32)
    bt_diag_vec = lib_math.segment_to_vector(g_p[1], g_p[3]).astype(np.float32)
    mod = lib_math.segment_length(g_p[0], g_p[4]) * coverage
    l_t = np.array([g_c - tb_diag_vec * mod, g_c + bt_diag_vec * mod, g_c + tb_diag_vec * mod],
                   np.float32)
    mat = Affine2DMat.from_3_pairs(l_t,
                                   np.float32(((0, 0), (output_size, 0), (output_size, output_size))))
    uni_mat = Affine2DUniMat.from_3_pairs((l_t / (w, h)).astype(np.float32),
                                          np.float32(((0, 0), (1, 0), (1, 1))))
    face_image = cv2.warpAffine(frame, mat, (output_size, output_size), cv2.INTER_CUBIC)
    _, H, W, _ = ImageProcessor(face_image).get_dims()

    # ---- 人脸LandMarker检测 ---- #
    lmrks = insightface_2d106.extract(face_image)[0]
    lmrks = lmrks[..., 0:2] / (W, H)
    face_ulmrks = FLandmarks2D.create(ELandmarks2D.L106, lmrks)
    face_ulmrks = face_ulmrks.transform(uni_mat, invert=True)

    # ---- Face Aligner ---- #
    if face_ulmrks is None:
        return 0
    face_align_img, uni_mat = face_ulmrks.cut(frame, face_coverage, aligner_resolution,
                                              exclude_moving_parts=exclude_moving_parts,
                                              head_yaw=head_yaw,
                                              x_offset=x_offset,
                                              y_offset=y_offset - 0.08)
    face_align_ulmrks = face_ulmrks.transform(uni_mat)
    face_align_lmrks_mask_img = face_align_ulmrks.get_convexhull_mask(face_align_img.shape[:2],
                                                                      color=(255,),
                                                                      dtype=np.uint8)

    # ---- Face Swap ---- #
    fai_ip = ImageProcessor(face_align_img)
    fai_ip.gaussian_sharpen(sigma=1.0, power=sharpen_power)
    face_align_img = fai_ip.get_image('HWC')

    return frame_raw, face_align_img, face_align_lmrks_mask_img, uni_mat


if __name__ == '__main__':
    # ---- 处理设备选择 ---- #
    print('device: ', YoloV5Face.get_available_devices())
    device_info = YoloV5Face.get_available_devices()[0]

    # ---- wav2lip 参数设置 ---- #
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # ---- 参数设置 ---- #
    coverage = 1.5
    output_size = 192

    aligner_resolution = 224
    face_coverage = 2.2
    exclude_moving_parts = True
    head_yaw = None
    x_offset = 0.0
    y_offset = 0.0
    sharpen_power = 1

    Face_Adjust = False
    median_blur_per = 50
    degrade_bicubic_per = 50

    face_x_offset = 0
    face_y_offset = 0
    face_scale = 1
    do_color_compression = 'rct'
    face_mask_erode = 25
    face_mask_blur = 20

    # ---- 模型加载 ---- #
    Yolov5 = YoloV5Face(device_info)  # 人脸检测模型
    insightface_2d106 = InsightFace2D106(device_info)  # 人脸Marker检测模型

    # ---- 数据加载 ---- #
    uni_rect = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], ], dtype=np.float32)  # Marker检测相关参数

    mode = 3
    # frame_driving = cv2.imread('tools/4_frame/4_1/511.jpg')
    # frame_driving = cv2.resize(frame_driving, (480, 800))
    frame_driving = cv2.imread('tools/chsy/huanlian2.jpg')
    frame_raw_driving, face_swap_img, face_swap_mask_img, uni_mat_driving = face_process(frame_driving)
    if mode == 2:
        face_swap_mask_img = face_parsing(face_swap_img)
    if mode == 3:
        face_swap_mask_img[:aligner_resolution // 2, :] = 0
    # ---- 循环推理 ---- #
    for i in range(36740, 38740):
        with torch.no_grad():
            # source_image
            if mode == 2:
                face_swap_mask_img = face_parsing(face_swap_img)
            if mode == 3:
                face_swap_mask_img[:aligner_resolution // 2, :] = 0
            print(i)
            frame = cv2.imread('tools/chsy2/1/' + str(i+1).zfill(8) + '.jpg')
            frame_raw, face_align_img, face_align_mask_img, uni_mat = face_process(frame)
            # print(face_align_mask_img.shape)
            if mode == 2:
                face_align_mask_img = face_parsing(face_align_img)
            if mode == 3:
                face_align_mask_img[:aligner_resolution // 2, :] = 0

            # ---- Face Merge ---- #
            face_height, face_width = face_align_img.shape[:2]
            frame_height, frame_width = frame.shape[:2]
            aligned_to_source_uni_mat = uni_mat.invert()
            aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_translated(-face_x_offset, -face_y_offset)
            aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_scaled_around_center(face_scale, face_scale)
            aligned_to_source_uni_mat = aligned_to_source_uni_mat.to_exact_mat(face_width, face_height, frame_width,
                                                                               frame_height)

            merged_frame = Face_Merge(frame_raw, aligner_resolution, face_align_img, face_align_mask_img,
                                      face_swap_img, face_swap_mask_img,
                                      aligned_to_source_uni_mat, frame_width, frame_height,
                                      do_color_compression=do_color_compression, face_mask_erode=face_mask_erode,
                                      face_mask_blur=face_mask_blur)

            # ---- 可视化显示 ---- #
            merged_frame = merged_frame * 255
            # res = cv2.resize(merged_frame.astype(np.uint8), (1040, 1732))
            cv2.imwrite('tools/chsy/1/' + str(i+1).zfill(8)  + '.jpg',
                        merged_frame.astype(np.uint8))
            # cv2.waitKey(1)

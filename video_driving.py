"""换脸-视频驱动视频-换脸"""
import time
from typing_extensions import OrderedDict
import matplotlib

matplotlib.use('Agg')
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 在import torch之前生效
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
import depth
from animate import normalize_kp, load_checkpoints
from scipy.spatial import ConvexHull
import pdb
import cv2
from xlib import math as lib_math
from xlib.math import Affine2DMat, Affine2DUniMat
from xlib.image import ImageProcessor
from xlib.face import ELandmarks2D, FLandmarks2D
from util import YoloV5Face, InsightFace2D106, Face_Merge, Face_Merge2

print(torch.cuda.is_available())

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def face_aligner_swap(frame):
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
    l_t = np.array([g_c - tb_diag_vec * mod, g_c + bt_diag_vec * mod, g_c + tb_diag_vec * mod], np.float32)
    mat = Affine2DMat.from_3_pairs(l_t, np.float32(((0, 0), (output_size, 0), (output_size, output_size))))
    uni_mat = Affine2DUniMat.from_3_pairs((l_t / (w, h)).astype(np.float32), np.float32(((0, 0), (1, 0), (1, 1))))
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
                                              head_yaw=head_yaw, x_offset=x_offset, y_offset=y_offset - 0.08)
    face_align_ulmrks = face_ulmrks.transform(uni_mat)
    face_align_lmrks_mask_img = face_align_ulmrks.get_convexhull_mask(face_align_img.shape[:2], color=(255,),
                                                                      dtype=np.uint8)

    # ---- Face Swap ---- #
    fai_ip = ImageProcessor(face_align_img)
    fai_ip.gaussian_sharpen(sigma=1.0, power=sharpen_power)
    face_align_img = fai_ip.get_image('HWC')

    return frame_raw, face_align_img, face_align_lmrks_mask_img, uni_mat, (x1, y1, x2, y2)


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True,
                   cpu=False):
    # 驱动视频数据加载
    cap_driving = cv2.VideoCapture(driving_video)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
    fps = cap_driving.get(cv2.CAP_PROP_FPS)  # 帧数

    count = -1
    count_start = count
    with torch.no_grad():
        while (cap_driving.isOpened()):
            t0 = time.time()
            frame_source = cv2.imread(
                '/u01/zhengyang/projects/virtual-human_video_driving/tools/imgs4/' + str(count) + '.jpg')  # 源图像/视频
            height_source, width_source = frame_source.shape[0], frame_source.shape[1]
            # frame_driving = cv2.imread('tools/driving_videos/lihong/' + str(count + 22) + '.jpg')  # 驱动视频
            # frame_driving = cv2.imread('tools/driving_videos/xinwen2/' + str(count + 125) + '.jpg')  # 驱动视频
            # frame_driving = cv2.imread('tools/driving_videos/xinwen3/' + str(count + 2) + '.jpg')  # 驱动视频
            frame_driving = cv2.imread(
                '/u01/zhengyang/projects/virtual-human_video_driving/tools/driving_videos/xinwen4/' + str(
                    count + 2) + '.jpg')  # 驱动视频
            # ret, frame_driving = cap_driving.read()  # 驱动视频
            # if not ret:
            #     break
            # 视频-视频:  人脸对齐-换脸
            frame_raw, face_align_img, face_align_mask_img, uni_mat, face_bbox = face_aligner_swap(frame_source)
            frame_raw_driving, face_align_img_driving, face_align_mask_img_driving, uni_mat_driving, _ = face_aligner_swap(
                frame_driving)
            # ---- Face Adjust ---- #
            if Face_Adjust:
                frame_image_ip = ImageProcessor(face_align_img_driving)
                frame_image_ip.median_blur(5, opacity=median_blur_per / 100.0)
                frame_image_ip.reresize(degrade_bicubic_per / 100.0, interpolation=ImageProcessor.Interpolation.CUBIC)
                face_align_img_driving = frame_image_ip.get_image('HWC')

            # ---- Face Merge ---- #
            face_height, face_width = face_align_img.shape[:2]
            aligned_to_source_uni_mat = uni_mat.invert()
            aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_translated(-face_x_offset, -face_y_offset)
            aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_scaled_around_center(face_scale, face_scale)
            aligned_to_source_uni_mat = aligned_to_source_uni_mat.to_exact_mat(face_width, face_height, width_source,
                                                                               height_source)
            frame_merged = Face_Merge(frame_raw, aligner_resolution, face_align_img, face_align_mask_img,
                                      face_align_img_driving, face_align_mask_img_driving,
                                      aligned_to_source_uni_mat, width_source, height_source,
                                      do_color_compression=do_color_compression, face_mask_erode=face_mask_erode,
                                      face_mask_blur=face_mask_blur)
            frame_merged = (frame_merged * 255).astype(np.uint8)

            if count == count_start:  # 初始帧源图片数据加载,DAGAN_source
                source_image0 = frame_source.copy()
                face_center = (face_bbox[2] + face_bbox[0]) / 2, (face_bbox[1] + face_bbox[3]) / 2
                l = (face_bbox[3] - face_bbox[1]) * 0.85
                crop_y1, crop_y2, crop_x1, crop_x2 = \
                    int(max(0, face_center[1] - l)), int(min(height_source, face_center[1] + l)), \
                    int(max(0, face_center[0] - l)), int(min(width_source, face_center[0] + l))

                source_image = source_image0[crop_y1:crop_y2, crop_x1:crop_x2]
                source_image = cv2.cvtColor(source_image.copy(), cv2.COLOR_BGR2RGB)
                source_image = resize(source_image, (256, 256))[..., :3]  # (256,256,3)
                source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                if not cpu:
                    source = source.cuda()
                outputs = depth_decoder(depth_encoder(source))
                depth_source = outputs[("disp", 0)]  # [1, 1, 256, 256])
                source_kp = torch.cat((source, depth_source), 1)  # [1, 4, 256, 256])
                kp_source = kp_detector(source_kp)

                height_driving, width_driving = frame_driving.shape[0], frame_driving.shape[1]
                out_video = cv2.VideoWriter('result.mp4', fourcc, fps, (width_source * 2, height_source))  # 写入视频
                out_video3 = cv2.VideoWriter('result3.mp4', fourcc, fps, (width_source, height_source // 2))  # 写入视频

            frame_driving0 = cv2.resize(frame_driving,
                                        (width_source, int(width_source / width_driving * height_driving)))
            frame_driving1 = np.zeros((height_source, width_source, 3), np.uint8)
            frame_driving1[0:int(width_source / width_driving * height_driving), 0:width_source] = frame_driving0

            # DAGAN_driving
            frame = frame_merged[crop_y1:crop_y2, crop_x1:crop_x2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            driving_frame = resize(frame, (256, 256))[..., :3]

            driving_frame = torch.tensor(driving_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                driving_frame = driving_frame.cuda()

            outputs = depth_decoder(depth_encoder(driving_frame))
            depth_driving = outputs[("disp", 0)]
            driving_kp = torch.cat((driving_frame, depth_driving), 1)
            kp_driving = kp_detector(driving_kp)

            if count == count_start:
                kp_driving_initial = kp_driving

            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm, source_depth=depth_source,
                            driving_depth=depth_driving)

            prediction = cv2.cvtColor(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0],
                                      cv2.COLOR_RGB2BGR)  # [256,256]
            # predictions = frame_merged.copy()[..., :3]  # 拼接驱动图像
            # predictions[crop_y1:crop_y2, crop_x1:crop_x2] = cv2.resize(img_as_ubyte(prediction),
            #                                                            (crop_x2 - crop_x1, crop_y2 - crop_y1))
            # 二次换脸
            frame_raw_driving, face_align_img_driving, face_align_mask_img_driving, uni_mat_driving, _ = face_aligner_swap(
                img_as_ubyte(prediction))
            # ---- Face Adjust ---- #
            if Face_Adjust:
                frame_image_ip = ImageProcessor(face_align_img_driving)
                frame_image_ip.median_blur(5, opacity=median_blur_per / 100.0)
                frame_image_ip.reresize(degrade_bicubic_per / 100.0, interpolation=ImageProcessor.Interpolation.CUBIC)
                face_align_img_driving = frame_image_ip.get_image('HWC')

            # ---- Face Merge ---- #
            face_height, face_width = face_align_img.shape[:2]
            aligned_to_source_uni_mat = uni_mat.invert()
            aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_translated(-face_x_offset, -face_y_offset)
            aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_scaled_around_center(face_scale, face_scale)
            aligned_to_source_uni_mat = aligned_to_source_uni_mat.to_exact_mat(face_width, face_height, width_source,
                                                                               height_source)
            frame_merged2 = Face_Merge(frame_raw, aligner_resolution, face_align_img, face_align_mask_img,
                                       face_align_img_driving, face_align_mask_img_driving,
                                       aligned_to_source_uni_mat, width_source, height_source,
                                       do_color_compression=do_color_compression, face_mask_erode=face_mask_erode,
                                       face_mask_blur=face_mask_blur)
            frame_merged2 = (frame_merged2 * 255).astype(np.uint8)

            # merge = np.concatenate((img_as_ubyte(source_image0), img_as_ubyte(frame_driving1),
            #                         img_as_ubyte(frame_merged), img_as_ubyte(frame_merged2)), 1)
            merge = np.concatenate((img_as_ubyte(frame_driving1), img_as_ubyte(frame_merged2)), 1)

            # cv2.imshow('merge', merge)
            # cv2.waitKey(1)
            # cv2.imwrite('image_video/results/' + str(count) + '.jpg', merge)
            out_video.write(merge)
            out_video3.write(frame_merged2[:height_source // 2, :])

            count += 1
            t1 = time.time()
            print('Frame{}_FPS: {:.2f}'.format(count - 1, 1 / (t1 - t0)))

        cap_driving.release()
        out_video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print('device: ', YoloV5Face.get_available_devices())
    device_info = YoloV5Face.get_available_devices()[0]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-adv-256.yaml', help="path to config")
    # DaGAN_vox_adv_256.pth.tar, SPADE_DaGAN_vox_adv_256.pth.tar
    parser.add_argument("--checkpoint", default='weights/SPADE_DaGAN_vox_adv_256.pth.tar',
                        help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='image_video/images/pei.jpg', help="path to source image")
    parser.add_argument("--driving_video",
                        default='/u01/zhengyang/projects/virtual-human_video_driving/tools/driving_videos/xinwen1.mp4',
                        help="path to driving video")  # ("rtsp://10.32.154.165:8554/live")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")

    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")
    # DepthAwareGenerator, SPADEDepthAwareGenerator
    parser.add_argument("--generator", type=str, default='SPADEDepthAwareGenerator')
    parser.add_argument("--kp_num", type=int, default='15')

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.set_defaults(relative=True)
    parser.set_defaults(adapt_scale=True)

    opt = parser.parse_args()

    depth_encoder = depth.ResnetEncoder(18, False)
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
    loaded_dict_enc = torch.load('weights/encoder.pth')
    loaded_dict_dec = torch.load('weights/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    if not opt.cpu:
        depth_encoder.cuda()
        depth_decoder.cuda()
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint,
                                              generator=opt.generator, kp_num=opt.kp_num, cpu=opt.cpu)

    # 人脸对齐-换脸 ---- 参数设置 ---- 
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
    face_mask_erode = 5
    face_mask_blur = 10

    # ---- 模型加载 ---- #
    Yolov5 = YoloV5Face(device_info)  # 人脸检测模型
    insightface_2d106 = InsightFace2D106(device_info)  # 人脸Marker检测模型

    # ---- 数据加载 ---- #
    uni_rect = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], ], dtype=np.float32)  # Marker检测相关参数

    start_time = time.time()
    make_animation(opt.source_image, opt.driving_video, generator,
                   kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    print('time', time.time() - start_time)

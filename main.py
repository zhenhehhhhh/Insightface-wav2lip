from typing_extensions import OrderedDict
import matplotlib

matplotlib.use('Agg')
import os, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import subprocess
from xftts import kxtts
from wavutil import parser_define, wav2lip_load_model, aud2feature
from re import T
import numpy as np
import cv2
import time
from xlib import math as lib_math
from xlib.math import Affine2DMat, Affine2DUniMat
from xlib.image import ImageProcessor
from xlib.face import ELandmarks2D, FLandmarks2D
from util import YoloV5Face, InsightFace2D106, Face_Merge
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from skimage.transform import resize
from skimage import img_as_ubyte
import depth
from animate import normalize_kp, load_checkpoints
from dytts import dytts_fun
from face_parsing.face_parsing2 import face_parsing
from face_recognition_update import FaceRecognition
import contextlib
import wave

def DAGAN_load_arguments():
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-adv-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='weights/SPADE_DaGAN_vox_adv_256.pth.tar', help="checkpoint")
    parser.add_argument("--generator", type=str, default='SPADEDepthAwareGenerator')
    opt = parser.parse_args()

    depth_encoder = depth.ResnetEncoder(18, False)
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
    loaded_dict_enc = torch.load('weights/encoder.pth', map_location=torch.device('cpu'))
    loaded_dict_dec = torch.load('weights/depth.pth', map_location=torch.device('cpu'))
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval().to(device)
    depth_decoder.eval().to(device)
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, generator=opt.generator, kp_num=15, cpu=True)
    return depth_encoder, depth_decoder, generator, kp_detector

def audio_synthesis(timbre, text, name):
    audio_path = 'outputs/audio'
    kxtts(text=text, timbre=timbre, audio_path=audio_path, aud_speed=aud_speed, ran_str=name)
    audio = audio_path + name + '_encryption.wav'

    return audio

def face_aligner_swap(frame, rects):
    # 人脸检测
    frame_raw = frame.copy()
    h, w = frame.shape[0:2]
    # detection = face_recognition.detect(frame)
    # if len(detection) == 0:
    #     return [None, None, None, None, None]
    # bbox = detection[0]['bbox']
    # rects = [(bbox[0], bbox[1], bbox[2], bbox[3])]
    # # rects = Yolov5.extract(frame, threshold=0.5, fixed_window=480)[0]

    if len(rects) == 0:
        return [None, None, None, None, None]
    x1, y1, x2, y2 = int(rects[0][0]), int(rects[0][1]), int(rects[0][2]), int(rects[0][3])
    
    # 人脸 Marker 检测
    pts = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], ], dtype=np.float32)
    mat = Affine2DMat.umeyama(pts, uni_rect, True)
    g_p = mat.invert().transform_points([(0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0.5)])
    g_c = g_p[4]

    # 全局空间中角点之间的对角向量
    tb_diag_vec = lib_math.segment_to_vector(g_p[0], g_p[2]).astype(np.float32)
    bt_diag_vec = lib_math.segment_to_vector(g_p[1], g_p[3]).astype(np.float32)
    mod = lib_math.segment_length(g_p[0], g_p[4]) * coverage
    l_t = np.array([g_c - tb_diag_vec * mod, g_c + bt_diag_vec * mod, g_c + tb_diag_vec * mod], np.float32)
    mat = Affine2DMat.from_3_pairs(l_t, np.float32(((0, 0), (output_size, 0), (output_size, output_size))))
    uni_mat = Affine2DUniMat.from_3_pairs((l_t / (w, h)).astype(np.float32), np.float32(((0, 0), (1, 0), (1, 1))))
    face_image = cv2.warpAffine(frame, mat, (output_size, output_size), cv2.INTER_CUBIC)
    _, H, W, _ = ImageProcessor(face_image).get_dims()

    # 人脸 LandMaker 检测
    lmrks = insightface_2d106.extract(face_image)[0]
    lmrks = lmrks[..., 0:2] / (W, H)
    face_ulmrks = FLandmarks2D.create(ELandmarks2D.L106, lmrks)
    face_ulmrks = face_ulmrks.transform(uni_mat, invert=True)

    # Face Aligner
    if face_ulmrks is None:
        return [None, None, None, None, None]
    face_align_img, uni_mat = face_ulmrks.cut(frame, face_coverage, aligner_resolution, \
                                              exclude_moving_parts=exclude_moving_parts,\
                                              head_yaw=head_yaw, x_offset=x_offset, y_offset=y_offset - 0.08)

    # Face Swap
    fai_ip = ImageProcessor(face_align_img)
    fai_ip.gaussian_sharpen(sigma=1.0, power=sharpen_power)
    face_align_img = fai_ip.get_image('HWC')

    return frame_raw, face_align_img, _, uni_mat, (x1, y1, x2, y2)

# 获得音频长度
def get_audio_length(file_path):
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        audio_fs = f.getnframes()
        audio_rate = f.getframerate()
        wav_length = audio_fs / float(audio_rate) # print("wav length: {}".format(wav_length)) 
    return wav_length

# 音频处理
def pro_audios(talking_before, stop):
    for people in talking_before:
        command = 'ffmpeg -y -i outputs/audio{}.wav -ss 00:00:00 -t 00:00:{} outputs/seg.wav'.format(people, stop)
        subprocess.call(command, shell=True)

        command = 'ffmpeg -y -i outputs/audio{}.wav -ss 00:00:{} -t 00:10:0 outputs/audio{}_temp.wav'.format(people, stop, people)
        subprocess.call(command, shell=True)

        os.remove('outputs/audio{}.wav'.format(people))
        shutil.move('outputs/audio{}_temp.wav'.format(people), 'outputs/audio{}.wav'.format(people))

        if os.path.exists('outputs/combine.wav'):
            if os.path.exists('outputs/temp.txt'):
                os.remove('outputs/temp.txt')

            f = open("outputs/temp.txt", "a+", encoding='utf-8')
            f.write("file 'combine.wav'\nfile 'seg.wav'")
            f.close()

            command = "ffmpeg -y -f concat -safe 0 -i outputs/temp.txt -c copy outputs/combine.wav"
            subprocess.call(command, shell=True)
        else:
            shutil.move('outputs/seg.wav', 'outputs/combine.wav')

def init_project():
    shutil.rmtree("outputs")
    shutil.rmtree("face_db")
    os.mkdir("outputs")
    os.mkdir("face_db")

# def error_pro(frame, ):

if __name__ == '__main__':
    init_project()
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ', YoloV5Face.get_available_devices())
    device_info = YoloV5Face.get_available_devices()[0]

    # DAGAN 模型及参数加载
    depth_encoder, depth_decoder, generator, kp_detector = DAGAN_load_arguments()

    # wav2lip 参数设置
    fps = 25
    frames = 0
    save_vid = True
    text = []
    timbre = ['xiaoyan', 'aisjiuxu', 'aisxping', 'aisjinger', 'aisbabyxu']
    aud_speed = 70
    audio = []
    video_path = 'tools/chsy/peoples.mp4'
    model_path = 'modelhub/wav2lip_cn.pth'

    # 参数设置
    coverage = 1.5
    output_size = 192
    aligner_resolution = 224
    face_coverage = 2.2
    exclude_moving_parts = True
    head_yaw = None 
    x_offset = 0.0
    y_offset = 0.0
    sharpen_power = 1
    median_blur_per = 50
    degrade_bicubic_per = 50
    face_x_offset = 0
    face_y_offset =  0
    face_scale = 1
    do_color_compression = 'rct'
    face_mask_erode = 10
    face_mask_blur = 20

    args = parser_define()

    # 模型加载
    Yolov5 = YoloV5Face(device_info)
    wav_model = wav2lip_load_model(model_path, device)
    insightface_2d106 = InsightFace2D106(device_info)
    face_recognition = FaceRecognition()

    # 数据加载
    uni_rect = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], ], dtype=np.float32)  # Marker检测相关参数
    if len(video_path) > 0:
        cap = cv2.VideoCapture(video_path) # 打开相机
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 帧宽度
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 帧高度
        fps = cap.get(cv2.CAP_PROP_FPS) # 获取视频帧率 
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # 获取视频的总帧数
    
    # 文本加载
    f = open('text.txt', 'r+', encoding='utf-8')
    texts = f.readlines()

    # 音频合成
    record = np.zeros(10, dtype=int)   # 用于在做 wav2lip 音频切换时记录音频播放到第几帧，初始设置视频最多有10个说话人
    mel_chunk = []          # 存储每段音频的特征
    audio_len = []          # 存储每段音频的长度
    for i in range(len(texts)):
        audio_encryption = audio_synthesis(timbre[i%len(timbre)], texts[i], '{}'.format(i)) # 科大讯飞开放平台合成的语音有加密处理
        command = 'ffmpeg -y -i {} outputs/audio{}.wav'.format(audio_encryption, i) # 音频解密
        subprocess.call(command, shell=True)
        os.remove(audio_encryption)
        audio.append("outputs/audio{}.wav".format(i))
        audio_len.append(get_audio_length(audio[i]))
        mel_chunk.append(aud2feature(args, fps, audio[i]))
    
    # 循环推理
    same_people = True  # 上一帧说话人和正在处理的这一帧的说话人是不是同一个
    talking = []        # 正在处理的这一帧的说话人
    talking_before = [] # 上一帧画面的说话人
    peoples = {}        # 视频中出现的说话人
    with torch.no_grad():
        t_begin = time.time() # 循环推理的开始时间
        for num in range(int(frames)):
            t_1f_begin = time.time() # 每一帧视频做推理的开始时间
            ret, frame = cap.read()  # 从相机读取视频帧数据
            if num == 47:
                print("stop")
            if frame is not None:
                frame_raw = frame.copy()
                if num == 0:
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    frame_w, frame_h = frame.shape[1], frame.shape[0]
                    out_video2 = cv2.VideoWriter('result2.mp4', fourcc, fps, (frame_w, frame_h))
        
                # ----- 人脸识别 ----- #
                result = face_recognition.register(frame)
                if result == '图片检测不到人脸': # 检测不到人脸则不做视频帧推理，直接拼接视频帧
                    out_video2.write(frame)
                    continue
                talking_before = talking
                talking = face_recognition.recognition(frame)
                if talking == talking_before:
                    same_people = True
                else:
                    same_people = False

                
                detection = face_recognition.detect(frame) # 人脸检测
                rects = []
                # 对画面中的每个人做 wav2lip
                i = 0
                while record[i] >= len(mel_chunk[int(talking[i])]):
                    i += 1
                if i >= len(talking):
                    break
                bbox = detection[i]['bbox']
                rects.append((bbox[0], bbox[1], bbox[2], bbox[3]))

                # ----- wav2lip ----- #
                bbox_x, bbox_y = (rects[0][0] + rects[0][2]) / 2, (rects[0][1] + rects[0][3]) / 2
                bbox_w, bbox_h = (rects[0][2] - rects[0][0]) / 2, (rects[0][3] - rects[0][1]) / 2 
                rate = 0.1
                x1, y1, x2, y2 = int(bbox_x - (1 + rate) * bbox_w), int(bbox_y - (1 + rate) * bbox_h), \
                                int(bbox_x + (1 + rate) * bbox_w), int(bbox_y + (1 + rate) * bbox_h)
                face_img_batch = frame[y1:y2, x1:x2].copy()
                cv2.imwrite("result/face_img_batch.jpg", face_img_batch)
                face_img_batch = cv2.resize(face_img_batch, (args.img_size, args.img_size))

                face_img_masked = face_img_batch.copy()
                face_img_masked[args.img_size // 2:, :] = 0
                face_img_batch = np.concatenate((face_img_masked, face_img_batch), axis=2) / 255.
                face_img_batch = [face_img_batch]
                if len(mel_chunk) < int(talking[i]):
                    print('Error.... The number of audio is not enough !')
                    out_video2.write(frame)
                    continue
                mel_batch = mel_chunk[int(talking[i])][record[i]]
                record[i] += 1
                mel_batch = np.reshape(mel_batch, [1, mel_batch.shape[0], mel_batch.shape[1], 1])

                face_img_batch = torch.FloatTensor(np.transpose(face_img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
                pred = wav_model(mel_batch, face_img_batch)
                pred = pred[0].cpu().numpy().transpose(1, 2, 0) * 255.

                dx, dy = x2 - x1, y2 - y1
                p = cv2.resize(pred.astype(np.uint8), (dx, dy))
                frame[y1 + int(dy * 1 / 2):y2 - int(dy / 12), x1 + int(dx / 10):x2 - int(dx / 10)] = \
                p[int(dy * 1 / 2):-int(dy / 12), int(dx / 10):-int(dx / 10)]  # 关键点！

                _, face_align_img, _, uni_mat, face_bbox = face_aligner_swap(frame_raw, rects)
                face_align_mask_img = face_parsing(face_align_img)

                # ----- DAGAN 人脸关键点，深度分析 ----- #
                if not same_people or num == frames - 1:
                    stop = 0
                    stop = (record[i] / len(mel_chunk[i])) * audio_len[i]
                    if stop != 0 and num != 0: 
                        pro_audios(talking_before, stop)
                    
                    # 初始帧源图片数据加载， DAGAN_source
                    source_image0 = frame_raw.copy()
                    face_center = (face_bbox[2] + face_bbox[0]) / 2, (face_bbox[1] + face_bbox[3]) / 2
                    l = (face_bbox[3] - face_bbox[1]) * 0.85
                    crop_y1, crop_y2, crop_x1, crop_x2 = \
                        int(max(0, face_center[1] - l)), int(min(frame_h, face_center[1] + l)), \
                        int(max(0, face_center[0] - l)), int(min(frame_w, face_center[0] + l))

                    source_image = source_image0[crop_y1:crop_y2, crop_x1:crop_x2]
                    cv2.imwrite("result/source_image.jpg", source_image)
                    source_image = cv2.cvtColor(source_image.copy(), cv2.COLOR_BGR2RGB)
                    source_image = resize(source_image, (256, 256))[..., :3]  # (256,256,3)
                    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
                    outputs = depth_decoder(depth_encoder(source))
                    depth_source = outputs[("disp", 0)]
                    source_kp = torch.cat((source, depth_source), 1)
                    kp_source = kp_detector(source_kp)

                # ----- DAGAN ----- #
                if not same_people:
                    frame_d = frame_raw[crop_y1:crop_y2, crop_x1:crop_x2]
                else:
                    frame_d = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                cv2.imwrite("result/initial/frame_d_raw{}.jpg".format(num), frame_raw[crop_y1:crop_y2, crop_x1:crop_x2])
                cv2.imwrite("result/process/frame_d{}.jpg".format(num), frame[crop_y1:crop_y2, crop_x1:crop_x2])

                frame_d = cv2.resize(frame_d, (256, 256))
                frame_d = cv2.cvtColor(frame_d, cv2.COLOR_BGR2RGB)
                driving_frame = torch.tensor(frame_d[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255

                outputs = depth_decoder(depth_encoder(driving_frame))
                depth_driving = outputs[("disp", 0)]
                driving_kp = torch.cat((driving_frame, depth_driving), 1)
                kp_driving = kp_detector(driving_kp)

                if not same_people:
                    kp_driving_initial = kp_driving
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial, use_relative_movement=True,
                                    use_relative_jacobian=True, adapt_movement_scale=True)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm, source_depth=depth_source, driving_depth=depth_driving)
                prediction = cv2.cvtColor(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0], cv2.COLOR_RGB2BGR)  # [256,256]
                frame_raw_driving, face_align_img_driving, _, uni_mat_driving, _ = face_aligner_swap(img_as_ubyte(prediction), rects)

                if frame_raw_driving is None:
                    out_video2.write(frame)
                    continue
                face_align_mask_img_driving = face_parsing(face_align_img_driving)

                # ---- Face Merge ---- #
                face_height, face_width = face_align_img.shape[:2]
                aligned_to_source_uni_mat = uni_mat.invert()
                aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_translated(-face_x_offset, -face_y_offset)
                aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_scaled_around_center(face_scale, face_scale)
                aligned_to_source_uni_mat = aligned_to_source_uni_mat.to_exact_mat(face_width, face_height, frame_w, frame_h)
                frame_merged = Face_Merge(frame_raw, aligner_resolution, face_align_img, face_align_mask_img,
                                        face_align_img_driving, face_align_mask_img_driving, aligned_to_source_uni_mat, frame_w, frame_h,
                                        do_color_compression=do_color_compression, face_mask_erode=face_mask_erode, face_mask_blur=face_mask_blur)
                frame_merged = (frame_merged * 255).astype(np.uint8)

                # ----- 可视化显示 ----- #
                if save_vid == True:
                    out_video2.write(frame_merged)
                t_1f_end = time.time()
            else:
                break

            print('Frame{}_FPS: {:.2f}'.format(num, 1 / (t_1f_end - t_1f_begin + 0.1)))
            
        print(time.time() - t_begin)
        if len(video_path) > 0:
            cap.release()
        
    if save_vid == True:
        out_video2.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 -vcodec libx264 {}'.format('outputs/combine.wav', 'result2.mp4', 'result2_audio.mp4')
    subprocess.call(command, shell=True)

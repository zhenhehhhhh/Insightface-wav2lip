import os
import cv2
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from util import YoloV5Face, InsightFace2D106
from wavutil import parser_define, wav2lip_load_model, aud2feature
from xftts import kxtts
from animate import dagan_parser, load_checkpoints, normalize_kp
import depth
from skimage import img_as_ubyte
import ctypes
from ctypes import c_buffer, c_char, c_char_p, c_int, c_void_p, c_wchar_p, cdll 
from threading import Thread
from queue import Queue
from flasgger import Swagger
from flask import Flask, request
import subprocess
import requests


app = Flask(__name__)
Swagger(app)


class Wav2lip_Producer(Thread):
    def __init__(self, name, mel_chunks, imgs_Queue, wav_queue):
        Thread.__init__(self, name=name)
        self.mel_chunks = mel_chunks
        self.imgs_Queue = imgs_Queue
        self.wav_queue = wav_queue

    def run(self):
        with torch.no_grad():
            for num in range(len(self.mel_chunks)):
                t0 = time.time()
                frame = self.imgs_Queue.get()[0]
                crop_frame_raw = frame.copy()
                crop_frame = frame[crop_area[1]:crop_area[3],crop_area[0]:crop_area[2]].copy()
                
                rects = Yolov5.extract(crop_frame, threshold=0.5, fixed_window=480)[0]
                if len(rects) == 0:
                  continue
                x1, y1, x2, y2 = int(rects[0][0]), int(rects[0][1]), int(rects[0][2]), int(rects[0][3])
                face_img_batch = crop_frame[y1:y2,x1:x2].copy()
                face_img_batch = cv2.resize(face_img_batch, (args.img_size, args.img_size))
                face_img_masked = face_img_batch.copy()
                face_img_masked[args.img_size//2:, :] = 0
                face_img_batch = np.concatenate((face_img_masked, face_img_batch), axis=2) / 255.
                face_img_batch = [face_img_batch]
                mel_batch = self.mel_chunks[num]
                mel_batch = np.reshape(mel_batch, [1, mel_batch.shape[0], mel_batch.shape[1], 1])
                face_img_batch = torch.FloatTensor(np.transpose(face_img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
                pred = wav_model(mel_batch, face_img_batch)
                pred = pred[0].cpu().numpy().transpose(1, 2, 0) * 255.
                dx = x2-x1
                dy = y2-y1
                p = cv2.resize(pred.astype(np.uint8), (dx, dy))
                crop_frame[y1+int(dy*1/2):y2, x1:x2] = p[int(dy*1/2):,:]  # 区域拼接设置
                wav_data = [crop_frame, crop_frame_raw, t0]
                self.wav_queue.put(wav_data)
            print("%s finished!" % self.getName())


class Depth_Producer(Thread):
    def __init__(self, name, mel_chunks, wav_queue, depth_queue):
        Thread.__init__(self, name=name)
        self.mel_chunks = mel_chunks
        self.wav_queue = wav_queue
        self.depth_queue = depth_queue
    def run(self):
        with torch.no_grad():
            for num in range(len(self.mel_chunks)):
                crop_frame, crop_frame_raw, t0 = self.wav_queue.get()
                crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
                driving_frame = torch.tensor(crop_frame[np.newaxis].astype(np.float32)).cuda().permute(0, 3, 1, 2)/255
                outputs = depth_decoder(depth_encoder(driving_frame))
                depth_driving = outputs[("disp", 0)]
                driving_kp = torch.cat((driving_frame, depth_driving), 1)
                kp_driving = kp_detector(driving_kp)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving, kp_driving_initial=kp_driving_initial, use_relative_movement=True, use_relative_jacobian=True, adapt_movement_scale=True)
                kp_norm_value = kp_norm['value'].cpu()
                kp_norm_jacobian = kp_norm['jacobian'].cpu()
                depth_driving_cpu = depth_driving.cpu()
                depth_data = [kp_norm_value, kp_norm_jacobian, depth_driving_cpu, crop_frame_raw, t0]
                self.depth_queue.put(depth_data)
            print("%s finished!" % self.getName())


class DaGAN_Producer(Thread):
    def __init__(self, name, mel_chunks, depth_queue, dagan_queue):
        Thread.__init__(self, name=name)
        self.mel_chunks = mel_chunks
        self.depth_queue = depth_queue
        self.dagan_queue = dagan_queue
    def run(self):
        with torch.no_grad():
            for num in range(len(self.mel_chunks)):
                kp_norm_value, kp_norm_jacobian, depth_driving_cpu, crop_frame_raw, t0= self.depth_queue.get()
                kp_norm = {'value':kp_norm_value.cuda(), 'jacobian':kp_norm_jacobian.cuda()}
                depth_driving = depth_driving_cpu.cuda()
                out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm, source_depth=depth_source, driving_depth=depth_driving)
                predictions = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                dagan_data = [predictions, crop_frame_raw, t0]
                self.dagan_queue.put(dagan_data)
            print("%s finished!" % self.getName())

class Swap_Producer(Thread):
    def __init__(self, name, mel_chunks, dagan_queue, task_id, callback_url):
        Thread.__init__(self, name=name)
        self.mel_chunks = mel_chunks
        self.dagan_queue = dagan_queue
        self.task_id = task_id
        self.callback_url = callback_url
    def run(self):
        global flag_glob
        flag_glob = True
        res = {"taskId":self.task_id, "triggerFlag":1}
        backinfo = requests.post(self.callback_url, json=res) 
        with torch.no_grad():
            for num in range(len(self.mel_chunks)):
                predictions, crop_frame_raw, t0 = self.dagan_queue.get()
                predictions = cv2.cvtColor(img_as_ubyte(predictions), cv2.COLOR_RGB2BGR)
                
                # ---- LandMarker检测并获得包围区域 ---- #
                lmrks = insightface_2d106.extract(predictions)[0]
                lmrks = (lmrks*192/256).astype(np.int32)
                mask = np.zeros((256,256,1), dtype=np.uint8)
                color=(255,)
                cv2.fillConvexPoly(mask, cv2.convexHull(lmrks), color)

                # ---- Erode与高斯模糊 ---- #
                erode = 10
                blur = 10
                erode, blur = int(erode), int(blur)
                el = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
                iterations = max(1,erode//2)
                mask = cv2.erode(mask, el, iterations = iterations)
                sigma = blur * 0.125 * 2
                mask = cv2.GaussianBlur(mask, (0, 0), sigma)

                # ---- 换脸操作 ---- #
                frame = crop_frame_raw
                crop_frame_raw = crop_frame_raw[crop_area[1]:crop_area[3],crop_area[0]:crop_area[2]]
                crop_frame_raw = crop_frame_raw.astype(np.float64)
                mask = mask.astype(np.float64)/255
                mask = np.expand_dims(mask, axis=2)
                mask =np.tile(mask, (1,1,3))
                dst_img = crop_frame_raw*(1-mask) + predictions*mask
                frame[crop_area[1]:crop_area[3],crop_area[0]:crop_area[2]] = dst_img
                t1 = time.time()
                # print('T:', t1-t0)
                send_Queue.put(frame.astype(np.uint8))
            flag_glob = False

            print("%s finished!" % self.getName())


# ---- 读取原始视频数据 ---- #
def raw_vid(imgs_Queue, frame_list):
    global queue_len
    
    flip_flag = True
    item = 0
    while True:
        if flip_flag == True:
            item = item + 1
            if item == len(frame_list):
                item = len(frame_list)-2
                flip_flag = False
        elif flip_flag == False:
            item = item - 1
            if item == 0:
                item = 2
                flip_flag = True

        frame = frame_list[item]
        imgs_Queue.put(frame)
        if queue_len>=10:
            time.sleep(0.1)
        elif queue_len<=8:
            time.sleep(0.01)
        else:
            time.sleep(0.02)


def send_vid(imgs_Queue, send_Queue):
    global queue_len
    global flag_glob
    os.remove('rtmp_img_sdk.log')
    print('log deleted')
    while True:
        if flag_glob != True:
            img = imgs_Queue.get()[1]
            send_Queue.put(img)
            img = send_Queue.get()
            frame = np.asarray(img, dtype=np.uint8)
            frame = frame.ctypes.data_as(ctypes.c_char_p)
            rtmpstr.RtmpImgSdk_SendRgb24.argtypes = [c_void_p, c_char_p, c_int, c_int]
            rtmpstr.RtmpImgSdk_SendRgb24.restype = c_int
            queue_len = rtmpstr.RtmpImgSdk_SendRgb24(pointer, frame, sqr_w*sqr_h*3, 0)  # const char* dataBuf, int dataSize, int isWithAudio
            print(queue_len)
        elif flag_glob == True:
            img = send_Queue.get()
            frame = np.asarray(img, dtype=np.uint8)
            frame = frame.ctypes.data_as(ctypes.c_char_p)
            rtmpstr.RtmpImgSdk_SendRgb24.argtypes = [c_void_p, c_char_p, c_int, c_int]
            rtmpstr.RtmpImgSdk_SendRgb24.restype = c_int
            queue_len = rtmpstr.RtmpImgSdk_SendRgb24(pointer, frame, sqr_w*sqr_h*3, 1)  # const char* dataBuf, int dataSize, int isWithAudio
            print(queue_len)



def run_task(content_txt, in_audio, callback_url, task_id):
    audio_path = 'outputs/' 
    kxtts(content_txt, in_audio, audio_path, aud_speed, 'audio')
    audio_path = audio_path + 'audio.wav'
    mel_chunks = aud2feature(args, fps, audio_path)

    # ---- 音频信号输入容器 ---- 
    command = 'ffmpeg -i {} -ss 00:00:00.30 -t 00:10:00.00 -ac 1 -ar 16000 -y {}'.format('outputs/audio.wav', 'outputs/audio2.wav')
    subprocess.call(command, shell=True)
    aud_dir = './outputs/audio2.wav'
    aud_dir = aud_dir.encode('utf-8')
    aud_dir = c_char_p(aud_dir)
    rtmpstr.RtmpImgSdk_SetWavFile.argtypes = [c_void_p, c_char_p]
    aud_flag = rtmpstr.RtmpImgSdk_SetWavFile(pointer, aud_dir)

    wav_queue = Queue(maxsize=3)
    depth_queue = Queue(maxsize=3)
    dagan_queue = Queue(maxsize=3)
    producer_wav2lip = Wav2lip_Producer('Wav2lip', mel_chunks, imgs_Queue, wav_queue)
    producer_depth = Depth_Producer('Depth', mel_chunks, wav_queue, depth_queue)
    producer_dagan = DaGAN_Producer('DaGAN', mel_chunks, depth_queue, dagan_queue)    
    producer_swap = Swap_Producer('Swap', mel_chunks, dagan_queue, task_id, callback_url)
    producer_wav2lip.start()
    producer_depth.start()
    producer_dagan.start()
    producer_swap.start()
    producer_wav2lip.join()
    producer_depth.join()
    producer_dagan.join()
    producer_swap.join()
    print('video-success')
    


@app.route('/actor', methods=['POST'])
def main():
    """
    AI虚拟人的API接口
    输入文本, 输出播报视频
    ---
    tags:
      - AI虚拟人 API
    parameters:
      - name: callbackUrl
        in: formData
        type: string
        required: true
        description: callback_url
      - name: content_txt
        in: formData
        type: string
        required: true
        description: text
      - name: id
        in: formData
        type: string
        required: true
        description: id
    responses:
      500:
        description: input data error
      200:
        description: AI anchor URL
        schema:
          id: results
          properties:
            result:
              type: string
              description: anchor result
              default: 
            time:
              type: string
              description: anchor time
              default: 
    """ 
    content_txt = request.form.get("content_txt")
    callback_url = request.form.get("callbackUrl")
    task_id = request.form.get("id")
    print('QQQQQQ:', content_txt)
    in_audio = 'x2_yifei'  # 
    content_txt = '[p200]' + content_txt
    
    # ---- 请求队列 ---- #
    global flag_glob
    try:
        if flag_glob == False:
            t = Thread(target=run_task, args=(content_txt, in_audio, callback_url, task_id))  # 新建线程，处理异步任务
            t.start()  # 启动线程
            re_txt = 'success'
        else:
            re_txt = 'skip'
            res = {"taskId":task_id, "triggerFlag":1}
            backinfo = requests.post(callback_url, json=res) 
    except:
        re_txt = 'fail'
        res = {"taskId":task_id, "triggerFlag":1}
        backinfo = requests.post(callback_url, json=res) 
    return re_txt


if __name__ == "__main__":
    
    # ---- 宏定义 ---- #
    dev_id = 0
    fps = 15
    device_info = YoloV5Face.get_available_devices()[dev_id]

    # ---- 输入输出数据路径 ---- #
    in_dir_vid = './inputs/caorui4_1.mp4'
    in_dir_vid2 = './inputs/caorui4_1_2.mp4'
    crop_x, crop_y = 242, 86
    crop_area = [crop_x, crop_y, crop_x+256, crop_y+256]

    # ---- TTS参数设置 ---- #
    timbre = 'x2_yifei'  # 声音音色 x2_yifei (女) x2_qige(男)  x2_xiaoxue
    aud_speed = 38  # 语速设置 1-100

    # ---- 其它模型加载 ---- #
    model_path = 'modelhub/wav2lip_cn.pth'
    device = 'cuda'
    args = parser_define()
    Yolov5 = YoloV5Face(device_info)  # 人脸检测模型
    wav_model = wav2lip_load_model(model_path, device)  # wav2lip模型加载
    insightface_2d106 = InsightFace2D106(device_info)
    
    # ---- DaGAN模型加载 ---- #
    depth_encoder = depth.ResnetEncoder(18, False)  # 18,50
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
    loaded_dict_enc = torch.load('weights/encoder.pth')
    loaded_dict_dec = torch.load('weights/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    depth_encoder.cuda()
    depth_decoder.cuda()
    opt = dagan_parser()
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    # ---- 读取视频数据 ---- #
    cap = cv2.VideoCapture(in_dir_vid)  # 打开相机
    sqr_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 帧高度
    sqr_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽度
    
    cap2 = cv2.VideoCapture(in_dir_vid2)  # 打开相机
    sqr_h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 帧高度
    sqr_w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽度

    frame_list = []
    while True:
        ret, frame = cap.read()  # 捕获一帧图像
        ret2, frame2 = cap2.read()  # 捕获一帧图像
        if ret and ret2:
            frame_list.append([frame, frame2])
            cv2.waitKey(1)
        else:
            break
    cap.release()
    print('frames loaded')

    # ---- rtmp-sdk-接口调用 ---- #
    rtmpstr = cdll.LoadLibrary('./librtmpimg_stream.so')
    rtmpstr.RtmpImgSdk_Enviroment_Init(True)
    rtmpstr.RtmpImgSdk_Create.argtypes = [c_int, c_int, c_int, c_int, c_char_p, c_int, c_int, c_int]
    rtmpstr.RtmpImgSdk_Create.restype = c_void_p
    url_rtmp = "rtmp://avatarsrs.wenge.com/live/livestream?secret=9cff4eb7ac174242a60f10ffd19609ab"
    url_rtmp = url_rtmp.encode('utf-8')
    url_rtmp = c_char_p(url_rtmp)
    pointer = rtmpstr.RtmpImgSdk_Create(sqr_w, sqr_h, 16000, 1, url_rtmp, 1500, fps, 25)  # int width, int height, int sampleRate, int channelCount, const char* rtmpUrl, int bitrate, int fps
    rtmpstr.RtmpImgSdk_Start.argtypes = [c_void_p]
    rtmpstr.RtmpImgSdk_Start(pointer)

    # ---- 读取视频首帧 ---- #
    frame = frame_list[0][0]
    crop_frame = frame[crop_area[1]:crop_area[3],crop_area[0]:crop_area[2]].copy()
    source_image = cv2.cvtColor(crop_frame.copy(), cv2.COLOR_BGR2RGB)
    source_image = torch.tensor(source_image[np.newaxis].astype(np.float32)).cuda().permute(0, 3, 1, 2)/255
    outputs = depth_decoder(depth_encoder(source_image))
    depth_source = outputs[("disp", 0)]  # [1, 1, 256, 256])
    source_kp = torch.cat((source_image, depth_source), 1)  # [1, 4, 256, 256])
    kp_source = kp_detector(source_kp)
    kp_driving_initial = kp_source

    # ---- 视频读取线程 ---- #
    global queue_len
    global flag_glob
    flag_glob = False
    queue_len = 0
    imgs_Queue = Queue(maxsize=2)
    send_Queue = Queue(maxsize=2)
    e = Thread(target=raw_vid, args=(imgs_Queue, frame_list)) 
    e.start()

    # ---- 视频发送线程 ---- #
    r = Thread(target=send_vid, args=(imgs_Queue, send_Queue)) 
    r.start() 

    # ---- 启动flask问答出发线程 ---- #
    app.run(host='0.0.0.0', port=7150, debug=False)  # 事件主程序    


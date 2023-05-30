from pathlib import Path
from typing import Iterator, List, Tuple, Union
import numpy as np
import cv2
import time
import numexpr as ne
from xlib import math as lib_math
from xlib import avecl as lib_cl
from xlib.math import Affine2DMat, Affine2DUniMat
from xlib.image import ImageProcessor
from xlib.face import ELandmarks2D, FLandmarks2D, FPose
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info, get_cpu_device_info, )
import logmmse
import wave


class YoloV5Face:
    """
    YoloV5Face face detection model.

    arguments

     device_info    ORTDeviceInfo

        use YoloV5Face.get_available_devices()
        to determine a list of avaliable devices accepted by model

    raises
     Exception
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        # print(YoloV5Face.get_available_devices())
        # if device_info not in YoloV5Face.get_available_devices():
        #     raise Exception(f'device_info {device_info} is not in available devices for YoloV5Face')

        path = 'modelhub/YoloV5Face.onnx'
        self._sess = sess = InferenceSession_with_device(path, device_info)
        self._input_name = sess.get_inputs()[0].name

    def extract(self, img, threshold: float = 0.3, fixed_window=0, min_face_size=8, augment=False):
        """
        arguments

         img    np.ndarray      ndim 2,3,4

         fixed_window(0)    int  size
                                 0 mean don't use
                                 fit image in fixed window
                                 downscale if bigger than window
                                 pad if smaller than window
                                 increases performance, but decreases accuracy

         min_face_size(8)

         augment(False)     bool    augment image to increase accuracy
                                    decreases performance

        returns a list of [l,t,r,b] for every batch dimension of img
        """

        ip = ImageProcessor(img)
        _, H, W, _ = ip.get_dims()
        # if H > 2048 or W > 2048:
        #     fixed_window = 2048

        if fixed_window != 0:
            fixed_window = max(32, max(1, fixed_window // 32) * 32)
            img_scale = ip.fit_in(fixed_window, fixed_window, pad_to_target=True, allow_upscale=False)
        else:
            ip.pad_to_next_divisor(64, 64)
            img_scale = 1.0

        ip.ch(3).to_ufloat32()

        _, H, W, _ = ip.get_dims()

        preds = self._get_preds(ip.get_image('NCHW'))

        if augment:
            rl_preds = self._get_preds(ip.flip_horizontal().get_image('NCHW'))
            rl_preds[:, :, 0] = W - rl_preds[:, :, 0]
            preds = np.concatenate([preds, rl_preds], 1)

        faces_per_batch = []
        for pred in preds:
            pred = pred[pred[..., 4] >= threshold]

            x, y, w, h, score = pred.T

            l, t, r, b = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            keep = lib_math.nms(l, t, r, b, score, 0.5)
            l, t, r, b = l[keep], t[keep], r[keep], b[keep]

            faces = []
            for l, t, r, b in np.stack([l, t, r, b], -1):
                if img_scale != 1.0:
                    l, t, r, b = l / img_scale, t / img_scale, r / img_scale, b / img_scale

                if min(r - l, b - t) < min_face_size:
                    continue
                faces.append((l, t, r, b))

            faces_per_batch.append(faces)

        if len(faces_per_batch[0]) == 0:
            print("interrupt")

        # f = open("debug.log", "a", encoding='utf-8')
        # f.write(str(preds))
        # f.write("\n\n\n---------------------------------------\n\n")
        # f.close()

        return faces_per_batch

    def _get_preds(self, img):
        N, C, H, W = img.shape
        preds = self._sess.run(None, {self._input_name: img})
        # YoloV5Face returns 3x [N,C*16,H,W].
        # C = [cx,cy,w,h,thres, 5*x,y of landmarks, cls_id ]
        # Transpose and cut first 5 channels.
        pred0, pred1, pred2 = [pred.reshape((N, C, 16, pred.shape[-2], pred.shape[-1])).transpose(0, 1, 3, 4, 2)[..., 0:5] for pred in preds]

        pred0 = YoloV5Face.process_pred(pred0, W, H, anchor=[[4, 5], [8, 10], [13, 16]]).reshape((N, -1, 5))
        pred1 = YoloV5Face.process_pred(pred1, W, H, anchor=[[23, 29], [43, 55], [73, 105]]).reshape((N, -1, 5))
        pred2 = YoloV5Face.process_pred(pred2, W, H, anchor=[[146, 217], [231, 300], [335, 433]]).reshape((N, -1, 5))

        return np.concatenate([pred0, pred1, pred2], 1)[..., :5]

    @staticmethod
    def process_pred(pred, img_w, img_h, anchor):
        pred_h = pred.shape[-3]
        pred_w = pred.shape[-2]
        anchor = np.float32(anchor)[None, :, None, None, :]

        _xv, _yv, = np.meshgrid(np.arange(pred_w), np.arange(pred_h), )
        grid = np.stack((_xv, _yv), 2).reshape((1, 1, pred_h, pred_w, 2)).astype(np.float32)

        stride = (img_w // pred_w, img_h // pred_h)

        pred[..., [0, 1, 2, 3, 4]] = YoloV5Face._np_sigmoid(pred[..., [0, 1, 2, 3, 4]])

        pred[..., 0:2] = (pred[..., 0:2] * 2 - 0.5 + grid) * stride
        pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * anchor
        return pred

    @staticmethod
    def _np_sigmoid(x: np.ndarray):
        """
        sigmoid with safe check of overflow
        """
        x = -x
        c = x > np.log(np.finfo(x.dtype).max)
        x[c] = 0.0
        # result = 1 / (1 + np.exp(x))
        with np.errstate(divide="ignore", under='ignore', over='ignore'):  # 解决gcc版本导致的编译报错
            result = 1 / (1 + np.exp(x))
        result[c] = 0.0
        return result


class InsightFace2D106:
    """
    arguments

     device_info    ORTDeviceInfo

        use InsightFace2D106.get_available_devices()
        to determine a list of avaliable devices accepted by model

    raises
     Exception
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        # if device_info not in InsightFace2D106.get_available_devices():
            # raise Exception(f'device_info {device_info} is not in available devices for InsightFace2D106')

        path = 'modelhub/InsightFace2D106.onnx'
        self._sess = sess = InferenceSession_with_device(path, device_info)
        self._input_name = sess.get_inputs()[0].name
        self._input_width = 192
        self._input_height = 192

    def extract(self, img):
        """
        arguments

         img    np.ndarray      HW,HWC,NHWC uint8/float32

        returns (N,106,2)
        """
        ip = ImageProcessor(img)
        N, H, W, _ = ip.get_dims()

        h_scale = H / self._input_height
        w_scale = W / self._input_width

        feed_img = ip.resize((self._input_width, self._input_height)).swap_ch().as_float32().ch(3).get_image('NCHW')

        lmrks = self._sess.run(None, {self._input_name: feed_img})[0]
        lmrks = lmrks.reshape((N, 106, 2))
        lmrks /= 2.0
        lmrks += (0.5, 0.5)
        lmrks *= (w_scale, h_scale)
        lmrks *= (W, H)

        return lmrks


class DFMModel:
    def __init__(self, model_path: Path, device: ORTDeviceInfo = None):
        if device is None:
            device = get_cpu_device_info()
        self._model_path = model_path

        sess = self._sess = InferenceSession_with_device(str(model_path), device)

        inputs = sess.get_inputs()

        if len(inputs) == 0:
            raise Exception(f'Invalid model {model_path}')
        else:
            if 'in_face' not in inputs[0].name:
                raise Exception(f'Invalid model {model_path}')
            else:
                self._input_height, self._input_width = inputs[0].shape[1:3]
                self._model_type = 1
                if len(inputs) == 2:
                    if 'morph_value' not in inputs[1].name:
                        raise Exception(f'Invalid model {model_path}')
                    self._model_type = 2
                elif len(inputs) > 2:
                    raise Exception(f'Invalid model {model_path}')

    def get_model_path(self) -> Path:
        return self._model_path

    def get_input_res(self) -> Tuple[int, int]:
        return self._input_width, self._input_height

    def has_morph_value(self) -> bool:
        return self._model_type == 2

    def convert(self, img, morph_factor=0.75):
        """
         img    np.ndarray  HW,HWC,NHWC uint8,float32

         morph_factor   float   used if model supports it

        returns

         img        NHW3  same dtype as img
         celeb_mask NHW1  same dtype as img
         face_mask  NHW1  same dtype as img
        """

        ip = ImageProcessor(img)

        N, H, W, C = ip.get_dims()
        dtype = ip.get_dtype()

        img = ip.resize((self._input_width, self._input_height)).ch(3).to_ufloat32().get_image('NHWC')

        if self._model_type == 1:
            out_face_mask, out_celeb, out_celeb_mask = self._sess.run(None, {'in_face:0': img})
        elif self._model_type == 2:
            out_face_mask, out_celeb, out_celeb_mask = self._sess.run(None, {'in_face:0': img, 'morph_value:0': np.float32([morph_factor])})

        out_celeb = ImageProcessor(out_celeb).resize((W, H)).ch(3).to_dtype(dtype).get_image('NHWC')
        out_celeb_mask = ImageProcessor(out_celeb_mask).resize((W, H)).ch(1).to_dtype(dtype).get_image('NHWC')
        out_face_mask = ImageProcessor(out_face_mask).resize((W, H)).ch(1).to_dtype(dtype).get_image('NHWC')

        return out_celeb, out_celeb_mask, out_face_mask


_n_mask_multiply_op_text = [
    f"float X = {'*'.join([f'(((float)I{i}) / 255.0)' for i in range(n)])}; O = (X <= 0.5 ? 0 : 1);" for n in range(5)]


def Face_Merge(frame_image, face_resolution, face_align_img, face_align_mask_img, face_swap_img, face_swap_mask_img,
               aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression, face_mask_erode, face_mask_blur):
    interpolation = lib_cl.EInterpolation.LINEAR  # lib_cl.EInterpolation.CUBIC  lib_cl.EInterpolation.LANCZOS4

    masks = []
    masks.append(lib_cl.Tensor.from_value(face_align_mask_img))  # face_mask_source
    masks.append(lib_cl.Tensor.from_value(face_swap_mask_img))  # face_mask_celeb

    masks_count = len(masks)
    if masks_count == 0:
        face_mask_t = lib_cl.Tensor(shape=(face_resolution, face_resolution), dtype=np.float32, initializer=lib_cl.InitConst(1.0))
    else:
        face_mask_t = lib_cl.any_wise(_n_mask_multiply_op_text[masks_count], *masks, dtype=np.uint8).transpose((2, 0, 1))

    face_mask_t = lib_cl.binary_morph(face_mask_t, face_mask_erode, face_mask_blur, fade_to_border=True, dtype=np.float32)
    face_swap_img_t = lib_cl.Tensor.from_value(face_swap_img).transpose((2, 0, 1), op_text='O = ((O_TYPE)I) / 255.0', dtype=np.float32)

    color_transfer = 'none'
    if color_transfer == 'rct':
        face_align_img_t = lib_cl.Tensor.from_value(face_align_img).transpose((2, 0, 1), op_text='O = ((O_TYPE)I) / 255.0', dtype=np.float32)
        face_swap_img_t = lib_cl.rct(face_swap_img_t, face_align_img_t, target_mask_t=face_mask_t, source_mask_t=face_mask_t)

    frame_face_mask_t = lib_cl.remap_np_affine(face_mask_t, aligned_to_source_uni_mat,
                                               interpolation=lib_cl.EInterpolation.LINEAR,
                                               output_size=(frame_height, frame_width),
                                               post_op_text='O = (O <= (1.0/255.0) ? 0.0 : O > 1.0 ? 1.0 : O);')
    frame_face_swap_img_t = lib_cl.remap_np_affine(face_swap_img_t, aligned_to_source_uni_mat,
                                                   interpolation=interpolation, output_size=(frame_height, frame_width),
                                                   post_op_text='O = clamp(O, 0.0, 1.0);')

    frame_image_t = lib_cl.Tensor.from_value(frame_image).transpose((2, 0, 1),
                                                                    op_text='O = ((float)I) / 255.0;' if frame_image.dtype == np.uint8 else None,
                                                                    dtype=np.float32 if frame_image.dtype == np.uint8 else None)

    opacity = 1
    if opacity == 1.0:
        frame_final_t = lib_cl.any_wise('O = I0*(1.0-I1) + I2*I1', frame_image_t, frame_face_mask_t, frame_face_swap_img_t, dtype=np.float32)
    else:
        frame_final_t = lib_cl.any_wise('O = I0*(1.0-I1) + I0*I1*(1.0-I3) + I2*I1*I3', frame_image_t, frame_face_mask_t, frame_face_swap_img_t, np.float32(opacity), dtype=np.float32)

    color_compression = 0
    if do_color_compression and color_compression != 0:
        color_compression = max(4, (127.0 - color_compression))
        frame_final_t = lib_cl.any_wise('O = ( floor(I0 * I1) / I1 ) + (2.0 / I1);', frame_final_t, np.float32(color_compression))

    return frame_final_t.transpose((1, 2, 0)).np()


def Face_Merge2(frame_image, face_resolution, face_align_img, face_align_mask_img, face_swap_img, face_swap_mask_img,
                aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression, face_mask_erode, face_mask_blur):
    interpolation = ImageProcessor.Interpolation.LINEAR  # lib_cl.EInterpolation.CUBIC  lib_cl.EInterpolation.LANCZOS4  # EInterpolation.LINEAR
    frame_image = ImageProcessor(frame_image).to_ufloat32().get_image('HWC')
    masks = []
    masks.append(ImageProcessor(face_align_mask_img).to_ufloat32().get_image('HW'))
    masks.append(ImageProcessor(face_swap_mask_img).to_ufloat32().get_image('HW'))

    masks_count = len(masks)  # 2
    if masks_count == 0:
        face_mask = np.ones(shape=(face_resolution, face_resolution), dtype=np.float32)
    else:
        face_mask = masks[0]
        for i in range(1, masks_count):
            face_mask *= masks[i]

    # Combine face mask
    face_mask = ImageProcessor(face_mask).erode_blur(face_mask_erode, face_mask_blur, fade_to_border=True).get_image('HWC')
    frame_face_mask = ImageProcessor(face_mask).warp_affine(aligned_to_source_uni_mat, frame_width, frame_height).clip2((1.0 / 255.0), 0.0, 1.0, 1.0).get_image('HWC')
    face_swap_img = ImageProcessor(face_swap_img).to_ufloat32().get_image('HWC')
    frame_face_swap_img = ImageProcessor(face_swap_img).warp_affine(aligned_to_source_uni_mat, frame_width, frame_height,
                                                                    interpolation=interpolation).get_image('HWC')

    # Combine final frame
    opacity = 1.0
    one_f = np.float32(1.0)
    if opacity == 1.0:
        out_merged_frame = ne.evaluate('frame_image*(one_f-frame_face_mask) + frame_face_swap_img*frame_face_mask')
    else:
        out_merged_frame = ne.evaluate(
            'frame_image*(one_f-frame_face_mask) + frame_image*frame_face_mask*(one_f-opacity) + frame_face_swap_img*frame_face_mask*opacity')

    return out_merged_frame


def audio_filter(path, file_save):
    f = wave.open(path, "r")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print("nchannels:", nchannels, "sampwidth:", sampwidth, "framerate:", framerate, "nframes:", nframes)
    data = f.readframes(nframes)
    f.close()
    data = np.fromstring(data, dtype=np.short)

    # 降噪
    data = logmmse.logmmse(data=data, sampling_rate=framerate)

    # 保存音频
    nframes = len(data)
    f = wave.open(file_save, 'w')
    f.setparams((1, 2, framerate, nframes, 'NONE', 'NONE'))  # 声道，字节数，采样频率，*，*
    # print(data)
    f.writeframes(data)  # outData
    f.close()
    return

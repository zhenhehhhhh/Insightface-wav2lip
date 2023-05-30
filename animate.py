import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import imageio
from scipy.spatial import ConvexHull
import numpy as np
import depth
from sync_batchnorm import DataParallelWithCallback
import yaml
import modules.generator as GEN
from modules.keypoint_detector import KPDetector
from collections import OrderedDict


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
    return kp_new


def load_checkpoints(config_path, checkpoint_path, generator, kp_num, cpu=True):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if kp_num != -1:
        config['model_params']['common_params']['num_kp'] = kp_num
    generator = getattr(GEN, generator)(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()
    config['model_params']['common_params']['num_channels'] = 4
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")

    ckp_generator = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['generator'].items())
    generator.load_state_dict(ckp_generator)
    ckp_kp_detector = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['kp_detector'].items())
    kp_detector.load_state_dict(ckp_kp_detector)

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

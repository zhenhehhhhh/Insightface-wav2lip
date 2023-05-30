from urllib.parse import quote_plus
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen
from models import Wav2Lip
import argparse
import torch
import audio
import json


# ---- wav2lip 宏定义 ---- #
def parser_define():
    parser = argparse.ArgumentParser(
        description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/wav2lip.pth',
                        help='Name of saved checkpoint to load weights from')
    parser.add_argument('--face', type=str,
                        help='Filepath of video/image that contains faces to use')
    parser.add_argument('--audio', type=str,
                        help='Filepath of video/audio file to use as raw audio source')
    parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                                    default='results/result_voice_1.mp4')
    parser.add_argument('--static', type=bool,
                        help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=32)
    parser.add_argument('--wav2lip_batch_size', type=int,
                        help='Batch size for Wav2Lip models(s)', default=64)
    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                        'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                        'Use if you get a flipped result, despite feeding a normal looking video')
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')

    args = parser.parse_args()  # 宏定义
    args.img_size = 96  # 设置图片处理中间环节参数
    args.mel_step_size = 16  #
    return args


def fetch_token(API_KEY, SECRET_KEY, TOKEN_URL, SCOPE):
	params = {'grant_type': 'client_credentials',
				'client_id': API_KEY,
				'client_secret': SECRET_KEY}
	post_data = urlencode(params)
	post_data = post_data.encode('utf-8')
	req = Request(TOKEN_URL, post_data)
	try:
		f = urlopen(req, timeout=5)
		result_str = f.read()
	except URLError as err:
		result_str = err.read()
	result_str = result_str.decode()

	result = json.loads(result_str)
	if 'access_token' in result.keys() and 'scope' in result.keys():
		if not SCOPE in result['scope'].split(' '):
			print('scope is not correct')
		print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' %
		      (result['access_token'], result['expires_in']))
		return result['access_token']
	else:
		return None


def _load(checkpoint_path, device):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def wav2lip_load_model(path, device):
	model = Wav2Lip()
	checkpoint = _load(path, device)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()


def aud2feature(args, fps, audio_path):
	wav = audio.load_wav(audio_path, 16000) # args.audio
	mel = audio.melspectrogram(wav)
	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + args.mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - args.mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + args.mel_step_size])
		i += 1
	print("Length of mel chunks: {}".format(len(mel_chunks)))
	return mel_chunks




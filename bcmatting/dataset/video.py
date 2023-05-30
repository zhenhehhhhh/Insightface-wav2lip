import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, path: str, transforms: any = None):
        self.cap = cv2.VideoCapture(path)  # 初始化视频读取参数
        self.transforms = transforms
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频参数
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.num = 0
    
    def __len__(self):
        return self.frame_count  # 返回总帧数
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):  # 判断数据类型
            return [self[i] for i in range(*idx.indices(len(self)))] 
        
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != idx:  # 读取视频当前帧同idx不对应的时候，设置为idx对应的帧
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
        ret, img = self.cap.read()  # 读取当前帧
        parse_img = cv2.imread('inputs/parsing/'+str(self.num)+'.png')
        if not ret:
            raise IndexError(f'Idx: {idx} out of length: {len(self)}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR2RGB转换
        img = Image.fromarray(img)  # 转为Image类型
        if self.transforms:
            img = self.transforms(img)  # 如果设置了转换参数，则进行图片的转换
        self.num = self.num + 1
        return img
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cap.release()  # 退出时释放视频缓存

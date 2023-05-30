from torch.utils.data import Dataset
from typing import List

class ZipDataset(Dataset):
    def __init__(self, datasets: List[Dataset], transforms=None, assert_equal_length=False):
        self.datasets = datasets
        self.transforms = transforms
        
        if assert_equal_length:  # 对所有数据集挨个比对，验证长度的一致性
            for i in range(1, len(datasets)):
                assert len(datasets[i]) == len(datasets[i - 1]), 'Datasets are not equal in length.'
    
    def __len__(self):
        return max(len(d) for d in self.datasets)  # 返回最长数据集的长度
    
    def __getitem__(self, idx):
        x = tuple(d[idx % len(d)] for d in self.datasets)
        if self.transforms:
            x = self.transforms(*x)
        return x

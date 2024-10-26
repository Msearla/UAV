import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import tools

coco_pairs = [
    (1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), 
    (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
    (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)
]

class Feeder(Dataset):
    def __init__(
        self, 
        data_path: str, 
        data_split: str, 
        p_interval: list = [0.95], 
        window_size: int = 64, 
        bone: bool = False, 
        vel: bool = False
    ):
        super(Feeder, self).__init__()
        self.data_path = data_path
        self.data_split = data_split
        self.p_interval = p_interval
        self.window_size = window_size
        self.bone = bone
        self.vel = vel
        self.load_data()
        
    def load_data(self):
        npz_data = np.load(self.data_path, allow_pickle=True)
        if self.data_split == 'train':
            self.data = npz_data['x_train']
            self.label = npz_data['y_train']
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        else:
            assert self.data_split == 'test'
            self.data = npz_data['x_test']
            self.label = npz_data['y_test']
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor, int):
        data_numpy = self.data[idx]  # M T V C
        label = self.label[idx]
        
        # 将 NumPy 数组转换为 PyTorch 张量
        data_tensor = torch.from_numpy(data_numpy).float()  # 转换为 float 张量
        
        # 调整维度顺序: M T V C -> C T V M
        data_tensor = data_tensor.permute(3, 1, 2, 0)  # C,T,V,M
        
        # 计算有效帧数 (不全为0的帧)
        valid_frame_num = torch.sum(data_tensor.sum(dim=0).sum(dim=-1).sum(dim=-1) != 0).item()
        
        if valid_frame_num == 0: 
            # 如果所有帧都是无效的，返回全零张量
            data_tensor = torch.zeros(
                (3, self.window_size, 17, 2), 
                dtype=data_tensor.dtype, 
                device=data_tensor.device
            )
        else:
            # 使用自定义工具函数进行裁剪和调整大小
            data_tensor = tools.valid_crop_resize(
                data_tensor, 
                valid_frame_num, 
                self.p_interval, 
                self.window_size
            )
        
        if self.bone:
            # 计算骨骼数据
            bone_data_tensor = torch.zeros_like(data_tensor)
            for v1, v2 in coco_pairs:
                bone_data_tensor[:, :, v1 - 1] = data_tensor[:, :, v1 - 1] - data_tensor[:, :, v2 - 1]
            data_tensor = bone_data_tensor
        
        if self.vel:
            # 计算速度（帧间差分）
            data_tensor[:, :-1] = data_tensor[:, 1:] - data_tensor[:, :-1]
            data_tensor[:, -1] = 0
        
        # 归一化：所有关节相对于第一个关节（通常是根关节）
        data_tensor = data_tensor - data_tensor[:, :, 0:1, :]  # all_joint - 0_joint
        
        # 确保 label 是 torch.Tensor
        label_tensor = torch.tensor(label).long()
        
        return data_tensor, label_tensor, idx  # C T V M

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

# 如果需要，可以取消注释以下调试代码
# if __name__ == "__main__":
#     # Debug
#     train_loader = torch.utils.data.DataLoader(
#                 dataset = Feeder(data_path = './save_2d_pose/V1.npz', data_split = 'train'),
#                 batch_size = 4,
#                 shuffle = True,
#                 num_workers = 2,
#                 drop_last = False)
    
#     val_loader = torch.utils.data.DataLoader(
#             dataset = Feeder(data_path = './save_2d_pose/V1.npz', data_split = 'test'),
#             batch_size = 4,
#             shuffle = False,
#             num_workers = 2,
#             drop_last = False)
    
#     for batch_idx, (data, label, idx) in enumerate(train_loader):
#         data = data.float() # B C T V M
#         label = label.long() # B
#         print("pause")

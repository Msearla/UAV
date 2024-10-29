import os
import sys
import torch
import numpy as np
import pickle
import yaml
from tqdm import tqdm
import random

# 添加模型代码所在的路径
sys.path.append('./model')

def init_seed(seed):
    """初始化随机种子，确保结果可重复"""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    """动态导入类"""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found' % class_str)

# 导入 tools.py 中的函数
from dataset.tools import valid_crop_resize

def preprocess_data(data_paths, window_size=64, p_interval=[0.95]):
    """使用与训练相同的预处理步骤处理数据"""
    # COCO关键点对，用于计算骨骼特征
    coco_pairs = [
        (1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7),
        (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
        (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)
    ]

    processed_data = {}
    for key, path in data_paths.items():
        if os.path.exists(path):
            print(f"\nProcessing {key} data from {path}")

            # 直接加载npy文件
            data = np.load(path)  # [N, C, T, V, M] format
            print(f"Original data shape: {data.shape}")

            N, C, T, V, M = data.shape

            # 根据数据类型设置参数
            is_bone = 'B' in key and 'M' not in key  # 骨骼数据
            is_vel = 'M' in key  # 运动数据

            # 初始化处理后的数据列表
            processed_samples = []

            for idx in tqdm(range(N), desc=f"Processing {key} data"):
                data_numpy = data[idx]  # [C, T, V, M]

                # 计算有效帧数 (不全为0的帧)
                valid_frame_mask = (data_numpy.sum(axis=(0, 2, 3)) != 0)
                valid_frame_num = valid_frame_mask.sum()

                if valid_frame_num == 0:
                    # 如果所有帧都是无效的，返回全零张量
                    data_tensor = np.zeros(
                        (C, window_size, V, M),
                        dtype=np.float32
                    )
                else:
                    # 使用 valid_crop_resize 进行裁剪和调整大小
                    data_tensor = valid_crop_resize(
                        data_numpy,
                        int(valid_frame_num),
                        p_interval,
                        window_size
                    )  # 返回的是 numpy 数组

                if is_bone:
                    # 计算骨骼数据
                    bone_data = np.zeros_like(data_tensor)
                    for v1, v2 in coco_pairs:
                        bone_data[:, :, v1 - 1] = data_tensor[:, :, v1 - 1] - data_tensor[:, :, v2 - 1]
                    data_tensor = bone_data

                if is_vel:
                    # 计算速度（帧间差分）
                    vel_data = np.zeros_like(data_tensor)
                    vel_data[:, :-1] = data_tensor[:, 1:] - data_tensor[:, :-1]
                    vel_data[:, -1] = 0
                    data_tensor = vel_data

                # 归一化：所有关节相对于第一个关节（通常是根关节）
                data_tensor = data_tensor - data_tensor[:, :, 0:1, :]  # all_joint - root_joint

                processed_samples.append(data_tensor)

            # 将处理后的数据堆叠起来
            processed_data[key] = np.stack(processed_samples)
            print(f"Processed {key} data shape: {processed_data[key].shape}")
        else:
            print(f"Data file {path} does not exist.")
            processed_data[key] = None

    return processed_data

def load_model(model_name, model_weights_path, model_config_path, device):
    # 根据模型名称导入模型代码
    if 'ctrgcn' in model_name.lower():
        from ctrgcn_xyz import Model
    elif 'mstgcn' in model_name.lower():
        from mstgcn_xyz import Model
    elif 'tdgcn' in model_name.lower():
        from tdgcn_xyz import Model
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # 加载配置文件
    with open(model_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 获取模型参数
    model_args = config.get('model_args', {})
    model_args['num_class'] = config.get('num_class', 155)
    model_args['in_channels'] = 3

    # 初始化模型
    model = Model(**model_args)
    model = model.to(device)

    # 加载模型权重
    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def predict(model, data, batch_size=64):
    device = next(model.parameters()).device
    num_samples = data.shape[0]
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc='Predicting'):
            batch_data = data[i:i + batch_size]
            batch_data = torch.tensor(batch_data).float().to(device)
            outputs = model(batch_data)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    return predictions

if __name__ == "__main__":
    # 设置随机种子
    init_seed(1)

    # 定义测试集数据路径
    data_paths = {
        'J': './TestdataB/test_B_joint.npy',
        'JM': './TestdataB/test_B_joint_motion.npy',
        'B': './TestdataB/test_B_bone.npy',
        'BM': './TestdataB/test_B_bone_motion.npy'
    }

    # 使用与训练相同的预处理步骤处理数据
    test_data = preprocess_data(data_paths, window_size=64, p_interval=[0.95])

    # 定义模型列表和对应的输入数据类型
    models_info = [
        # CTR-GCN Models
        {'name': 'ctrgcn_V1_B_3D', 'data_type': 'B'},
        {'name': 'ctrgcn_V1_BM_3D', 'data_type': 'BM'},
        {'name': 'ctrgcn_V1_J_3D', 'data_type': 'J'},
        {'name': 'ctrgcn_V1_JM_3D', 'data_type': 'JM'},
        # MS-TGCN Models
        {'name': 'mstgcn_V1_B_3D', 'data_type': 'B'},
        {'name': 'mstgcn_V1_BM_3D', 'data_type': 'BM'},
        {'name': 'mstgcn_V1_J_3D', 'data_type': 'J'},
        {'name': 'mstgcn_V1_JM_3D', 'data_type': 'JM'},
        # TD-GCN Models
        {'name': 'tdgcn_V1_B_3D', 'data_type': 'B'},
        {'name': 'tdgcn_V1_J_3D', 'data_type': 'J'},
        {'name': 'tdgcn_V1_JM_3D', 'data_type': 'JM'},
    ]

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预测并保存得分
    score_files = []
    for model_info in models_info:
        model_name = model_info['name']
        data_type = model_info['data_type']
        print(f"\nProcessing model: {model_name}, data type: {data_type}")

        # 模型权重路径
        model_weights_path = f'./pt/{model_name}.pt'
        # 模型配置文件路径
        config_name = model_name.lower().replace('3d', '3d') + '.yaml'
        model_config_path = f'./config/{config_name}'

        # 检查文件是否存在
        if not os.path.exists(model_weights_path):
            print(f"Model weights file {model_weights_path} does not exist.")
            continue
        if not os.path.exists(model_config_path):
            print(f"Model config file {model_config_path} does not exist.")
            continue
        if test_data.get(data_type) is None:
            print(f"No test data for data type {data_type}.")
            continue

        # 加载模型
        model = load_model(model_name, model_weights_path, model_config_path, device)

        # 加载对应的数据
        data = test_data[data_type]

        # 检查数据形状
        if data.ndim == 5:
            N, C, T, V, M = data.shape
            print(f"Data shape is correct: [N, C, T, V, M] = {data.shape}")
        else:
            print(f"Unexpected data shape: {data.shape}")
            continue

        # 进行预测
        scores = predict(model, data, batch_size=64)

        # 保存预测得分
        score_file = f'{model_name}_test_score.pkl'
        with open(score_file, 'wb') as f:
            pickle.dump(scores, f)
        print(f"Saved prediction scores to {score_file}")

        score_files.append(score_file)

    # 模型融合权重
    weights = [1.5, 0.05, 1.3, 0.05,
               1.2, 0.05, 1.4, 0.05,
               1.5, 1, 0.05]

    # 加权融合
    def ensemble_scores(score_files, weights):
        final_score = None
        for idx, score_file in enumerate(score_files):
            if not os.path.exists(score_file):
                print(f"Score file {score_file} does not exist.")
                continue
            with open(score_file, 'rb') as f:
                scores = pickle.load(f)
            scores = torch.tensor(scores)
            weight = weights[idx]
            if final_score is None:
                final_score = weight * scores
            else:
                final_score += weight * scores
        return final_score

    final_score = ensemble_scores(score_files, weights)

    # 保存最终的置信度文件
    np.save('pred.npy', final_score.numpy())
    print("Final confidence scores saved to pred.npy")

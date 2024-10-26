import os
import sys
import torch
import numpy as np
import pickle
import yaml
from tqdm import tqdm

# 添加模型代码所在的路径
sys.path.append('./model')

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
    model_args['num_class'] = config.get('num_class', 155)  # 根据您的数据集类别数量
    model_args['in_channels'] = 3  # 一般情况下为3

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
            batch_data = data[i:i+batch_size]
            batch_data = torch.tensor(batch_data).float().to(device)
            outputs = model(batch_data)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    return predictions

if __name__ == "__main__":
    # 定义测试集数据路径
    data_paths = {
        'J': './TestdataB/test_B_joint.npy',
        'JM': './TestdataB/test_B_joint_motion.npy',
        'B': './TestdataB/test_B_bone.npy',
        'BM': './TestdataB/test_B_bone_motion.npy'
    }

    # 加载测试集数据
    test_data = {}
    for key, path in data_paths.items():
        if os.path.exists(path):
            test_data[key] = np.load(path)
            print(f"Loaded {key} data from {path}, shape: {test_data[key].shape}")
        else:
            print(f"Data file {path} does not exist.")
            test_data[key] = None

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
        # TD-GCN Models (没有 BM)
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
        # 模型配置文件路径（注意小写的 '3d'）
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
            # 数据已经是 [N, C, T, V, M] 形状
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

    # 定义模型得分文件和对应的权重
    # 更新 score_files 列表，确保只包含实际生成的得分文件
    weights = [1, 1, 0.05, 0.05,
            1, 1, 0.05, 0.05,
            1, 1, 0.05]

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

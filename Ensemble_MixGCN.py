import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 
    parser.add_argument(
        '--val_label',
        type=str,
        default='./Process_data/test_A_label.npy',
        help='Path to the validation set labels (.npy file)'),
    parser.add_argument(
        '--ctrgcn_J3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--ctrgcn_B3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--ctrgcn_JM3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--ctrgcn_BM3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--tdgcn_J3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_J_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--tdgcn_B3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_B_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--tdgcn_JM3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_JM_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--mstgcn_J3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_J_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--mstgcn_B3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_B_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--mstgcn_JM3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_JM_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--mstgcn_BM3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_BM_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--benchmark', 
        type = str,
        default = 'V1')
    return parser

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        fr = open(file,'rb') 
        inf = pickle.load(fr)

        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values)
        final_score += Rate[idx] * score
    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(label_npy_path):
    # 直接加载标签文件
    true_label = np.load(label_npy_path)
    # 如果需要转换为 Torch 张量
    true_label = torch.from_numpy(true_label)
    return true_label

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # 获取您的模型得分文件路径
    ctrgcn_j3d_file = args.ctrgcn_J3d_Score
    ctrgcn_b3d_file = args.ctrgcn_B3d_Score
    ctrgcn_jm3d_file = args.ctrgcn_JM3d_Score
    ctrgcn_bm3d_file = args.ctrgcn_BM3d_Score
    
    mstgcn_j3d_file = args.mstgcn_J3d_Score
    mstgcn_b3d_file = args.mstgcn_B3d_Score
    mstgcn_jm3d_file = args.mstgcn_JM3d_Score
    mstgcn_bm3d_file = args.mstgcn_BM3d_Score
    
    tdgcn_j3d_file = args.tdgcn_J3d_Score
    tdgcn_b3d_file = args.tdgcn_B3d_Score
    tdgcn_jm3d_file = args.tdgcn_JM3d_Score
    
    val_label_file = args.val_label
    
    # 将模型得分文件路径添加到列表中
    File = [ctrgcn_j3d_file, ctrgcn_b3d_file, ctrgcn_jm3d_file, ctrgcn_bm3d_file,
            mstgcn_j3d_file, mstgcn_b3d_file, mstgcn_jm3d_file, mstgcn_bm3d_file,
            tdgcn_j3d_file, tdgcn_b3d_file, tdgcn_jm3d_file]   
     
    # 设置对应的权重，根据模型的重要性和性能来设置
    Rate = [1, 1, 0.05, 0.05,
            1, 1, 0.05, 0.05,
            1, 1, 0.05]
    
    # 设置类别数量和样本数量
    Numclass = 155  # 根据您的数据集确定
    Sample_Num = 2000  # 根据您的验证集样本数量确定
    
    # 计算得分和准确率
    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
    true_label = gen_label(val_label_file)
    Acc = Cal_Acc(final_score, true_label)
    
    print('acc:', Acc)

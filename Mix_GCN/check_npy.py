import numpy as np

# 替换为您的 .npy 文件路径
file_path = 'Model_inference\Mix_GCN\pred.npy'

# 加载 .npy 文件
data = np.load(file_path)

# 打印数据的维度
print(f"数据维度（shape）：{data.shape}")

# 显示数据类型
print(f"数据类型（dtype）：{data.dtype}")

# 显示部分数据内容
# 如果数据是多维数组，您可以根据需要进行切片
print("数据示例：")
print(data)

# 如果数据量较大，可以只打印部分内容，例如前5个元素
print("前5个元素：")
print(data[:5])

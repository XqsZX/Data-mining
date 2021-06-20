import numpy as np
import pandas as pd


# 读取数据
def loadIris(address):
    # 读取文件
    spf = pd.read_csv(address, sep=',', index_col=False, header=None)
    # 提取第四行
    strs = spf[4]
    # 去掉标签
    spf.drop([4], axis=1, inplace=True)
    return spf.values, strs


# 归一化处理
def normalization(data_matrix):
    e = 1e-5 # 防止出现 0，加一个拉普拉斯因子
    for c in range(4):
        maxNum = np.max(data_matrix[:, c])
        minNum = np.min(data_matrix[:, c])
        data_matrix[:, c] = (data_matrix[:, c] - minNum + e)/(maxNum - minNum + e)
    return data_matrix


if __name__ == '__main__':
    url_Path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    writepath = r'C:\Users\XqsZX\Desktop\实验一\iris_normalized.csv'
    data_matrix, str_name = loadIris(url_Path)
    data_matrix = normalization(data_matrix)
    spf = pd.DataFrame(data_matrix)
    strs = str_name.values
    spf.insert(4, 4, strs)
    spf.to_csv(writepath, index=False, header=False)

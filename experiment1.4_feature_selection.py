import numpy as np
import pandas as pd
from collections import Counter


# 读取数据
def loadIris(address):
    # 读取文件
    spf = pd.read_csv(address, sep=',', index_col=False, header=None)
    # 提取第四行
    strs = spf[4]
    # 去掉标签
    spf.drop([4], axis=1, inplace=True)
    return spf.values, strs


def featureSelection(features, label):
    featurelen = len(features[0, :])
    label_count = Counter(label)
    samples_energy = 0.0
    data_len = len(label)
    # print(label_count)
    # print(label_count.keys())
    # 计算信息熵
    for i in label_count.keys():
        label_count[i] /= float(data_len)
        samples_energy -= label_count[i] * np.log2(label_count[i])

    informationGain=[]
    # 计算信息增益
    for f in range(featurelen):
        af = features[:, f]
        minf = np.min(af)
        maxf = np.max(af) + 1e-4
        width = (maxf-minf) / 10.0
        d = (af - minf) / width
        dd = np.floor(d)
        c = Counter(dd)
        sub_energy = getEnergy(c, dd, label)
        informationGain.append(samples_energy-sub_energy)
    return informationGain


# 熵的计算
def getEnergy(c, data, label):
    datalen = len(label)
    energy = 0.0
    # print(c.items())
    for key, value in c.items():
        c[key] /= float(datalen)
        label_picked = label[data == key]
        l = Counter(label_picked)
        e = 0.0
        for k, v in l.items():
            r = v / float(value)
            e -= r * np.log2(r)
            energy += c[key] * e
    return energy


if __name__ == '__main__':
    url_Path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data_matrix, str_name = loadIris(url_Path)
    informationGain = featureSelection(data_matrix, str_name.values)
    print(informationGain)

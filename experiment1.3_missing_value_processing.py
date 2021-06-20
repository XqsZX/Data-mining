import pandas as pd
import numpy as np
from collections import Counter


# 读取数据
def loadLabor(address):
    # 读文件
    spf = pd.read_csv(address, sep=',', index_col=False)
    # 所有的数据名称
    column = spf.columns
    # 标签型数据
    str_typeName = ['cost-of-living-adjustment', 'pension', 'education-allowance', 'vacation',
                    'longterm-disability-assistance', 'contribution-to-dental-plan',
                    'bereavement-assistance', 'contribution-to-health-plan', 'class']
    # 缺失值
    str2numeric = {'?': '-1'}
    spf = spf.replace(str2numeric)
    return spf, str2numeric, str_typeName


# 缺失值处理
def fillMissData(spf, str2numeric):
    row, col = spf.shape
    columns = spf.columns
    for column_name in columns:
        # 数字型数据
        if column_name not in str2numeric:
            tmp = spf[column_name].apply(float)
            ave = np.average(tmp[tmp != -1])
            # 求中位数
            # mid = np.median(tmp[tmp != -1])
            # 求众数
            # counts = np.bincount(tmp[tmp != -1])
            # num = np.argmax(counts)
            tmp[tmp == -1] = ave
            spf[column_name] = tmp
        # 标签型数据
        else:
            v = spf[column_name].values
            v1 = v[v != '-1']
            # 对于字符串进行计数
            c = Counter(v1)
            # 找到最常见的字段
            cc = c.most_common(1)
            # 进行缺失值填充
            v[v == '-1'] = cc[0][0]
            spf[column_name] = v
    return spf


if __name__ == '__main__':
    filepath = r'C:\Users\XqsZX\Desktop\实验一\labor.csv'
    fillFilepath = r'C:\Users\XqsZX\Desktop\实验一\labor_processed.csv'
    spf, str2numeric, str_typeName=loadLabor(filepath)
    spf = fillMissData(spf, str_typeName)
    spf.to_csv(fillFilepath, index=False)

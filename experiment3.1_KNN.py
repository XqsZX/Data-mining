# -*- coding: utf-8 -*-
import pandas as pd


def distance(data1, data2):  # 计算两点距离
    dist = 0
    for i, j in zip(data1, data2):
        dist = dist + (i - j) ** 2
    dist = dist ** 0.5
    return dist


def knn(predict_data, train_data, k):
    dist_list = []
    for index, row in train_data.iterrows():  # 与每一个训练集中数据计算距离
        dist = distance(predict_data[:-1], row[:-1])
        dist_list.append(dist)
    dist_df = train_data.loc[:, ['class_name']]
    dist_df['distance'] = dist_list  # 将距离和类标签放入同一DataFrame中
    dist_df = dist_df.sort_values(by=['distance'],ascending=True)  # 根据距离进行升序排序
    dist_df_k = dist_df[:k]  #取前K个
    predict_class = 'Iris-setosa'
    class_num = 0
    for class_name in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:  # 统计三个类别那个最多
        temp_num = dist_df_k.groupby(['class_name']).size().get(class_name,0)  # 各类别的数量
        if temp_num > class_num:
            predict_class = class_name
            class_num = temp_num
    return predict_class


def predict(test_data, train_data, k):
    predict_class_list = []
    for _, test_row in test_data.iterrows():
        predict_class = knn(test_row, train_data, k)
        predict_class_list.append(predict_class)
    result_df = test_data.copy()
    result_df['predict_class'] = predict_class_list
    print(result_df.loc[:, ['class_name', 'predict_class']].head(10))
    return result_df


# 计算准确率
def calculate_accuracy(result_df):
    sum = len(result_df)
    right = 0
    for index, row in result_df.iterrows():
        if row['class_name'] == row['predict_class']:
            right += 1
    accuracy = right / sum
    print('准确率：', accuracy)


# 读取数据，并指定列名
names = ['petal_length', 'petal_width', 'class_name']  # 为每列指定一个列名
iris_training_data = pd.read_csv(r'D:\study\数据挖掘\实验三\forKNN\iris.2D.train.arff.csv', names=names)
iris_test_data = pd.read_csv(r'D:\study\数据挖掘\实验三\forKNN\iris.2D.test.arff.csv', names=names)

# 数据归一化
for col in names[:-1]:
    clo_max = iris_training_data[col].max()
    clo_min = iris_training_data[col].min()
    iris_training_data[col] = (iris_training_data[col] - clo_min) / (clo_max - clo_min)

# 数据归一化
for col in names[:-1]:
    clo_max = iris_test_data[col].max()
    clo_min = iris_test_data[col].min()
    iris_test_data[col] = (iris_test_data[col] - clo_min) / (clo_max - clo_min)

train_data = iris_training_data
test_data = iris_test_data

result_df = predict(test_data, train_data, 10)

calculate_accuracy(result_df)

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt


# 加载数据
def loaddataset(filepath):
    label_x = []
    label_y = []
    try:
        file = open(filepath, "r")  # 以读模式打开文件
    except FileNotFoundError:  # 如果文件不存在，给提示
        print("file is not found")
    else:
        contents = file.readlines()  # 读取全部行
        for content in contents:
            label_x.append(float(content.split(' ')[0]))
            label_y.append(float(content.split(' ')[1]))
    return label_x, label_y


# 分配
def assignment(_df, _centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        _df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (_df['x'] - _centroids[i][0]) ** 2
                + (_df['y'] - _centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    _df['closest'] = _df.loc[:, centroid_distance_cols].idxmin(axis=1)
    _df['closest'] = _df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    _df['color'] = _df['closest'].map(lambda x: colmap[x])
    return _df


# 更新
def update(x):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return x


X, Y = loaddataset('D:/study/数据挖掘/实验四/data.txt')
# 初始化–随机生成K个初始“均值”（质心）
df = pd.DataFrame({'x': X, 'y': Y})

np.random.seed(200)
k = 5
# centroids[i] = [x, y]
centroids = {
    i + 1: [np.random.randint(0, 100), np.random.randint(0, 100)]
    for i in range(k)
}

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color='k', marker='o')
colmap = {1: 'red', 2: 'green', 3: 'blue', 4: 'yellow', 5: 'purple'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], marker='v')
plt.xlim(-10, 110)
plt.ylim(-10, 110)
plt.show()
# 分配–通过将每个观测值与最近的质心相关联来创建K个聚类
df = assignment(df, centroids)
print(df.head())

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k', marker='o')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], marker='v')
plt.xlim(-10, 110)
plt.ylim(-10, 110)
plt.show()
# 更新–群集的质心成为新的均值
old_centroids = copy.deepcopy(centroids)
centroids = update(centroids)

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k', marker='o')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], marker='v')
plt.xlim(-10, 110)
plt.ylim(-10, 110)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
plt.show()
# 重复分配阶段
df = assignment(df, centroids)

# Plot results
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k', marker='o')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], marker='v')
plt.xlim(-10, 110)
plt.ylim(-10, 110)
plt.show()

# 继续，直到所有分配的类别不再发生变化
while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k', marker='o')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], marker='v')
plt.xlim(-10, 110)
plt.ylim(-10, 110)
plt.show()

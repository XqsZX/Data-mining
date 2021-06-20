import matplotlib.pyplot as plt
import pandas
# 导入数据集iris  
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# 读取csv数据
dataset = pandas.read_csv(url, names=names)
print(dataset.describe())
# 直方图 histograms
dataset.hist()
plt.show()
# 绘制散点图
dataset.plot(x='sepal-length', y='sepal-width', kind='scatter')
plt.show()
# 绘制密度图
dataset.plot(kind='kde')
plt.show()
# 绘制盒图
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

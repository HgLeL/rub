"""
步骤：（此项目是监督学习中的分类问题）
导入数据
概述数据
数据可视化
评估算法
实施预测
"""

# 1.导入数据
# 导入类库
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold              #K折交叉验证
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix         # 混淆矩阵
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression        # 逻辑回归，仅用于分类问题
from sklearn.tree import DecisionTreeClassifier           # 决策树算法，可用于回归和分类问题
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis     # 线性判别分析，分类
from sklearn.neighbors import KNeighborsClassifier        # K近邻算法，可用于回归和分类问题
from sklearn.naive_bayes import GaussianNB                # 朴素贝叶斯算法， 仅用于分类问题
from sklearn.svm import SVC                               # 支持向量机， 可用于分类（SVC）和回归（SVR）问题

# 导入数据
fileName = 'iris.data.csv'
names = ['separ-length','separ-width','petal-length','petal-width', 'class']
dataset = read_csv(fileName, names = names)
# 数据描述
dataset.shape() # 查看行列数（行，列）
print(dataset.head(10))  # 前10行
print(dataset.describe())  # 数据概括
print(dataset.groupby('class').size())  # 查看数据分类分布

# 数据可视化
dataset.plot(kind = 'box',subplots = True, layout = (2,2), sharex=False, sharey=False)  # 箱线图
dataset.hist()
scatter_matrix(dataset)
pyplot.show()

# 评估算法
"""
分割数据集为训练集和测试集
采用10折交叉验证评估模型
预测
选择最优
"""
array = dataset.values
x = array[:,0:4]
y = array[:,4]
validation_size = 0.2
seed = 7
x_train,x_validation,y_train,y_validation = train_test_split(x, y, test_size= validation_size, random_state= seed)

# 算法审查
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()
# 评估算法
results = []
for key in models:
    kfold = KFold(n_splits=10, random_state= seed)
    cv_results = cross_val_score(models[key], x_train,y_train,cv= kfold, scoring='accuracy')
    results.append(cv_results)
    print('%s: %f (%f)' %(key, cv_results.mean(),cv_results.std()))

# 箱线图比较算法
fig = pyplot.figure()
fig.suptitle("algorithm comparison")
ax = fig.add_subplot(111)  # 添加子图，参数一：子图总行数；参数二：子图总列数；参数三：子图位置
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())  # 设置刻度标注
pyplot.show()

# 使用评估数据集评估算法
svm = SVC()
svm.fit(x_train,y_train)
predictions = svm.predict(x_validation)
print(accuracy_score(y_validation,predictions))  # 精确度
print(confusion_matrix(y_validation,predictions))   # 混淆矩阵
print(classification_report(y_validation, predictions))  # 生成数据报告，各个类别的准确率，召回率，F1值，支持值

# 特征选择
# 移除零方差特征
from sklearn.feature_selection import VarianceThreshold
x = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
clf = VarianceThreshold(threshold=(0.8*(1-0.8)))
clf.fit_transform(x)

# 单变量特征选择
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
x,y = iris.data,iris.target
x_new = SelectKBest(chi2,k=2).fit_transform(x,y)  # 如何直接知道选择了哪些特征呢（不用复杂代码）
x_new.shape

# 递归特征消除
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
x = digits.images.reshape(len(digits.images),-1)  #  这里的-1是指根据原先的总个数和已给行数，来调整列数为合理的数
y = digits.target

svc = SVC(kernel='linear',C=1)
rfe = RFE(estimator=svc,n_features_to_select=1, step=1)   # n_features_to_select：（int or none）The number of features to select. If None, half of the features are selected.
rfe.fit(x,y)                                              # step 是指每次迭代移除的特征的个数或百分比
ranking = rfe.ranking_.reshape(digits.images[0].shape)   # rfe.ranking_=64,为1-64的某个排序     digits.images[0].shape=(8,8)

plt.matshow(ranking,cmap = plt.cm.Blues)  # 显示一个数组或矩阵
plt.colorbar()  # 颜色渐变条
plt.title("Ranking of pixels with RFE")
plt.show()

# 特征选择和数据建模结合在一起
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
clf = Pipeline([
    ("feature_selection",SelectFromModel(estimator=LinearSVC(penalty='l1'))),
    ("classification",RandomForestClassifier())
])
clf.fit(x,y)

# np.random.RandomState(1) # 1 为种子
# RandomState.rand(d0, d1, ..., dn)  #The dimensions of the returned array, should all be positive. If no argument is given a single Python float is returned.

# 多层感知机分类
from sklearn.neural_network import MLPClassifier   # 利用了BP
x = [[0,0],[1,1]]                           # 使用 SGD 或 Adam ，训练过程支持在线模式和小批量学习模式
y = [0,1]                                        # 小数据集下，lbfgs运行更快，表现更好；大数据集，adam更好
clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)  # alpha：L2 penalty (regularization term) parameter
clf.fit(x,y)
clf.predict([[2,2],[-1,-2]])

# 区域中浣熊脸的图片分割
from time import time
from sklearn.cluster import spectral_clustering
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.feature_extraction import image
from scipy.misc import face
face = face(gray=True)  # face.shape=(768, 1024), type(face)= <class 'numpy.ndarray'>
face = sp.misc.imresize(face,0.1)/255  #face.shape=(76,102),缩小当前尺寸的0.1，除以255，将数组归一化
graph = image.img_to_graph(face)    # graph.shape=(7752, 7752),sparse matrix of type '<class 'numpy.float64'>'

beta = 5
eps = 1e-5
graph.data = np.exp(-beta*graph.data/graph.data.std())+eps    # shape为（38404，)
N_REGIONS= 25

for assign_label in ('kmeans','discretize'):
    t0 = time()
    labels = spectral_clustering(graph,n_clusters=N_REGIONS,assign_labels=assign_label,random_state=1)
    t1 = time()
    labels = labels.reshape(face.shape)

    plt.figure(figsize=(10,10))
    plt.imshow(face,cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels==l,contours=1,colors=[plt.cm.spec(l/float(N_REGIONS))])
        plt.xticks(())
        plt.yticks(())
        title = 'spectral clustering:%s (%.2fs)'% (assign_label,(t1-t0))
        print(title)
        plt.title(title)
plt.show()

# 聚类性能度量
# 调整的rand指数
from sklearn import metrics
labels_true = [0,0,0,1,1,1]
labels_pred = [0,0,1,1,2,2]                        # 对于较小的样本数量或者较大数量的簇，使用 adjusted index 例如 Adjusted Rand Index (ARI)
metrics.adjusted_rand_score(labels_true,labels_pred)  # 对称的，交换参数不会改变分值，完美标签得分为1，在实践中几乎不可用，或者需要人工标注者手动分配（如在监督学习环境中）
metrics.adjusted_mutual_info_score(labels_true,labels_pred)
metrics.mutual_info_score(labels_true,labels_pred)  # 其完美标签不是1，因此不好判断
metrics.normalized_mutual_info_score(labels_true,labels_pred)
metrics.homogeneity_score(labels_true,labels_pred)  # 同质性，两者均在 0.0 以下 和 1.0 以上（越高越好）
metrics.completeness_score(labels_true,labels_pred)  # 完美性，两者均在 0.0 以下 和 1.0 以上（越高越好）
metrics.homogeneity_completeness_v_measure(labels_true,labels_pred)  # 同时计算三个指标
"""当不知道真实类别的标签时，可以使用sklearn.metrics.calinski_harabaz_score和sklearn.metrics.silhouette_score，
应用方式与以上不同
"""
# np.newaxis 为 numpy.ndarray（多维数组）增加一个轴
import numpy as np
data = np.arange(100).reshape(10, 10)
rows = np.array([0, 2, 3])[:, np.newaxis]
columns = np.array([1, 2])
print(data[rows, columns])

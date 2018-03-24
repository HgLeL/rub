from sklearn.linear_model import SGDClassifier
x = [[0,0],[1,1]]
y = [0,1]
clf = SGDClassifier(loss='hinge',penalty='l2').fit(x,y)
clf.coef_   # 系数
clf.decision_function([[2,2]])   # To get the signed distance to the hyperplane

from optparse import OptionParser
op = OptionParser()
op.add_option('-v',action = 'store',dest = 'verbose',help = "make many noise" )
op.add_option('-f',action = 'store_true',type='string',dest = 'fileName',help ="could see filename" )
fakearg = ['-v','good luck to you']
options,args = op.parse_args()
print(op.print_help())

# 无监督学习：最近邻
from sklearn.neighbors import NearestNeighbors
import numpy as np
x = np.array([[-1,-1],[-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nn = NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(x)
distances, indices = nn.kneighbors(x)
print(indices)
print(distances)
nn.kneighbors_graph(x).toarray()

"""
下面是给出人脸上半部分，预测下半部分的例子--使用k近邻回归
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state  # 将种子变成np.random.RandomState实例

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

data = fetch_olivetti_faces()   # 导入数据
targets = data.target    # 因变量

data = data.images.reshape((len(data.images),-1))   # (400,4096)
train = data[targets<30]   #(300,4096),图像是64x64的
test = data[targets>=30]  #（100,4096）

n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0],size=(n_faces,))  #  产生了五个随机数array([46, 55, 69,  1, 87])
test = test[face_ids,:]

n_pixels = data.shape[1]
x_train = train[:,:(n_pixels+1)//2]   # '//'表示整数除法，返回不大于结果的一个最大的整数，而" / " 则单纯的表示浮点数除法
y_train = train[:,n_pixels//2:]
x_test = test[:,:(n_pixels+1)//2]
y_test = test[:,n_pixels//2:]

ESTIMATORS={
    "Extra trees":ExtraTreesRegressor(n_estimators=10,max_features=32,random_state=0),
    'K-nn':KNeighborsRegressor(),
    "Linear regression":LinearRegression(),
    "RidgeCV":RidgeCV()
}

y_test_predict = dict()
for name,estimator in ESTIMATORS.items():
    estimator.fit(x_train,y_train)
    y_test_predict[name] = estimator.predict(x_test)

# 下面绘图
image_shape=(64,64)
n_col = 1+len(ESTIMATORS)
plt.figure(figsize=(2.0*n_col,2.26*n_faces))
plt.suptitle("face completion with multi-output estimtors",size=16)

for i in range(n_faces):
    true_face = np.hstack((x_test[i],y_test[i]))
    if i:
        sub = plt.subplot(n_faces,n_col,i*n_col+1)  # i*n_col+1 表示第几个子图，下同
    else:
        sub = plt.subplot(n_faces, n_col, i * n_col + 1,
                          title="true faces")

    sub.axis("off")   # 不显示坐标尺寸
    sub.imshow(true_face.reshape(image_shape),cmap=plt.cm.gray,interpolation="nearest")  # 二维矩阵数据的平面色彩显示

    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
    for j,est in enumerate(sorted(ESTIMATORS)):
        complete_face = np.hstack((x_test[i],y_test_predict[est][i]))
        if i:
            sub = plt.subplot(n_faces, n_col, i * n_col + 2 + j)
        else:
            sub = plt.subplot(n_faces, n_col, i * n_col + 2 + j,title = est)
        sub.axis("off")
        sub.imshow(complete_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")    # interpolation代表的是插值运算,
plt.show()

"""图像的批量处理
skimage.io.ImageCollection(load_pattern,load_func=None)，第一个参数load_pattern, 表示图片组的路径，可以是一个str字符串
第二个参数load_func是一个回调函数，我们对图片进行批量处理就可以通过这个回调函数实现。回调函数默认为imread(),即默认这个函数是批量读取图片。
"""
# import skimage.io as io   需要Microsoft visual c++ 14.0,才能安装scikit-image
# from skimage import data_dir
# str = data_dir+'/*.png'   str='d:/pic/*.jpg:d:/pic/*.png'  可既读取jpg又能读取png格式的图片，其他类似
# coll = io.ImageCollection(str)
# print(len(coll))

"""集成学习中bagging的应用"""
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(),   # 以KNeighborsClassifier()为基学习器
                            max_samples=0.5, max_features=0.5)
# 随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
iris = load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.3)
clf=RandomForestClassifier(n_estimators=10)
clf.fit(x_train,y_train)
print(clf.predict(x_test))
print(classification_report(y_test,clf.predict(x_test)))
print(accuracy_score(y_test,clf.predict(x_test)))
print(confusion_matrix(y_test,clf.predict(x_test)))
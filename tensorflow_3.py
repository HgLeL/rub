import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()
x_vals = np.linspace(0,10,100)
y_vals = x_vals + np.random.normal(0,1,100)
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1,100)))
A = np.column_stack((ones_column,x_vals_column))  # 这里必须先将要合并的矩阵用括号括起来
b = np.transpose(np.matrix(y_vals))

A_tensor = tf.constant(A)
b_tensor = tf.constant(b)
# 法1，效率低
tA_A = tf.matmul(tf.transpose(A_tensor),A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv,tf.transpose(A_tensor))
solution = tf.matmul(product,b_tensor)
solution_eval = sess.run(solution)

slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
# solution_eval = array([[0.08288525],
#                       [0.96574723]])
# 法2 ，通过分解矩阵的方法求解有时更高效，并且数值稳定
L = tf.cholesky(tA_A)
sol1 = tf.matrix_solve(L,tf.matmul(tf.transpose(A_tensor),b_tensor))  # 注意这里使用常数张量
sol2 = tf.matrix_solve(tf.transpose(L),sol1)
solution_eval = sess.run(sol2)
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]

fit = []
plt.figure()
for i in x_vals:
    fit.append(slope+i*y_intercept)
plt.plot(x_vals,y_vals,'ro',label='data')
plt.plot(x_vals,fit,'b-',label='fit')
plt.legend(loc='best')
plt.title('linear regression')
plt.show()

from tensorflow.python.framework import ops
from sklearn.datasets import load_iris
ops.reset_default_graph()  # 后面有括号
sess = tf.Session
iris = load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

leanrning_rate = 0.05
batch_size = 25
x_data = tf.placeholder(tf.float32,shape=[None,1])  # 这里的shape不是shape=[1],因为要批量计算
y_data = tf.placeholder(tf.float32,shape=[None,1])
A = tf.Variable(np.random.normal(shape=[1,1]))
b = tf.Variable(np.random.normal(shape=[1,1]))

"""
进行逻辑回归，将所有特征缩放到0-1区间（min-max缩放），收敛的效果较好，在缩放数据前，先分割训练集和测试集是相当重要的，
因为我们要确保训练集和测试集不相互受影响
tf.equal(A,B)  对比两个矩阵或向量的元素是否相等，如果相等就返回True,否则返回False
tf.cast(x,dtype,name=None) 将x的数据格式转换成dtype类型
tf.round(x)    舍入最近的整数
np.nan_to_num（x, copy=True)    将x中的NAN转换成0，inf转换成很大的数字，并代替之
"""

"""
lasso回归，岭回归，逻辑回归，线性回归，多元线性回归，戴明回归（total regression,与最小二乘回归最小化的距离不同），弹性网络回归，
这些处理的步骤均类似，区别在于损失函数的不同
"""

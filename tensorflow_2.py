import tensorflow as tf
sess = tf.Session()

# 声明张量和占位符
import numpy as np
x_vals = np.array([1.,3.,5.,7.,9.])
x_data = tf.placeholder(tf.float32)  # 括号中不是填float(32)
m_const = tf.constant(3.)
my_product = tf.multiply(x_data, m_const)
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data:x_val}))

# 在同一个计算图中进行多个乘法操作
#创建数据和占位符
my_array = np.array([[1.,3.,5.,7.,9.],
                     [-2.,0.,2.,4.,6.],
                     [-6.,-3.,0.,3.,6.]])
x_vals = np.array([my_array,my_array+1])  # x_vals.shape=(2,3,5)
x_data = tf.placeholder(tf.float32,shape=(3,5))
# 创建矩阵乘法和加法要用的常量矩阵
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])
# 声明操作，表示成计算图
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2,a1)
# 通过计算图赋值
for x_val in x_vals:
    print(sess.run(add1,feed_dict={x_data:x_val}))

# 多层layer
x_shape = [1,4,4,1] # [图片数量，高度，宽度，颜色通道]
x_val = np.random.uniform(size=x_shape)
x_data = tf.placeholder(tf.float32,shape=x_shape)
    # 创建过滤4x4像素图片的滑动窗口
my_filter = tf.constant(0.25,shape=[2,2,1,1])
my_strides = [1,2,2,1]
mov_avg_layer = tf.nn.conv2d(x_data,my_filter,my_strides,padding="SAME",name="Moving_Avg_Window")
def custom_layer(input_matrix):
    input_matrix_squeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1.,2.],[-1.,3.]])
    b = tf.constant(1.,shape=[2,2])
    temp1 = tf.matmul(A,input_matrix_squeezed)
    temp = tf.add(temp1,b)  # Ax+b
    return (tf.sigmoid(temp))
with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)
print(sess.run(custom_layer1,feed_dict={x_data:x_val}))

import matplotlib.pyplot as plt
x_vals = tf.linspace(-1.,1.,500)
target = tf.constant(0.0)

l2_y_vals = tf.square(target-x_vals)
l2_y_out = sess.run(l2_y_vals)

l1_y_vals = tf.abs(target-x_vals)
l1_y_out = sess.run(l1_y_vals)
x_array = sess.run(x_vals)
plt.plot(x_array,l2_y_out,'b-',label='L2 Loss')
plt.plot(x_array,l1_y_out,'g:',label='L1 Loss')
plt.ylim(-0.2,0.4)
plt.legend(loc="lower right",prop={'size':11})
plt.show()

delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(delta1,tf.sqrt(1+tf.square((target-x_vals)/delta1))-1)
phuber1_y_out = sess.run(phuber1_y_vals)
# 加权交叉熵损失函数
targets = tf.fill([500,],1.0)
weight = tf.constant(0.5)
xentroy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals,targets,weight)

# tensorflow 实现反向传播：例1：简单回归算法
import numpy as np
import tensorflow as tf
# 创建计算图会话
sess = tf.Session()
# 生成数据，创建占位符和变量A
x_vals = np.random.normal(1.0,0.1,100)
y_vals = np.repeat(10.,repeats=100)
x_data = tf.placeholder(dtype=tf.float32,shape=[1])
y_target = tf.placeholder(tf.float32,shape=[1])
A = tf.Variable(tf.random_normal(shape=[1]))
# 增加乘法操作
my_output = tf.multiply(x_data,A)
# 增加L2正则损失函数
loss = tf.square(y_target-my_output)
# 运行之前，初始化变量
init = tf.initialize_all_variables()
sess.run(init)
# 声明变量的优化器
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)
# 训练算法
for i in range(2000):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    if (i+1)%20==0:  # %取余数
        print('step #'+str(i+1)+' A = '+str(sess.run(A)))
        print('loss = '+str(sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})))

# 例2
# 首先，重置计算图，并重新初始化变量
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()
# 生成数据，目标标签，占位符和偏差
x_vals = np.concatenate((np.random.normal(-1.,1.,50),np.random.normal(3.,1.,50)))  # 数组拼接，参数axis=0按行拼接，为默认
y_vals = np.concatenate((np.repeat(0.,50),np.repeat(1.,50)))
x_data = tf.placeholder(tf.float32,shape=[1])
y_target = tf.placeholder(tf.float32,shape=[1])
A = tf.Variable(tf.random_normal(mean=10,shape=[1]))  # A是变量
# 增加转换操作
my_output = tf.add(x_data,A)
# 增加维度
my_output_expanded = tf.expand_dims(my_output,0)
y_target_expanded = tf.expand_dims(y_target,0)
# 初始化变量A
init = tf.initialize_all_variables()
sess.run(init)
# 声明损失函数
xentroy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded,labels=y_target_expanded)
# 增加一个优化器函数让tensorflow知道如何更新和偏差变量
my_opt= tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentroy)
# 通过随机选择的数据迭代，更新变量A
for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    if (i+1)%200==0:
        print('第'+str(i+1)+'步的 A = '+str(sess.run(A)))
        print('Loss: '+str(sess.run(xentroy,feed_dict={x_data:rand_x,y_target:rand_y})))

"""实现随机训练和批量训练"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()
#声明批量大小
batch_size = 20
# 声明模型的数据，占位符，变量
x_vals = np.random.normal(1,0.1,100)
y_vals = np.repeat(10.,100)
x_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))

init = tf.initialize_all_variables() # 一定要初始化变量
sess.run(init)
# 增加矩阵乘法操作
my_output = tf.matmul(x_data,A)
# 改变损失函数
loss = tf.reduce_mean(tf.square(my_output-y_target))
# 声明优化器
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)
# 循环迭代优化模型算法
loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100,size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])

    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    if (i+1)%5==0:
        print('第'+str(i+1)+'步的 A = '+str(sess.run(A)))
        temp_loss = sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
        print("Loss = "+str(temp_loss))
        loss_batch.append(temp_loss)

"""创建分类器"""
import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf
import matplotlib.pyplot as plt
sess = tf.Session()

iris = load_iris()
binary_target = np.array([1. if x==0 else 0 for x in iris.target])
iris_2d = np.array([[x[2],x[3]] for x in iris.data])

batch_size = 20
x1_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
x2_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

my_add= tf.add(tf.matmul(x2_data,A),b)
my_output = tf.subtract(x1_data,my_add)
xentroy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target,logits=my_output)

init = tf.initialize_all_variables() # 无参数
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentroy)

for i in range(1000):
    rand_index = np.random.choice(len(iris_2d),size = batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step,feed_dict={x1_data:rand_x1,x2_data:rand_x2,y_target:rand_y})
    if (i+1)%200==0:
        print('step #'+str(i+1)+'A='+str(sess.run(A))+', b='+str(sess.run(b)))

[[slope]] = sess.run(A)
[[interpt]] = sess.run(b)
x = np.linspace(0,3,50)
ablienValues = []
for i in x:
    ablienValues.append(slope*i+interpt)
setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==1]  # [[a[0]]]错误，np.array([[a[0]]])错误
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==0]
plt.plot(setosa_x,setosa_y,'rx',ms=10,mew=2,label='setosa')
plt.plot(non_setosa_x,non_setosa_y,'ro',label='Non-setosa')
plt.plot(x,ablienValues,'b-')
plt.xlim([0.0,2.7])
plt.ylim([0.0,7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()
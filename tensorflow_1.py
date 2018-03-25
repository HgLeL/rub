import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
import numpy as np
"""

# tf归一化数据
data = tf.nn.batch_norm_with_global_normalization()

# 设置机器学习参数
learning_rate = 0.01
batch_size = 100
iterations = 1000

# 初始化变量和占位符
a_var = tf.constant(42)
x_input = tf.placeholder(tf.float32,[None,input_size])
y_input = tf.placeholder(tf.float32,[None,num_classes])

# 定义模型结构： 通过选择操作、变量和占位符的值来构建计算图
y_pred = tf.add(tf.mul(x_input,weight_matrix),b_matrix)

# 声明损失函数：损失函数能说明预测值和实际值的差距
loss = tf.reduce_mean(tf.square(y_actual-y_pred))

# 初始化模型和训练模型：tf创建计算图实例，通过占位符赋值，维护变量的状态信息
with tf.Session(graph=graph) as session:
    ...
    session.run(...)
    ...
           # 或者以下方式
session = tf.Session(graph=graph)
session.run()
# 评估机器学习模型
# 调优超参数
# 发布/预测结果
"""

# 测试
w = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
liner_model = w * x + b
y = tf.placeholder(dtype=tf.float32)
loss = tf.reduce_sum(tf.square(liner_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
curr_w, curr_b, curr_loss = sess.run([w, b, loss], {x: x_train, y: y_train})
print("curr_w:%s,curr_b:%s,curr_loss:%s " % (curr_w, curr_b, curr_loss))

# 固定张量
zero_tsr = tf.zeros([2,2])  # ([行，列])
one_tsr = tf.ones([2,2])   # 创建指定维度的单位张量
filled_tsr = tf.fill([2,2],42) # 创建指定维度的常数填充的张量
constant_tsr = tf.constant([1,2,3]) # 用已知常数张量创建一个张量

# 相似形状的张量
zeros_similar = tf.zeros_like(constant_tsr) # 新建一个与给定的tensor类型大小一致 的张量，其所有元素为0或1
ones_similar = tf.ones_like(constant_tsr)

# 序列张量
linear_tsr = tf.linspace(start=float(0),stop=1,num=3)
integer_seq_tsr = tf.range(start = 6,limit=15,delta=3)

# 随机张量  下面的[2,2] 是指 shape=[行，列]
randunif_tsr = tf.random_uniform([2,2],minval=0,maxval=1) # 生成均匀分布的随机数
randnorm_tsr = tf.random_normal([2,2],mean=0,stddev=1) # 正态分布的随机数
runcnorm_tsr = tf.truncated_normal([2,2],mean=0,stddev=1,seed=None,name=None)# 其正态分布的随机数位于指定均值到两个标准差之间的区间
# shuffled_output = tf.random_shuffle(input_tensor) 张量的随机化
# cropped_output = tf.random_crop(input_tensor,crop_size)  对张量指定大小的随机剪裁，crop_size是指剪裁尺寸

"""创建好张量后，就可以通过tf.Variable()函数封装张量来作为变量"""
# my_var = tf.Variable(zero_tsr)
"""占位符是tensorflow对象，用于表示输入输出数据的格式，允许传入指定类型和形状的数据，并依赖计算图的计算结果，比如期望的计算结果"""

# 声明变量后需要初始化变量，下面创建变量并初始化
my_var = tf.Variable(tf.zeros([2,3]))
sess = tf.Session()
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)

sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[2,2]) #声明一个占位符
y = tf.identity(x) # 定义y为x的identity操作
x_vals = np.random.rand(2,2)
sess.run(y,feed_dict={x:x_vals})  # sess.run(x,feed_dict={x:x_vals}) 会报错

sess = tf.Session()
first_var = tf.Variable(tf.zeros([2,2]))
sess.run(first_var.initializer)
second_var = tf.Variable(tf.zeros_like(first_var))
sess.run(second_var.initializer)

sess = tf.Session()
identity_matrix = tf.diag([1.0,1.0,1.0])
A = tf.truncated_normal([2,3])
B = tf.fill([2,3],5.0)
C = tf.random_uniform([3,2])
D = tf.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-7.,-1.],[0.,5.,-2.]]))
print(sess.run(identity_matrix))
print(sess.run(D))
print(sess.run(tf.matmul(B,identity_matrix,transpose_a=False,transpose_b=False))) # 矩阵相乘
print(sess.run(tf.transpose(C))) # 矩阵转置
print(sess.run(tf.matrix_determinant(D))) # 矩阵行列式
print(sess.run(tf.matrix_inverse(D))) # 矩阵的逆
print(sess.run(tf.self_adjoint_eig(D)))# 第一行为特征值，剩下 的为特征向量,self_adjoint_eigvals可单独获得特征值

"""下面两行代表着创建一个计算图会话"""
import tensorflow as tf
sess = tf.Session()
print(sess.run(tf.div(3,4))) #=0 div返回值的数据类型与输入数据类型一致
print(sess.run(tf.truediv(3,4))) #=0.75,truediv在除法操作前强制转换整数为浮点数
print(sess.run(tf.floordiv(3.0,4.0))) # 对浮点数进行整数除法，返回浮点数结果，但是其会向下舍去小数位到最近的整数
print(sess.run(tf.mod(4,3)))# 取摸，返回除法余数
print(sess.run(tf.cross([2.,0.,0.],[1.,1.,1.]))) # 计算两个张量间的叉乘，叉乘函数只为三维向量定义，所以以两个三维张量作为输入
print(sess.run(tf.squared_difference([2.,0.,0.],[1.,1.,1.]))) #返回差值的平方
# 创建自定义函数
def custom_ploy(value):
    return(tf.subtract(3 * tf.square(value),value)+10)
print(sess.run(custom_ploy(11)))

def sigmoid(value):
    return (tf.truediv(1.,1.+tf.exp(-value)))
print(sess.run(sigmoid(0.))) # 输入0.0，而不是0

print(sess.run(tf.nn.relu([-3.,3.,10.]))) # 整流线型单元：max(0,x),对张量的每个元素进行操作；relu6=min(max(0,x),6)
"""激励函数"""
print(sess.run(tf.nn.sigmoid([-1.,0.,1.]))) # logistic函数
print(sess.run(tf.nn.tanh([-1.,0.,1.])))  # 双曲正切tanh函数
print(sess.run(tf.nn.softsign([-1.,0.,1.])))  # x/(abs(x)+1),符号函数的连续估计
print(sess.run(tf.nn.softplus([-1.,0.,1.])))  # log(exp(x)+1),ReLU激励函数的平滑版
print(sess.run(tf.nn.elu([-1.,0.,1.]))) # ELU激励函数 if x<0 exp(x)-1 else x

"""波士顿房价"""
import requests
housing_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
birth_file = requests.get(housing_url)
housing_header = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV0']
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>= 1] for y in housing_file.text.split('\n') if len(y) >= 1]
print(len(housing_data))

"""MINIST手写体字库"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print(len(mnist.train.images))  # mnist.train.images.shqpe=(55000,784)  28x28的图片

"""垃圾短信文本数据集"""
"""
 zipfile.ZipFile(file[, mode[, compression[, allowZip64]]])
 创建一个ZipFile对象，表示一个zip文件。参数file表示文件的路径或类文件对象(file-like object)；
 参数mode指示打开zip文件的模式，默认值为’r’，表示读已经存在的zip文件，也可以为’w’或’a’，
 ’w’表示新建一个zip文档或覆盖一个已经存在的zip文档，’a’表示将数据附加到一个现存的zip文档中。
 参数compression表示在写zip文档时使用的压缩方法，它的值可以是zipfile. ZIP_STORED 或zipfile. ZIP_DEFLATED。
 如果要操作的zip文件大小超过2G，应该将allowZip64设置为True

 ZipFile.read(name[, pwd]):获取zip文档内指定文件的二进制数据

 str和unicode都是basestring的子类。编码是指unicode-->str，解码是指str-->unicode
str是一个字节数组，这个字节数组表示的是对unicode对象编码(可以是utf-8、gbk、cp936、GB2312)后的存储的格式。这里它仅仅是一个字节流，没有其它的含义，
如果你想使这个字节流显示的内容有意义，就必须用正确的编码格式，解码显示。 对UTF-8编码的str'哈哈'使用len()函数时，结果是6，
因为实际上，UTF-8编码的'哈哈' == '\xe5\x93\x88\xe5\x93\x88'。
unicode才是真正意义上的字符串，对字节串str使用正确的字符编码进行解码后获得，例如'哈哈'的unicode对象为 u'\u54c8\u54c8' ,len(u”哈哈”) == 2
字符串在Python内部的表示是unicode编码，因此，在做编码转换时，通常需要以unicode作为中间编码，即先将其他编码的字符串解码（decode）成unicode，再从unicode编码（encode）成另一种编码。
decode的作用是将其他编码的字符串转换成unicode编码，如str1.decode('gb2312')，表示将gb2312编码的字符串str1转换成unicode编码。
encode的作用是将unicode编码转换成其他编码的字符串，如str2.encode('gb2312')，表示将unicode编码的字符串str2转换成gb2312编码。
因此，转码的时候一定要先搞明白，字符串str是什么编码，然后decode成unicode，然后再encode成其他编码
str.decode([encoding[, errors]])
使用encoding指示的编码，对str进行解码，返回一个unicode对象。默认情况下encoding是“字符串默认编码”，比如ascii。
errors指示如何处理解码错误，默认情况下是”strict”，也就是遇到解码错误时，直接抛出UnicodeError异常。
其他的errors值可以有”ignore”，”replace”等
https://blog.csdn.net/gqtcgq/article/details/47068817
 """
import requests
import io
from zipfile import ZipFile
zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
r = requests.get(zip_url)  # 返回<Response [200]>,  r.content的类型是<class 'bytes'>
z = ZipFile(io.BytesIO(r.content))   # z为 <zipfile.ZipFile file=<_io.BytesIO object at 0x0000000014801A98> mode='r'>; io.BytesIO实现了在内存中读写bytes
file = z.read("SMSSpamCollection")  # file是<class 'bytes'>：b'ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...\nham\t...
text_data = file.decode()   # text_data类型是：<class 'str'>
text_data = text_data.encode('ascii',errors='ignore') # 又为<class 'bytes'>
text_data = text_data.decode().split('\n')            # 为<class 'list'>
text_data = [x.split('\t') for x in text_data if len(x)>=1]  # list
text_data_target, text_data_train = [list(x) for x in zip(*text_data)]


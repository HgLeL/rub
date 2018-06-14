import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
sess = tf.Session()

start_time = time.time()
batch_size = 200         # 训练批次大小
output_every = 10        # 每迭代‘output_every’次输出一次结果
generations = 500      # 迭代次数
eval_every = 10         # 每迭代'eval_every'次对测试集进行评价
image_height = 100
image_width = 100
num_channels = 3
num_targets = 3
IMG_PIXELS = image_height * image_width * num_channels
# 指数级减少学习率
learning_rate = 0.1
lr_decay = 0.9
num_gens_to_wait = 250.

TRAIN_FILE = './train.tfrecords'
TEST_FILE = './test.tfrecords'

# 批量读取图片
def read_and_decode(filename_queue):
    #创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    #从文件中读出一个样例
    _,serialized_example = reader.read(filename_queue)
    #解析读入的一个样例
    features = tf.parse_single_example(serialized_example,features={
        'label':tf.FixedLenFeature([],tf.int64),
        'img_raw':tf.FixedLenFeature([],tf.string)
        })
    #将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['img_raw'],tf.uint8)
    label = tf.cast(features['label'],tf.int32)

    image.set_shape([IMG_PIXELS])
    image = tf.reshape(image,[image_height,image_width,num_channels])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image,label

#用于获取一个batch_size的图像和label
def inputs(data_set, batch_size, num_epochs):
    if not num_epochs:
        num_epochs = None
    if data_set == 'train':
        file = TRAIN_FILE
    else:
        file = TEST_FILE

    with tf.name_scope('input') as scope:
        filename_queue = tf.train.string_input_producer([file], num_epochs=num_epochs)
    image,label = read_and_decode(filename_queue)
    #随机获得batch_size大小的图像和label
    images,labels = tf.train.shuffle_batch([image, label],
        batch_size=batch_size,
        num_threads=64,
        capacity=100 + 3 * batch_size , # 队列的长度, min_after_dequeue + 3 * batch_size
        min_after_dequeue=100         # 出队后，队列至少剩下min_after_dequeue个数据，官方推荐设置(#threads+error_margin)*batch_size
    )
    return images,labels

# 声明模型函数
def cnn_model(input_images, batch_size, train_logical = True):
    def truncated_nomal_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape,dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.5)))

    def zero_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape,dtype=dtype, initializer=tf.constant_initializer(0.0)))

    # 第一卷积层
    with tf.variable_scope('conv1') as scope:
        ## 卷积核是5x5*64,创建64个特征
        conv1_kernel = truncated_nomal_var(name='conv_kernel1', shape=[5,5,3,64],dtype=tf.float32)
        ## 用步长为1来卷积图像
        conv1 = tf.nn.conv2d(input_images, conv1_kernel, strides=[1,1,1,1], padding='SAME')
        ## 初始化并增加偏置
        conv1_bias = zero_var(name='conv_bias1', shape=[64], dtype=tf.float32)
        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
        ## RELU激活函数
        relu_conv1 = tf.nn.relu(conv1_add_bias)
    # 第一池化层
    pool1 = tf.nn.max_pool(relu_conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool_layer1')
    # 局部响应归一化
    norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')

    # 第二层卷积层
    with tf.variable_scope('conv2') as scope:
        ## 卷积核大小仍为5x5*64
        conv2_kernel = truncated_nomal_var(name='conv_kernel2', shape=[5,5,64,64], dtype=tf.float32)
        ## 对先前的输出进行卷积
        conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1,1,1,1], padding='SAME')
        ## 初始化并增加偏置
        conv2_bias = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)
        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
        ## RELU激活函数
        relu_conv2 = tf.nn.relu(conv2_add_bias)
    # 第二池化层
    pool2 = tf.nn.max_pool(relu_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool_layer2')
    # 局部响应归一化
    norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
    # Reshape output into a single matrix for multiplication for the fully connected layers
    reshaped_output = tf.reshape(norm2, [batch_size,-1])
    reshaped_dim = reshaped_output.get_shape()[1].value

    # 第一全连接层
    with tf.variable_scope('full1') as scope:
        ## 第一全连接层有384个输出
        full_weigth1 = truncated_nomal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
        full_bias1 = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weigth1), full_bias1))
        # 第二全连接层，192个输出
        with tf.variable_scope('full2') as scope:
            full_weight2 = truncated_nomal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
            full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
            full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
    # 最后输出3个类别
    with tf.variable_scope('full3') as scope:
        full_weight3 = truncated_nomal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
        full_bias3 = zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
        final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)
    return final_output

# 创建损失函数
def loss(logits, targets):
    # get rid of extra dimensions and cast targets into integers
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # 计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    # 取平均交叉熵（一个批次）
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean

# 定义训练步骤函数
def train_step(loss_value, generation_num):
    # 学习率指数减少
    model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num, num_gens_to_wait, lr_decay, staircase=True)
    # 优化器
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    # 初始化train_step
    train_step = my_optimizer.minimize(loss_value)
    return train_step

# 创建批量图片的准确度函数
def accuracy_of_batch(logits, targets):
    # 确保targets是integers, 并且进行降维
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # 根据选择最大概率值，给出预测类别
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # 检查与真实是否相等
    predicted_correctly = tf.equal(batch_predictions, targets)
    # average (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy

# 利用inputs函数开始训练
images, targets = inputs(data_set='train', batch_size=batch_size, num_epochs=None)
test_images, test_targets = inputs(data_set='test', batch_size=batch_size, num_epochs=None)

# 初始化训练模型
with tf.variable_scope('model_definition') as scope:
    # 声明训练模型
    model_output = cnn_model(images, batch_size)
    # use same variable within scope
    scope.reuse_variables()
    # 声明测试模型
    test_output = cnn_model(test_images, batch_size)

# 初始化损失函数和测试准确度函数，并声明迭代变量，该迭代变量需要声明为非训练型变量，并传入训练函数，用于计算学习率的指数级衰减值
loss = loss(model_output, targets)
accuracy = accuracy_of_batch(test_output, test_targets)
generation_num = tf.Variable(0, trainable=False)
train_op = train_step(loss, generation_num)

# 初始化所有模型变量，然后运行start_queue_runners()函数启动图像管道
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

# 现在遍历迭代训练，保存训练集损失函数和测试集准确度
train_loss = []
test_accuracy = []
for i in range(generations):
    _, loss_value = sess.run([train_op, loss])
    if (i+1) % output_every == 0:
        train_loss.append(loss_value)
        output = '迭代 {}: Loss = {:.5f}'.format((i+1), loss_value)
        print(output)
    if (i+1) % eval_every == 0:
        [temp_accuracy] = sess.run([accuracy])
        acc_output = ' --- 测试集准确率 = {:.3f}%.'.format(100.*temp_accuracy)
        print(acc_output)
    print('第%d次迭代，已耗时 %d 秒' % (i,int(time.time()-start_time)))

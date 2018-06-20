# resize图片，并转化为numpy格式，这里是全部将数据载入内存
import os
import numpy as np
from PIL import Image
def load_data(img_path, label_path, seed=1):
    """path：训练集文件夹地址、测试集文件夹地址
       seed:将数据打乱的随机种子"""

    # 以下两行代码，可以给出文件夹中排序后的图片名称
    file_name = os.listdir(img_path)   # 元素格式为：××.jpg
    file_name = sorted([int(''.join(list(x)[:-4])) for x in file_name])

    count = len(file_name)    # count张图片
    image_width = 100         # 需要resize的尺寸
    image_height = 100
    data = np.empty((count, 3, image_width, image_height), dtype='float32')
    for i in range(count):
        img = Image.open(img_path + str(file_name[i]) +'.jpg').resize((image_width,image_height))
        arr = np.array(img, dtype='float32')
        data[i,:,:,:] = [arr[:,:,0], arr[:,:,1], arr[:,:,2]]
    label = np.loadtxt(label_path)

    # 将数据集顺序打乱
    np.random.seed(seed=seed)
    random_index = list(np.random.choice(count, size=count, replace=False))
    data = data[random_index,:,:,:]
    label = label[random_index]
    return data, label

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
import keras
from keras.utils import plot_model    # 此模块用于模型可视化

# 导入数据，对多分类标签one hot编码，建立模型
data, label = load_data(img_path='/home/iqx/文档/项目/cnn_6.16_640x480/',label_path='/home/iqx/文档/项目/label2.txt')
label = keras.utils.to_categorical(label, num_classes=3)
model = Sequential()

# 第一卷积层（每经过一个池化层，卷基层的卷积核个数都会乘2）
model.add(Conv2D(32,(3,3),padding='same', input_shape=data.shape[1:]))
model.add(Activation('relu'))
# 第二卷积层
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128,(5,5), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128,(5,5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(1024))
model.add(Dense(3))
model.add(Activation('softmax'))

opt = keras.optimizers.adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
data /= 255      # 归一化
model.fit(data, label, batch_size=64, epochs=100, verbose=1, shuffle=True)
score = model.evaluate(data, label, verbose=1, steps=10)
# 模型可视化
plot_model(model, to_file='/home/iqx/文档/项目/model.jpg')
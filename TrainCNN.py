"""
模型训练
"""
# ########### #
# 导入必要的模块 #
# ########### #
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Softmax
from tensorflow.keras.applications import ResNet50, MobileNet
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ########### #
# 训练前参数定义 #
# ########### #
epochs = 10      # epoch数
model_backbone = "mobilenet"     # resnet50, mobilenet, inception_v3

# ################ #
# 搭建卷积神经网络模型 #
# ################ #
if model_backbone == 'resnet50':
    # 构建基本层，使用已经存在的模型
    base_model = ResNet50(weights='imagenet', include_top=False)   # 导入使用imagenet训练后的网络作为基础网络，不包含顶层

    # 搭建适合自己数据集的模型顶层
    x = base_model.output               # 一个包含输出张量的列表
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(1024)(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(53)(x)                    # 加入全连接层
    preds = Softmax()(x)                # 最后一层使用softmax激活函数

    # 搭建完整模型
    model = Model(inputs=base_model.input, outputs=preds)

elif model_backbone == 'mobilenet':
    base_model = MobileNet(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(53, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=preds)

# ########################## #
# 数据集图像增强，预处理训练数据集 #
# ########################## #
train_datagen = ImageDataGenerator(rotation_range=30,       # 数据提升时图片随机转动的角度
                                   width_shift_range=0.2,   # 数据提升时图片随机水平偏移的幅度
                                   height_shift_range=0.2,  # 数据提升时图片随机竖直偏移的幅度
                                   shear_range=0.2,         # 用来进行剪切变换的程度
                                   zoom_range=0.2,          # 随机缩放的幅度
                                   horizontal_flip=True,    # 进行随机水平翻转
                                   preprocessing_function=preprocess_input)

# ############################################################# #
# 以文件夹路径为参数,生成经过预处理后的数据,在一个无限循环中无限产生batch数据 #
# ############################################################# #
train_generator = train_datagen.flow_from_directory('./train/',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

# 打印图像生成器的类索引
print(train_generator.class_indices)
# 打印图像生成器所产生图像的路径
print(train_generator.filenames)

# 图像生成器中图像的总数
total_train = train_generator.n

# 图像生成器的类索引
class_indices = train_generator.class_indices
# 转换类索引字典的键和值
inverse_dict = dict((val, key) for key, val in class_indices.items())
# 将该字典写入json文件中
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# Adam优化器; 损失函数将是分类交叉熵; 评估指标将是准确性
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 使用逐批生成的数据，按批次训练模型。
step_size_train = np.ceil(train_generator.n / train_generator.batch_size)

# 数据使用图像生成器产生，使用fit_generator进行训练
H = model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        epochs=epochs)

# 保存模型结构和权重
model.save(model_backbone+'_model.h5')

# 画图
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(model_backbone+"_plot.png")
plt.show()


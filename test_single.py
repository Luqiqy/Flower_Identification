"""
对单张图片进行预测分类
"""
# 导入相应的模块
import json
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

model_h5 = 'resnet50_model.h5'  # resnet50_model.h5, mobilenet_model.h5

# 加载模型结构以及模型权重
model = keras.models.load_model('./'+model_h5)

# 测试数据预处理
test_path = './test/荷花/image_01876.jpg'    # 根据最终路径打开图像

img = Image.open(test_path)                 # 根据路径打开图像
img = img.resize((224, 224))                # 将图像进行resize
img = np.array(img).astype(np.float32)      # 将图像转换为numpy数组
img = preprocess_input(img)                 # 对图像进行和训练时相同的预处理
img = (img.reshape(1, 224, 224, 3))         # 将图像进行reshape以满足模型输入

result = model.predict(img)                     # 对图像进行预测，result的shape为(1, 53)
prediction = np.squeeze(result)                 # 从result中删除单维度条目，即把其shape中为1的维度去掉
predict_class = np.argmax(result)               # 取result中最大数的索引值
json_file = open('./class_indices.json', 'r')
class_indict = json.load(json_file)             # class_indict为字典，键为索引值，值为花卉种类

print(class_indict[str(predict_class)], prediction[predict_class])


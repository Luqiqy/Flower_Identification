"""
对图像进行随机抽取进行批量预测，并且统计预测对的个数，计算正确率
"""
# 导入相应的模块
import os
import random
import json
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input


model_h5 = 'resnet50_model.h5'  # resnet50_model.h5, mobilenet_model.h5

# 加载模型结构以及模型权重
model = keras.models.load_model('./'+model_h5)

# 对图像进行预处理并且预测
test_root = './test/'           # 测试图片所在的根路径
_name = os.listdir(test_root)   # 将此文件下的文件夹名存入列表,即存入类别名称

total_num = 500                # 进行多少次随机抽取预测
right_num = 0                   # 保存预测正确的次数
ls = []

# 开始进行随机抽取预测
for i in range(total_num):
    random.shuffle(_name)                       # 将_name列表进行随机打乱
    test_classes = test_root + _name[0] + '/'   # 抽取打乱后的列表的第一项类别，完善路径
    name_ = os.listdir(test_classes)            # 将该类别下的文件名存入列表name_
    random.shuffle(name_)                       # 将name_列表进行随机打乱
    test_path = test_classes + name_[0]         # 抽取打乱后的列表的第一个文件，扩充为为测试文件的最终路径

    img = Image.open(test_path)                 # 根据最终路径打开图像
    img = img.resize((224, 224))                # 将图像进行resize
    img = np.array(img).astype(np.float32)      # 将图像转换为numpy数组
    img = preprocess_input(img)                 # 对图像进行和训练时相同的预处理
    img = (img.reshape(1, 224, 224, 3))         # 将图像进行reshape以满足模型输入

    result = model.predict(img)                         # 对图像进行预测
    prediction = np.squeeze(result)
    predict_class = np.argmax(result)
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
    if _name[0] == class_indict[str(predict_class)]:
        right_num += 1
    else:
        ls.append(_name[0])
    print(test_path)
    print(class_indict[str(predict_class)], prediction[predict_class])
    print("###############################################################")

print("预测正确的个数：", right_num)
print("进行预测的总个数：", total_num)
print("预测正确率：", right_num/total_num*100, end='%\n')
print("错误的种类：", set(ls))

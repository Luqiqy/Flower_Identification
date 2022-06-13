# 文件名称及功能详情 
1. test中存放的是花卉测试数据集的图像（自己创建）
2. train中存放的是花卉训练数据集的图像（自己创建）
3. TrainCNN.py是进行网络训练的代码
4. test_single.py是单张图片进行预测的代码
5. test_random.py是循环随机抽取测试集图片并打印整体准确率和错误种类的代码
6. TK.py是GUI软件的源代码
7. class_indices.json中是数据标签索引（resnet_50.py运行后自行生成）
8. .h5文件是网络模型的框架结构以及参数（resnet_50.py运行后自行生成）

# 使用指南
1. 首先确保数据集的完备，可无test测试数据集
2. 运行TrainCNN.py文件，运行结束会产生对应的.h5权重文件以及训练过程中Loss/Acc曲线
3. 在有test测试数据集的情况下，选用test_single.py或者test_random.py进行测试

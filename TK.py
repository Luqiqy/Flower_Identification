import time
from os import listdir
from random import shuffle
from json import load as j_load
import numpy as np
from tkinter import Tk, Toplevel, Label, Button, Entry, StringVar
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tkinter.filedialog import askopenfilename
from tensorflow.keras.applications.resnet50 import preprocess_input


root = Tk()
root.title('花卉识别')
root.geometry('500x250')
root.resizable(0, 0)

# 主界面分区，分为功能按键和图像显示两个区域
title_label1 = Label(root, text="功能按键👇", font=('微软雅黑', 8)).place(x=17, y=2)
title_label2 = Label(root, text="图像显示👇", font=('微软雅黑', 8)).place(x=259, y=0)

# 加载模型结构以及模型权重
model_h5 = 'resnet50_model.h5'  # resnet50_model.h5, mobilenet_model.h5
model = load_model('./'+model_h5)


def image_plot(image):
    img_jpg = ImageTk.PhotoImage(image)
    image_label.config(image=img_jpg)
    image_label.image = img_jpg


# 图像显示区域就是对图像进行显示（到时候把导入图像部分拿出去）
def show_result_acc(result, acc):
    result_var = StringVar()
    acc_var = StringVar()
    result_var.set(result)
    acc_var.set(acc)
    Label(root, textvariable=result_var, font=('黑体', 16), width=10, anchor='w').place(x=133, y=130)
    Label(root, textvariable=acc_var, font=('黑体', 16), width=10, anchor='w').place(x=133, y=170)


def model_pred_random(image_path):
    """
    功能：实现对输入图片的预测，显示出图像并且打印出种类和准确率
    输入：待预测图片的路径
    输出：
    """
    start = time.perf_counter()
    # 进行图像预处理
    image = Image.open(image_path)      # 根据最终路径打开图像
    image = image.resize((224, 224))    # 将图像进行resize

    # 绘图
    image_plot(image)

    img = np.array(image).astype(np.float32)  # 将图像转换为numpy数组
    img = preprocess_input(img)               # 对图像进行和训练时相同的预处理
    img = (img.reshape(1, 224, 224, 3))       # 将图像进行reshape以满足模型输入

    # 对图像进行预测
    result = model.predict(img)                     # 对图像进行预测
    prediction = np.squeeze(result)                 
    predict_class = np.argmax(result)
    json_file = open('./class_indices.json', 'r')
    class_indict = j_load(json_file)
    pre_result = class_indict[str(predict_class)]
    pre_acc = prediction[predict_class]

    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    # 在GUI显示预测结果和准确率
    show_result_acc(pre_result, pre_acc)


def model_pred_input(image_path):
    """
    功能：实现对输入图片的预测，显示出图像并且打印出种类和准确率
    输入：待预测图片的路径
    输出：
    """

    warn1 = '请阅读说明'
    warn2 = '-.--------'

    start = time.perf_counter()

    # 进行图像预处理
    image = Image.open(image_path)      # 根据最终路径打开图像
    image = image.resize((224, 224))    # 将图像进行resize

    # 绘图
    image_plot(image)

    img = np.array(image).astype(np.float32)  # 将图像转换为numpy数组
    img = preprocess_input(img)               # 对图像进行和训练时相同的预处理
    img = (img.reshape(1, 224, 224, 3))       # 将图像进行reshape以满足模型输入

    # 对图像进行预测
    result = model.predict(img)
    prediction = np.squeeze(result)
    predict_class = np.argmax(result)
    json_file = open('./class_indices.json', 'r')
    class_indict = j_load(json_file)
    pre_result = class_indict[str(predict_class)]
    pre_acc = prediction[predict_class]

    end = time.perf_counter()
    # 在GUI显示预测结果和准确率
    if pre_acc > 0.65:
        print('Running time: %s Seconds' % (end - start))
        show_result_acc(pre_result, pre_acc)
    else:
        show_result_acc(warn1, warn2)


def instruction():
    window_instruction = Toplevel(root)
    window_instruction.title('使用说明')
    window_instruction.geometry('300x300')
    window_instruction.resizable(0, 0)

    Label(window_instruction, text='使用说明\n'
                                   '---------\n'
                                   '本程序有两种模式，模式一为随机抽取，模式二为指定输入\n'
                                   '①随机抽取：在本地测试集中随机抽取进行识别\n'
                                   '②指定输入：将用户想识别的花卉图片读入进行识别\n'
                                   '\n'
                                   '特别说明\n'
                                   '---------\n'
                                   '①目前模型所能识别的花卉种类有限，具体种类见“花卉指南”\n'
                                   '②花朵本身在花卉图片中所占有图像的比例越大，识别成功概率越高\n',
          height=300,
          width=300,
          wraplength=299,
          justify='left',
          anchor='nw').pack()


def flowers():
    window_flowers = Toplevel(root)
    window_flowers.title('花卉指南')
    window_flowers.geometry('300x300')
    window_flowers.resizable(0, 0)

    Label(window_flowers, text='花卉是指具有观赏价值的草本植物\n'
                               '\n'
                               '本程序能够识别分类的花卉种类如下：\n'
                               '一品红, 万寿菊, 三色菫, 仙客来, 六出花, 凌霄花, 勋章菊, 叶子花, 向日葵, 唐菖蒲, 嘉兰, 大丽花, '
                               '天人菊, 天竺葵, 射干花, 山茶花, 康乃馨, 报春花, 旱金莲, 朱顶红, 松果菊, 果子蔓, 桂竹香, 桔梗花, '
                               '款冬花, 牵牛花, 玉兰, 玫瑰, 番红花, 睡莲, 硬叶兜兰, 秋英花, 耧斗菜, 花烛, 花葵, 荷花, 葡萄风信子, '
                               '蒲公英花, 蓝目菊, 虎皮百合, 贝母花, 金盏花, 铁线莲, 雏菊, 非洲菊, 风铃花, 马蹄莲, 鸡蛋花, 鹤望兰, '
                               '黄水仙, 黄花鸢尾, 黑心金光菊, 龙胆花',
          height=300,
          width=300,
          wraplength=300,
          justify='left',
          anchor='nw').pack()


def mode_random():
    # 抽取图像
    test_root = './test/'                       # 测试图片所在的根路径
    _name = listdir(test_root)                  # 将此文件下的文件夹名存入列表,即存入类别名称
    shuffle(_name)                              # 将_name列表进行随机打乱
    test_classes = test_root + _name[0] + '/'   # 抽取打乱后的列表的第一项类别，完善路径
    name_ = listdir(test_classes)               # 将该类别下的文件名存入列表name_
    shuffle(name_)                              # 将name_列表进行随机打乱
    test_path = test_classes + name_[0]         # 抽取打乱后的列表的第一个文件，扩充为为测试文件的最终路径

    # 进行图像预测显示
    model_pred_random(test_path)


def mode_input():
    path_ = askopenfilename()
    path.set(path_)
    model_pred_input(e1.get())  # 根据最终路径打开图像


path = StringVar()

# 功能按键区主要是使用说明，模式选择和预测结果
b1 = Button(root, text="使用说明", width=10, relief='groove', command=instruction).place(x=25, y=25)
b2 = Button(root, text="花卉指南", width=10, relief='groove', command=flowers).place(x=133, y=25)
b3 = Button(root, text="随机抽取", width=10, command=mode_random).place(x=25, y=65)
b4 = Button(root, text="指定输入", width=10, command=mode_input).place(x=133, y=65)
l1 = Label(root, text="识别结果：", font=('黑体', 16)).place(x=25, y=130)
l2 = Label(root, text="准确率：", font=('黑体', 16)).place(x=25, y=170)

e1 = Entry(root, state='readonly', text=path)
image_label = Label(root)
image_label.place(x=259, y=20)

mode_random()

root.mainloop()

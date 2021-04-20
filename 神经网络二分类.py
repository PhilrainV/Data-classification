# 导入模块
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pylab as plt


# 创建加载数据读取数据以及划分数据集的函数，返回数据特征集以及数据标签集
# def loaddataset(filename):
#     fp = open(filename)  # （299，22）
#
#     # 存放数据
#     dataset = []
#
#     # 存放标签
#     labelset = []
#     for i in fp.readlines():  # 按照行来进行读取，每次读取一行，一行的数据作为一个元素存放在了类别中
#         a = i.strip().split()  # 去掉每一行数据的空格以及按照默认的分隔符进行划分
#
#         # 每个数据行的最后一个是标签
#         dataset.append([float(j) for j in a[:len(a) - 1]])  # 读取每一行中除最后一个元素的前面的元素，并且将其转换为浮点数
#         labelset.append(int(float(a[-1])))  # 读取每一行的最后一个数据作为标签数据
#     return dataset, labelset  # dataset是（299,21）的列表，labelset是（299,1）的列表
def loaddataset(input_path, debug=True):
    """
    Read nasdaq stocks data.

    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    df = pd.read_csv(input_path, nrows=250 if debug else None)
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'answered_correctly']].values
    y = np.array(df.answered_correctly)

    return X, y

# x为输入层神经元个数，y为隐层神经元个数，z输出层神经元个数
# 创建的是参数初始化函数，参数有各层间的权重weight和阈值即偏置value就是b
# 本例的x,y=len(dataset[0])=22，z=1
def parameter_initialization(x, y, z):
    # 隐层阈值
    value1 = np.random.randint(-5, 5, (1, y)).astype(np.float64)  # 随机生成（-5，5）之间的整数组成（1，y）的数组，然后再将其转为浮点数显示

    # 输出层阈值
    value2 = np.random.randint(-5, 5, (1, z)).astype(np.float64)

    # 输入层与隐层的连接权重
    weight1 = np.random.randint(-5, 5, (x, y)).astype(np.float64)

    # 隐层与输出层的连接权重
    weight2 = np.random.randint(-5, 5, (y, z)).astype(np.float64)

    return weight1, weight2, value1, value2


# 创建激活函数sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


'''
weight1:输入层与隐层的连接权重
weight2:隐层与输出层的连接权重
value1:隐层阈值
value2:输出层阈值
权重和阈值的个数和神经网络的隐层层数有关，若隐层为n，则权重和阈值的个数为n+1
'''


# 创建训练样本的函数，返回训练完成后的参数weight和value，这里的函数是经过一次迭代后的参数，即所有的样本经过一次训练后的参数
# 具体参数的值可以通过设置迭代次数和允许误差来进行确定
def trainning(dataset, labelset, weight1, weight2, value1, value2):
    # x为步长
    x = 0.01  # 学习率
    for i in range(len(dataset)):  # 依次读取数据特征集中的元素，一个元素即为一个样本所含有的所有特征数据

        # 输入数据
        # （1,21）
        inputset = np.mat(dataset[i]).astype(np.float64)  # 每次输入一个样本，将样本的特征转化为矩阵，以浮点数显示

        # 数据标签
        # （1，1）
        outputset = np.mat(labelset[i]).astype(np.float64)  # 输入样本所对应的标签

        # 隐层输入，隐层的输入是由输入层的权重决定的，wx
        # input1：（1，21）.（21，21）=（1，21）
        input1 = np.dot(inputset, weight1).astype(np.float64)

        # 隐层输出，由隐层的输入和阈值以及激活函数决定的，这里的阈值也可以放在输入进行计算
        # sigmoid（（1，21）-（1，21））=（1，21）
        output2 = sigmoid(input1 - value1).astype(np.float64)

        # 输出层输入，由隐层的输出
        # （1，21）.（21，1）=（1，1）
        input2 = np.dot(output2, weight2).astype(np.float64)

        # 输出层输出，由输出层的输入和阈值以及激活函数决定的，这里的阈值也可以放在输出层输入进行计算
        # （1，1）.（1，1）=（1，1）
        output3 = sigmoid(input2 - value2).astype(np.float64)

        # 更新公式由矩阵运算表示
        # a:(1,1)
        a = np.multiply(output3, 1 - output3)  # 输出层激活函数求导后的式子，multiply对应元素相乘，dot矩阵运算
        # g:(1,1)
        g = np.multiply(a, outputset - output3)  # outputset - output3：实际标签和预测标签差
        # weight2:(21,1),np.transpose(weight2):(1,21),b:(1,21)
        b = np.dot(g, np.transpose(weight2))
        # (1,21)
        c = np.multiply(output2, 1 - output2)  # 隐层输出激活函数求导后的式子，multiply对应元素相乘，dot矩阵运算
        # (1,21)
        e = np.multiply(b, c)

        value1_change = -x * e  # （1，21）
        value2_change = -x * g  # （1，1）
        weight1_change = x * np.dot(np.transpose(inputset), e)  # （21，21）
        weight2_change = x * np.dot(np.transpose(output2), g)  # （21，1）

        # 更新参数，权重与阈值的迭代公式
        value1 += value1_change
        value2 += value2_change
        weight1 += weight1_change
        weight2 += weight2_change
    return weight1, weight2, value1, value2


# 创建测试样本数据的函数
def testing(dataset1, labelset1, weight1, weight2, value1, value2):
    # 记录预测正确的个数
    rightcount = 0
    probs = []
    for i in range(len(dataset1)):
        # 计算每一个样例的标签通过上面创建的神经网络模型后的预测值
        inputset = np.mat(dataset1[i]).astype(np.float64)
        outputset = np.mat(labelset1[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = sigmoid(np.dot(output2, weight2) - value2)

        # 确定其预测标签
        if output3 > 0.5:
            flag = 1
        else:
            flag = 0
        if labelset1[i] == flag:
            rightcount += 1
        # 输出预测结果
        probs.append(flag)
        #print("预测为%d   实际为%d" % (flag, labelset1[i]))
    # 返回正确率
    return rightcount / len(dataset1) , probs

def calcAUC_byRocArea(labels,preds):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total_pair = n_pos * n_neg

    labels_preds = zip(labels, preds)
    labels_preds = sorted(labels_preds, key=lambda x: x[1])
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(len(labels_preds)):
        if labels_preds[i][0] == 1:
            satisfied_pair += accumulated_neg
        else:
            accumulated_neg += 1

    return satisfied_pair / float(total_pair)

def main():
    # 读取训练样本数据并且进行样本划分
    dataset, labelset = loaddataset("E:\Python\Project\成绩预测网络\实验/10万.csv")
    dataset = dataset[:,1:]
    # 读取测试样本数据并且进行样本划分
    dataset1, labelset1 = loaddataset("E:\Python\Project\成绩预测网络\实验\简单数据.csv")
    dataset1 = dataset1[:, 1:]
    # 得到初始化的待估参数的值
    weight1, weight2, value1, value2 = parameter_initialization(len(dataset[0]), len(dataset[0]), 1)
    pronum = []
    AUC = []
    acc = []
    # 迭代次数为1500次，迭代次数一般越大准确率越高，但是其运行时间也会增加
    for epoch in tqdm(range(100000)):
        for i in range(10):
            # 获得对所有训练样本训练迭代一次后的待估参数
            weight1, weight2, value1, value2 = trainning(dataset, labelset, weight1, weight2, value1, value2)
    # 对测试样本进行测试，并且得到正确率
            rate = testing(dataset1, labelset1, weight1, weight2, value1, value2)
            pronum = rate[1]
            acc.append(rate[0])
            AUC.append(calcAUC_byRocArea(labelset1, pronum))
            #print(calcAUC_byRocArea(labelset1,rate[1]))
            #print("正确率为%f" % (rate[0]))

    picture = plt.figure()
    plt.ylim(0, 1)
    plt.plot(AUC, label="AUC")
    plt.plot(acc, label="ACC")
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
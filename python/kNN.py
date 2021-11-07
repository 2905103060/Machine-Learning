import numpy as np
import timeit

from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
#学习曲线的绘制
from sklearn.model_selection import learning_curve
import matplotlib as mplmpl
mplmpl.use('Agg')
import matplotlib.pyplot as plt

import struct



#计算准确率
from sklearn.metrics import accuracy_score
#计算混淆矩阵
from sklearn.metrics import confusion_matrix


#引入sklearn的k近邻算法库
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

TRAIN_ITMES = 10000
TEST_ITEMS = 2000


def loadMnistData():
    mnist_data = []
    for img_file, label_file, items in zip(
            ['dataset/train-images.idx3-ubyte', 'dataset/t10k-images.idx3-ubyte'],
            ['dataset/train-labels.idx1-ubyte', 'dataset/t10k-labels.idx1-ubyte'],
            [TRAIN_ITMES, TEST_ITEMS]):
        # 打开一个文件并进行读取操作，读取的内容放在缓冲区中，read()表示全部读取
        data_img = open(img_file, 'rb').read()
        data_label = open(label_file, 'rb').read()
        # fmt of struct unpack, > means big endian, i means integer, well, iiii mean 4 integers
        # '>iiii'是说使用大端法读取4个unsinged int32
        fmt = '>iiii'
        offset = 0
        magic_number, img_number, height, width = struct.unpack_from(fmt, data_img, offset)
        print('magic number is {}, image number is {}, height is {} and width is {}'.format(magic_number, img_number,
                                                                                            height, width))
        # slide over the 2 numbers above
        offset += struct.calcsize(fmt)
        # 28x28
        image_size = height * width
        # B means unsigned char
        fmt = '>{}B'.format(image_size)
        # because gemfield has insufficient memory resource
        if items > img_number:
            items = img_number
        images = np.empty((items, image_size))
        for i in range(items):
            images[i] = np.array(struct.unpack_from(fmt, data_img, offset))
            # 0~255 to 0~1
            images[i] = images[i] / 256
            offset += struct.calcsize(fmt)

        # fmt of struct unpack, > means big endian, i means integer, well, ii mean 2 integers
        fmt = '>ii'
        offset = 0
        magic_number, label_number = struct.unpack_from(fmt, data_label, offset)
        print('magic number is {} and label number is {}'.format(magic_number, label_number))
        # slide over the 2 numbers above
        offset += struct.calcsize(fmt)
        # B means unsigned char
        fmt = '>B'
        # because gemfield has insufficient memory resource
        if items > label_number:
            items = label_number
        labels = np.empty(items)
        for i in range(items):
            labels[i] = struct.unpack_from(fmt, data_label, offset)[0]
            offset += struct.calcsize(fmt)

        mnist_data.append((images, labels.astype(int)))

    return mnist_data

if __name__ == "__main__":

    training_data, test_data = loadMnistData()
    train_data  = training_data[0][0:10000, :]
    train_lable = training_data[1][0:10000]

    # #对knn算法里面的k参数进行网格搜索调优
    # k_range = []
    # for i in range(10):
    #     if (i%2)==1 :
    #         k_range.append(i)
    # #k_range    [1, 3, 5, 7, 9]
    # knn = KNeighborsClassifier()
    # param_grid = dict(n_neighbors = k_range)
    # gs = GridSearchCV(knn, param_grid,refit=True, cv=10, verbose=1, n_jobs=-1)
    # gs.fit(train_data,train_lable)
    # print('最优参数: ', gs.best_params_)
    # print('最佳性能: ', gs.best_score_)

    #knn = KNeighborsClassifier(n_neighbors=1)
    knn = KNeighborsClassifier()
    knn.fit(training_data[0], training_data[1])
    # 预测
    pred_result = knn.predict(test_data[0])
    print(accuracy_score(test_data[1], pred_result))
    print(confusion_matrix(test_data[1], pred_result))


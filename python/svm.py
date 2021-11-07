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

def get_train_num():
    start_time = timeit.default_timer()
    training_data, test_data = loadMnistData()
    #对不同训练集数据量的分类精度进行测量
    sample_sum = []
    score = []
    for i in range(15):
        sample_sum.append(500+i*500)

    for i  in sample_sum:
        train_data  = training_data[0][0:i, :]
        train_lable = training_data[1][0:i]

        clf = svm.SVC()
        scores = cross_val_score(clf, train_data, train_lable, cv=10,scoring='accuracy',n_jobs=10)
        scores = np.mean(scores)
        score.append(scores)
    plt.figure('Fig1')
    plt.plot(sample_sum,score,'ro-',label='accuracy')

    plt.grid()
    plt.xlabel('train size')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.title('SVM')
    plt.savefig('train-number-size.png')
    plt.show()
    #[0.89, 0.9099999999999999, 0.9199999999999999, 0.9275000000000002, 0.9363999999999999, 0.9416666666666667,
    # 0.9442857142857145, 0.9477499999999999, 0.9484444444444444, 0.9513999999999999]


def forwardWithSVM():
    start_time = timeit.default_timer()
    training_data, test_data = loadMnistData()

    #start

    # #对不同训练集数据量的分类精度进行测量
    # sample_sum = []
    # score = []
    # for i in range(15):
    #     sample_sum.append(500+i*500)
    #
    # for i  in sample_sum:
    #     train_data  = training_data[0][0:i, :]
    #     train_lable = training_data[1][0:i]
    #
    #     clf = svm.SVC()
    #     scores = cross_val_score(clf, train_data, train_lable, cv=10,scoring='accuracy',n_jobs=10)
    #     scores = np.mean(scores)
    #     score.append(scores)
    # plt.figure('Fig1')
    # plt.plot(sample_sum,score,'ro-',label='accuracy')
    #
    # plt.grid()
    # plt.xlabel('train size')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')
    # plt.title('SVM')
    # plt.savefig('train-number-size.png')
    # plt.show()
    # #[0.89, 0.9099999999999999, 0.9199999999999999, 0.9275000000000002, 0.9363999999999999, 0.9416666666666667,
    # # 0.9442857142857145, 0.9477499999999999, 0.9484444444444444, 0.9513999999999999]



    # #默认参数下的准确率和混淆矩阵  0.9445
    # clf = svm.SVC()
    # clf.fit(training_data[0], training_data[1])
    #
    # test_result = clf.predict(test_data[0])
    # print(accuracy_score(test_data[1], test_result))
    #
    # print(confusion_matrix(test_data[1], test_result))

    #确定使用哪一个核函数
    clf = svm.SVC()
    parameters = {'kernel':['linear','poly','rbf','sigmoid']}
    gs = GridSearchCV(clf, parameters, scoring='accuracy', refit=True, cv=5, verbose=1, n_jobs=5)
    gs.fit(training_data[0],training_data[1] )  # Run fit with all sets of parameters.
    print('最优参数: ', gs.best_params_)
    print('最佳性能: ', gs.best_score_)
    means = gs.cv_results_['mean_test_score']
    params = gs.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (mean, param))

    # #网格搜索调优
    # clf = svm.SVC()
    # c_range = np.linspace(0,5,6)
    # print(c_range)
    # gamma_range = np.logspace(-10,1,20)
    # param_grid = dict(C = c_range,
    #                   gamma = gamma_range)
    # gs = GridSearchCV(clf, param_grid,refit=True, cv=5, verbose=1, n_jobs=-1)
    # gs.fit(training_data[0],training_data[1] )
    # print('最优参数: ', gs.best_params_)
    # print('最佳性能: ', gs.best_score_)

    # #默认参数下的准确率和混淆矩阵  0.9445
    # #clf = svm.SVC(C=2.0,gamma = 0.012742749857031322)  #0.954
    # #clf = svm.SVC(C=2.0) #0.9535
    # clf = svm.SVC() #0.9445
    # clf.fit(training_data[0], training_data[1])
    # test_result = clf.predict(test_data[0])
    # print(accuracy_score(test_data[1], test_result))
    # print(confusion_matrix(test_data[1], test_result))


if __name__ == "__main__":
    forwardWithSVM()

# coding: utf-8
import gdal
import cv2
import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_image_datasets(path):
    """
    :param path: 測試圖像集路徑
    :return: 所有測試圖像數據，所有測試圖像名稱
    """
    datasets = []
    name_datasets = []
    for k, filename in enumerate(os.listdir(path)):
        # if k > 5:
        #     break
        if filename[0] != '.':
            image = load_test_image(os.path.join(path, filename))
            datasets.append(image)
            name_datasets.append(filename)
    return np.array(name_datasets), np.array(datasets)

def load_test_image(img_path):
    """
    :param img_path:測試單張圖像的路徑
    :return:測試圖像
    """
    img = gdal.Open(img_path)
    try:
        im_width = img.RasterXSize
        im_height = img.RasterYSize
        dataset = img.ReadAsArray(0, 0, im_width, im_height)
    except:
        print(1)
    # t = np.array(dataset)
    # t = np.ceil((t / np.max(t))*255)
    # plt.imshow(t)
    return np.array(dataset)

def segnet_predict(test_image_path, predict_label_path):
    """
    :param test_path:測試集的路徑
    :return: 是否成功預測完所有圖像
    """
    # 定义测试区域的大小
    image_size = 256
    # 定义类别标签
    classes = [0., 1., 2., 3., 4., 5., 6., 7.]
    labelencoder = LabelEncoder()
    labelencoder.fit(classes)
    path = './myweb/Detector/segnet/model_256_matlab.H5'
    model = load_model(os.path.abspath(path))

    test_image_name, test_image_ori = load_image_datasets(test_image_path)

    for index, test_image in enumerate(test_image_ori):
        _, test_image_height, test_image_width = test_image.shape
        mask = np.zeros((test_image_height, test_image_width), dtype=np.uint8)
        test_image = np.expand_dims(test_image, axis=0)
        predict = model.predict_classes(test_image, verbose=2)
        predict = labelencoder.inverse_transform(predict[0])
        predict = predict.reshape((image_size, image_size)).astype(np.uint8)

        mask = predict

        cv2.imwrite(predict_label_path + '{}.png'.format(test_image_name[index][:-4]), mask)

    return 0


def main():


    img_path = '../temp/images/'
    label_path = '../temp/labels/'
    segnet_predict(img_path, label_path)
    return 0

if __name__ == '__main__':
    main()
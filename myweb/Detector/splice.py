
from skimage import io, exposure, morphology, measure
import numpy as np
import scipy.io as sio
import math
import os
import tkinter as tk
import tkinter.filedialog as tkfd
from scipy.misc import imread
from PIL import Image
import cv2
import gdal

def splice(path, save_path):
    def get_filename(path, filetype):
        name = []
        final_name = []
        for root, dirs, files in os.walk(path):
            for i in files:
                if filetype in i:
                    name.append(i.replace(filetype, ''))
        final_name = [item + filetype for item in name]
        return final_name
    def location(final_name):
        imgNum = len(final_name)
        assert (imgNum > 0)
        locate = []
        for i in range(imgNum):
            t = final_name[i]
            l1 = t.rindex('_')
            l2 = t[1:l1 - 1].rindex('_')
            h = int(t[l2 + 2:l1])
            w = int(t[l1 + 1:-4])
            locate.append([h, w])
        return locate


    def load_Img(path, final_name):
         imgNum = len(final_name)
         assert(imgNum > 0)
         img = cv2.imread(path+final_name[0])
         [width, height] = img.size
         data = np.empty((height, width, imgNum))
         for i in range(imgNum):
             img = cv2.imread(path+final_name[i])
             data[:, :, i] = img
         return data

    def sp(path, final_name, loc):
        imgNum = len(final_name)
        assert (imgNum > 0 and len(loc) == imgNum)
        print(path)
        print(path + final_name[0])
        img = Image.open(path + final_name[0])
        [width, height] = img.size
        c = 16
        N = 8
        t = np.array(loc)
        Label = (np.zeros([max(t[:, 0]) * (height - c) + c, max(t[:, 1]) * (width - c) + c]))#.astype('uint8')
        print(Label.shape)
        Label[:, :] = np.NaN
        for i in range(imgNum):
            img = imread(path + final_name[i])
            # img = np.around(img/N)
            corner = []
            rectangle = []
            corner.append(img[0:c, 0:c])
            corner.append(img[-c:, 0:c])
            corner.append(img[0:c, -c:])
            corner.append(img[-c:, -c:])
            rectangle.append(img[c:-c, 0:c])
            rectangle.append(img[c:-c, -c:])
            rectangle.append(img[0:c, c:-c])
            rectangle.append(img[-c:, c:-c])
            square = img[c:-c, c:-c]  # center
            t_x = []
            t_x.append((loc[i][0] - 1) * (height - c))
            t_x.append((loc[i][0] - 1) * (height - c) + c)
            t_x.append(loc[i][0] * (height - c))
            t_x.append(loc[i][0] * (height - c) + c)
            t_y = []
            t_y.append((loc[i][1] - 1) * (width - c))
            t_y.append((loc[i][1] - 1) * (width - c) + c)
            t_y.append(loc[i][1] * (width - c))
            t_y.append(loc[i][1] * (width - c) + c)

            Label[t_x[1]:t_x[2], t_y[1]:t_y[2]] = square
            if np.isnan(Label[t_x[0]:t_x[1], t_y[0]:t_y[1]]).sum():
                Label[t_x[0]:t_x[1], t_y[0]:t_y[1]] = corner[0]
            if np.isnan(Label[t_x[0]:t_x[1], t_y[2]:t_y[3]]).sum():
                Label[t_x[0]:t_x[1], t_y[2]:t_y[3]] = corner[2]
            if np.isnan(Label[t_x[2]:t_x[3], t_y[0]:t_y[1]]).sum():
                Label[t_x[2]:t_x[3], t_y[0]:t_y[1]]=corner[1]
            if np.isnan(Label[t_x[2]:t_x[3], t_y[2]:t_y[3]]).sum():
                Label[t_x[2]:t_x[3], t_y[2]:t_y[3]] = corner[3]


            if np.isnan(Label[t_x[0]:t_x[1], t_y[1]:t_y[2]]).sum():
                Label[t_x[0]:t_x[1], t_y[1]:t_y[2]] = rectangle[2]
            if np.isnan(Label[t_x[2]:t_x[3], t_y[1]:t_y[2]]).sum():
                Label[t_x[2]:t_x[3], t_y[1]:t_y[2]] = rectangle[3]
            if np.isnan(Label[t_x[1]:t_x[2], t_y[0]:t_y[1]]).sum():
                Label[t_x[1]:t_x[2], t_y[0]:t_y[1]] = rectangle[0]
            if np.isnan(Label[t_x[1]:t_x[2], t_y[2]:t_y[3]]).sum():
                Label[t_x[1]:t_x[2], t_y[2]:t_y[3]] = rectangle[1]

        return Label

    filetype = '.png'
    final_name = get_filename(path, filetype)
    loc = location(final_name)
    # data=load_Img(path,final_name)
    data = sp(path, final_name, loc)
    data = data.astype('int8')
    def writeTiff(im_data, path):
        datatype = gdal.GDT_Byte
        im_bands, (im_height, im_width) = 1, im_data.shape
            # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
        dataset.GetRasterBand(1).WriteArray(im_data)
        del dataset
    name = os.path.join(save_path,'label.tif')
    # cv2.imwrite(save_path + 'fusion.png',data*32)
    writeTiff(data, name)
    print('Splice Done!')
    return name

def main():
    # path = "G:/DL/data/"
    # path = tkfd.askdirectory() + '/'
    path = '/media/zhou/System/projects/Detectron-master/Detectron-master/lib/datasets/data/OUR/label/'
    save_path = '/home/zhou/'
    print(path)
    splice(path, save_path)
    # filetype = '.png'
    # final_name = get_filename(path, filetype)
    # loc = location(final_name)
    # # data=load_Img(path,final_name)
    # data = sp(path, final_name, loc)
    # data = data.astype('uint8')
    # io.imsave(save_path + 'fusion.png', data)
    # new_im = Image.fromarray(data)
    #new_im.show()


if __name__ == '__main__':
    main()

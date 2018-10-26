# coding: utf-8

import gdal
import numpy as np
import os

def partition(image_path, image_name, save_path, szm, szn, cd, height0, width0):
    def readTif(fileName):
        dataset = gdal.Open(fileName)
        if dataset == None:
            print(fileName + "文件无法打开")
            return
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        return dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据

    def writeTiff(im_data, im_width, im_height, im_bands, path):
        if ('uint8' or 'int8') in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif ('int16' or 'uint16') in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        elif len(im_data.shape) == 2:
            im_data = np.array([im_data])
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape
            # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
        for i in range(im_bands):
            try:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
            except:
                print(1)
        del dataset


    # save_path = save_path + image_name[:-4] + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = readTif(image_path + image_name)
    #print(image_path + image_name)
    # img = img[:, :-1, :]
    [c, height, width] = np.shape(img)
    if [height, width] != [height0, width0]:
        if height0 > height:
            img = np.hstack((img, img[:, height - height0:, :]))
        if width0 > width:
            img = np.dstack((img, img[:, :, width - width0:]))
    height, width = [height0, width0]
    lm = [k for k in range(0, height, szm-cd)]
    ln = [k for k in range(0, width, szn-cd)]
    lm2 = [x + szm for x in lm]
    ln2 = [x + szn for x in ln]
    if lm2[-1] != height:
        img = np.hstack((img, np.zeros([img.shape[0], lm2[-1] - height, img.shape[2]]).astype('uint16')))
    if ln2[-1] != width:
        img = np.dstack((img, np.zeros([img.shape[0], img.shape[1], ln2[-1] - width]).astype('uint16')))

    for index_i, i in enumerate(lm):
        for index_j, j in enumerate(ln):
            save_name = "image" + '_' + str(index_i + 1) + '_' + str(index_j + 1) + ".tif"
            writeTiff(img[:, i:i+szm, j:j+szn], szn, szm, c, save_path + save_name)
    return True
def main():
    image_path = '/media/zhou/系统/二期图像2/天津航天城/GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826/'
    image_name = 'GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826-MSS2_fusion.tif'
    save_path = '/home/zhou/projects/Detectron-master/Detectron-master/lib/datasets/data/OUR/'
    szm, szn, cd = [428, 472, 16]
    szm, szn, cd = [256, 256, 16]
    height0, width0 = [27620, 29200]  # 预设标准尺寸：[4, 27620, 29200] 有可能比这个小1

    partition(image_path, image_name, save_path, szm, szn, cd, height0, width0)

if __name__ == '__main__':
    main()
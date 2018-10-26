# coding: utf-8
import sys
import os
from myweb.Detector.partition import partition
from myweb.Detector.splice import splice
# sys.path.append('../Back/')

import shutil
import warnings
warnings.filterwarnings('ignore')
import time

def detection(image_path, image_name,label_path):
    # 临时路径
    # part_img_path = './maskrcnn/lib/datasets/data/test/'  # 切割图像路径文件夹
    try:
        part_img_path = './myweb/Detector/temp/images/'
        part_label_path = './myweb/Detector/temp/labels/'  # label生成路径文件夹
        # label_path = './myweb/Detector/temp/label/'  # 拼和后的label
        temp_path = [part_img_path, part_label_path, label_path]
        for path in temp_path:
            if not os.path.exists(path):
                os.makedirs(path)

        height0, width0 = [27620, 29200]  # 预设标准尺寸：[4, 27620, 29200] 有可能比这个小1 如果小于这个尺寸进行镜像填充
        model = 'mask r-cnn'
        model = 'segnet'

        #  这里进行预测
        if model == 'mask r-cnn':
            szm, szn, cd = [428, 472, 16]  # 切割后图像高度、宽度、覆盖像素个数
        else:
            szm, szn, cd = [256, 256, 16]
        partition(image_path, image_name, part_img_path, szm, szn, cd, height0, width0)
        if model == 'mask r-cnn':
            from myweb.Detector.maskrcnn.mask_rcnn import mask_rcnn_predict
            mask_rcnn_predict(part_img_path, part_label_path)
        elif model == 'segnet':
            from myweb.Detector.segnet.segnet import segnet_predict
            segnet_predict(part_img_path, part_label_path)
        else:
            raise Exception("Invalid predict mothod!")  # 强制触发异常
        return splice(part_label_path, label_path)
    except Exception as e:
        return e

    # geojson_make(image_path + image_name, label_path + 'fusion.png', json_path)
def main():
    start_time = time.time()
    image_path = '/media/zhou/系统/二期图像2/天津航天城/GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826/'  # 原图像保存文件夹
    image_name = 'GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826-MSS2_fusion.tif'  # 原图文件名
    transformed_image_path = '../ImagryProcess/GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826-MSS2_fusionR.tif'
    json_path = './temp/'
    detection(image_path, image_name)
    print(time.time() - start_time)

if __name__ == '__main__':
    main()
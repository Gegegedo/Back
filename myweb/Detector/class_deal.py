# coding:utf-8
import json
import sys
import os
import time
sys.path.append('./cocoapi-master/PythonAPI/')
import pycocotools.mask as mask_util
import numpy as np
import cv2
import copy


def json2label(json_path, save_dir, score_threshold):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(json_path, "r") as file:
        list = json.load(file)
    images_set = []
    [images_set.append(segm['image_id']) for segm in list if segm['image_id'] not in images_set]
    list_class = [[] for n in range(8)]
    list_image = [copy.deepcopy(list_class) for n in range(len(images_set))]
    del list_class
    # [list_class[segm['category_id']].append(segm) for segm in list]
    list_image[0][1] = ['test']
    [list_image[images_set.index(segm['image_id'])][segm['category_id']].append(segm) for segm in list]
    del list
    list_image[0][1].remove('test')
    t_deal = np.zeros(max(images_set)).astype('uint16')
    for k, segms in enumerate(list_image):
        def single_label(segms):
            for cls in range(len(segms)):
                if len(segms[cls]) > 0:
                    label = np.zeros(segms[cls][0]['segmentation']['size']).astype('uint8')
                    flag = True
                    image_id = segms[cls][0]['image_id']
                    break
                else:
                    flag = False
                    image_id = []
            if flag:
                prob = []
                for segm in segms:  # segm 同一张图同一类别不同物体
                    if len(segm) == 0: continue
                    # bilabel = np.zeros(segm[0]['segmentation']['size']).astype('uint8')
                    p = np.zeros(segm[0]['segmentation']['size'])
                    if len(prob) == 0:
                        prob = np.copy(p)
                    for seg in segm:  # seg 每个实例
                        if seg['score'] > score_threshold:
                            mask = np.array(mask_util.decode(seg['segmentation']), dtype=np.uint8)  # 这里需要改为概率最大的类别
                            p = np.array(max(p.tolist(), (seg['score'] * mask).tolist()), dtype=np.float)
                        else: continue
                    #     bilabel += mask
                    # bilabel[bilabel > 0] = 1
                    # location = prob < p  # 需要更改类别的区域
                    label[prob < p] = seg['category_id']
                    # label = np.array(max(label.tolist(), (seg['category_id'] * bilabel).tolist()), dtype=np.uint8)
                cv2.imwrite(save_dir + str(image_id) + '.png', label)
            return image_id
        start_time = time.time()
        t_deal[single_label(segms)-1] = 1
        # print('%d / %d : %f s' % (k, len(list_image), time.time()-start_time))


def main():
    score_threshold = 0.5
    json_path = '/home/zhou/projects/Detectron-master/Detectron-master/lib/datasets/data/OUR/train_out_0708/test/OUR/generalized_rcnn/segmentations_OUR_results.json'
    save_dir = '/home/zhou/projects/Detectron-master/Detectron-master/lib/datasets/data/OUR/train_out_0708/test/OUR/generalized_rcnn/labels/'
    json2label(json_path, save_dir, score_threshold)


if __name__ == '__main__':
    main()


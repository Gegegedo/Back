# utf-8
import numpy as np
import gdal
import cv2
import matplotlib.pyplot as plt

path = '/home/zhou/projects/Detectron-master/Detectron-master/lib/datasets/data/OUR/label/'
label_path = 'GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826-MSS2_fusion_5_22.png'

label = cv2.imread(path + label_path)


def open(img):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

def close(img):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))


t = np.unique(label).tolist()
t.remove(0)
cz = {0: open, 1: close}
for cls in t:
    done = close(close(close(open(open(label)))))

cv2.imshow('1', done * 255 / 8)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.subplot(1,2,1)
plt.imshow(label * 255 / 8)
plt.subplot(1,2,2)
plt.imshow(done * 255 / 8)
plt.show()
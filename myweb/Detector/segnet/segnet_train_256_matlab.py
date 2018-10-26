import matplotlib
matplotlib.use("Agg")
from osgeo import gdal
import argparse
import numpy as np  
from keras.models import Sequential  
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt  
import cv2
import random
import os

from keras import backend
backend.set_image_dim_ordering('th')

seed = 7  
np.random.seed(seed)  

#读入图片的width和height
img_size = 256

#label的总类数
classes = [0., 1., 2., 3., 4., 5., 6., 7.]
n_label = len(classes)

labelencoder = LabelEncoder()  
labelencoder.fit(classes)

path = 'D:/Workspace/Matlab/DataAugment/AugmentDataset/'
image_path = path + 'image/'
label_path = path + 'label/'


#读取label
def load_label(filepath):
    label = cv2.imread(filepath, 0)
    label = np.array(label, dtype="float")
    label[label == 8] = 0
    return label


#读取tif图片
def load_tifimg(filepath):
    img = gdal.Open(filepath)
    im_width = img.RasterXSize
    im_height = img.RasterYSize
    return img.ReadAsArray(0, 0, im_width, im_height)


def get_train_val(validate_rate=0.25):
    filename_list = [filename for filename in os.listdir(image_path) if filename.endswith('.tif')]
    random.shuffle(filename_list)
    validate_num = int(validate_rate * len(filename_list))
    return filename_list[validate_num:], filename_list[:validate_num]


# data for training  
def generateTrainData(batch_size, dataset=[]):
    while True:  
        train_image = []
        train_label = []  
        batch = 0
        for filename in dataset:
            batch += 1 
            img = load_tifimg(image_path + filename)
            train_image.append(img)
            labelname = label_path + filename[:-4] + '.png'
            label = load_label(labelname)
            train_label.append(label)  
            if batch % batch_size == 0:
                train_image = np.array(train_image)
                train_label = np.array(train_label).flatten()
                train_label = labelencoder.transform(train_label)  
                train_label = to_categorical(train_label, num_classes=n_label)  
                train_label = train_label.reshape((batch_size, img_size * img_size, n_label))
                yield (train_image, train_label)
                train_image = []
                train_label = []  
                batch = 0  


# data for validation 
def generateValidateData(batch_size, data=[]):
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_tifimg(image_path + url)
            img = np.array(img)
            valid_data.append(img)
            url_label = os.path.splitext(url)[0]
            label = load_label(label_path + url_label + '.png')
            label = img_to_array(label).reshape((img_size * img_size,))
            valid_label.append(label)  
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label).flatten()  
                valid_label = labelencoder.transform(valid_label)  
                valid_label = to_categorical(valid_label, num_classes=n_label)  
                valid_label = valid_label.reshape((batch_size, img_size * img_size, n_label))
                yield (valid_data, valid_label)
                valid_data = []  
                valid_label = []  
                batch = 0  
  
def SegNet():  
    model = Sequential()  
    #encoder  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(4, img_size, img_size),padding='same', activation='relu'))
    model.add(BatchNormalization())  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #(128,128)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(64,64)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(32,32)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(16,16)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(8,8)  
    #decoder  
    model.add(UpSampling2D(size=(2, 2)))
    #(16,16)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(32,32)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(64,64)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(128,128)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(256,256)  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(4, img_size, img_size), padding='same', activation='relu'))
    model.add(BatchNormalization())  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))  
    model.add(Reshape((n_label, img_size * img_size)))
    #axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2)  
    model.add(Permute((2,1)))  
    model.add(Activation('softmax'))  
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
    model.summary()  
    return model  
  
  
def train(args): 
    EPOCHS = 100
    BS = 2
    model = SegNet()  
    modelcheck = ModelCheckpoint(args['model'], monitor='val_acc', save_best_only=True, mode='max')
    callable = [modelcheck]  
    train_set, val_set = get_train_val()
    train_len = len(train_set)
    valid_len = len(val_set)
    print ("the number of train data is ", train_len)
    print ("the number of val data is ", valid_len)
    H = model.fit_generator(generator=generateTrainData(BS, train_set), steps_per_epoch=train_len//BS, epochs=EPOCHS, verbose=1,
                    validation_data=generateValidateData(BS, val_set), validation_steps=valid_len//BS, callbacks=callable, max_q_size=1)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False,
                    help="path to output model", default="model_256_matlab.H5")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args


if __name__=='__main__':  
    args = args_parse()
    # if args['augment'] == True:
    #     filepath ='./aug/train/'

    train(args)
    #predict()  

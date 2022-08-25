import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.constraints import maxnorm
from keras.utils import np_utils
# from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import glob
from keras.models import load_model
import math
import os
import shutil
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing import image
import warnings
warnings.filterwarnings('ignore')


seed = 35
epochs = 40
optimizer = 'adam'


class CIFAR10:
    def __init__(self):
        self.class_dict = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }

        self.model_file_name = 'cifar10_model.h5'
        self.path_to_test_pic = 'cifar10\\'
        self.name = 'cifar10'


class Rice:
    def __init__(self):
        self.class_dict = {
            0: 'Arborio',
            1: 'Basmati',
            2: 'Ipsala',
            3: 'Jasmine',
            4: 'Karacadag'
        }

        self.class_dict_name_ind = {
            'Arborio': 0,
            'Basmati': 1,
            'Ipsala': 2,
            'Jasmine': 3,
            'Karacadag': 4
        }

        self.model_file_name = 'rice_model.h5'
        self.path_to_the_dataset = '..\\Rice_Image_Dataset\\'
        self.path_to_test_pic = 'rice\\'
        self.general_img_count = 15000
        self.target_size = (64, 64)
        self.name = 'rice'

class Flowers:
    def __init__(self):
        self.class_dict = {
            0: 'daisy',
            1: 'dandelion',
            2: 'rose',
            3: 'sunflower',
            4: 'tulip'
        }

        self.class_dict_name_ind = {
            'daisy': 0,
            'dandelion': 1,
            'rose': 2,
            'sunflower': 3,
            'tulip': 4
        }

        self.model_file_name = 'flowers_model.h5'
        self.path_to_the_dataset = '..\\flowers\\'  # 'F:\\Adversarial Attacks\\flowers\\'
        self.path_to_test_pic = 'flowers\\'
        self.name = 'flowers'
        self.target_size = (150, 150)


def create_CNN_module(image_shape, class_num):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3)),
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model, None


def create_CNN_module_flowers(image_shape, class_num, x_train):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=image_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(class_num, activation="softmax"))

    red_lr = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.1)
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)
    model.compile(optimizer=adam_v2.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model, datagen

# def create_CNN_module_flowers(image_shape, class_num, x_train):
#     model = Sequential()
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=image_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(rate=0.5))
#     model.add(Dense(class_num, activation="softmax"))
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         zoom_range=0.20,
#         width_shift_range=0.3,
#         height_shift_range=0.3,
#         horizontal_flip=True,
#         vertical_flip=True)
#
#     datagen.fit(x_train)
#     model.compile(optimizer=adam_v2.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#     print(model.summary())
#     return model, datagen


def train(model, x_train, y_train, x_test, y_test, info, datagen=None):
    # np.random.seed(seed)
    checkpoint_callback = ModelCheckpoint(info.model_file_name, monitor='val_accuracy', save_best_only=True, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64,
                        callbacks=[reduce_lr, checkpoint_callback])
    if datagen is not None:
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
                                      epochs=epochs, validation_data=(x_test, y_test),
                                      verbose=1, steps_per_epoch=x_train.shape[0] // 64,
                                      callbacks=[reduce_lr, checkpoint_callback])
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('train_accuracy.png')

    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('train_loss.png')


# def maincifar10():
#     # loading in the data
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#     # normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train = x_train / 255.0
#     x_test = x_test / 255.0
#     # one hot encode outputs
#     y_train = np_utils.to_categorical(y_train)
#     y_test = np_utils.to_categorical(y_test)
#
#     class_num = y_test.shape[1]
#     image_shape = x_train.shape[1:]
#
#     model = create_CNN_module(image_shape, class_num)
#     train(model, x_train, y_train, x_test, y_test)
#     info = CIFAR10()
#     model.save(info.model_file_name)
#
#     scores = model.evaluate(x_test, y_test, verbose=0)
#     print("Accuracy: ", scores[1] * 100)


def split_dataset(dataset_class):
    # train_count = math.floor(dataset_class.general_img_count*0.7)
    # validation_and_test_count = math.floor(dataset_class.general_img_count*0.15)
    ind = 0.0
    for class_folder in glob.glob(dataset_class.path_to_the_dataset + '*'):
        general_img_count = len(os.listdir(class_folder))
        train_count = math.floor(general_img_count * 0.7)
        validation_and_test_count = math.floor(general_img_count * 0.15)
        for img in glob.glob(class_folder + '\\*'):
            if ind < train_count:
                folder = 'train'
            elif train_count <= ind < train_count+validation_and_test_count:
                folder = 'test'
            else:
                folder = 'validation'
            ind += 1.0

            destination_path = dataset_class.path_to_the_dataset + Path(class_folder).name + '\\' + folder + '\\'
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.move(img, destination_path)
        ind = 0.0


def main_rice():
    # split_dataset(Rice())
    info = Rice()
    x_test = []
    y_test = []
    x_train = []
    y_train = []
    for class_folder in glob.glob(info.path_to_the_dataset + '*'):
        for folder in ['train', 'test']:
            for img_file in glob.glob(info.path_to_the_dataset + Path(class_folder).name + '\\' + folder + '\\*'):
                img = image.load_img(img_file, target_size=(50, 50))
                img = image.img_to_array(img)
                if folder == 'test':
                    x_test.append(img)
                    y_test.append(info.class_dict_name_ind[Path(class_folder).name])
                else:
                    x_train.append(img)
                    y_train.append(info.class_dict_name_ind[Path(class_folder).name])

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    class_num = y_test.shape[1]
    image_shape = x_train.shape[1:]

    model = create_CNN_module(image_shape, class_num)
    train(model, x_train, y_train, x_test, y_test, info)
    model.save(info.model_file_name)

    model = load_model(info.model_file_name)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: ", scores[1] * 100)


def main_flowers():
    info = Flowers()
    # split_dataset(info)
    x_test = []
    y_test = []
    x_train = []
    y_train = []
    for class_folder in glob.glob(info.path_to_the_dataset + '*'):
        for folder in ['train', 'test']:
            for img_file in glob.glob(info.path_to_the_dataset + Path(class_folder).name + '\\' + folder + '\\*'):
                img = image.load_img(img_file, target_size=(150, 150))
                img = image.img_to_array(img)
                if folder == 'test':
                    x_test.append(img)
                    y_test.append(info.class_dict_name_ind[Path(class_folder).name])
                else:
                    x_train.append(img)
                    y_train.append(info.class_dict_name_ind[Path(class_folder).name])

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    class_num = y_test.shape[1]
    image_shape = x_train.shape[1:]

    model, datagen = create_CNN_module_flowers(image_shape, class_num, x_train)
    train(model, x_train, y_train, x_test, y_test, info, datagen)

    model = load_model(info.model_file_name)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: ", scores[1] * 100)


if __name__ == '__main__':
    main_flowers()
    # main_rice()
    # maincifar10()
    # info = Flowers()
    # x_test = []
    # y_test = []
    #
    # for class_folder in glob.glob(info.path_to_the_dataset + '*'):
    #     for img_file in glob.glob(info.path_to_the_dataset + Path(class_folder).name + '\\' + 'test' + '\\*'):
    #         img = image.load_img(img_file, target_size=(150, 150))
    #         img = image.img_to_array(img)
    #         x_test.append(img)
    #         y_test.append(info.class_dict_name_ind[Path(class_folder).name])
    #
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)
    #
    # x_test = x_test.astype('float32')
    # x_test = x_test / 255.0
    #
    # y_test = np_utils.to_categorical(y_test)
    #
    # model = load_model(info.model_file_name)
    # scores = model.evaluate(x_test, y_test, verbose=0)
    # print("Accuracy: ", scores[1] * 100)




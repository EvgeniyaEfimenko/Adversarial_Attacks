from foolbox.attacks import FGSM, L2BasicIterativeAttack, PGD, \
    L2CarliniWagnerAttack, L2DeepFoolAttack, InversionAttack, NewtonFoolAttack, LinfDeepFoolAttack,\
    LinfBasicIterativeAttack, LinfinityBrendelBethgeAttack, DDNAttack
# from foolbox.attacks import FGSM, BIM, PGD, MomentumIterativeAttack, DeepFoolAttack, CarliniWagnerL2Attack
from matplotlib import rcParams

from train import *
import eagerpy as ep
from foolbox import accuracy, samples
import tensorflow as tf
from foolbox.models import TensorFlowModel
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import foolbox
import keras
import logging
import os
import sys
from skimage.io import imread, imshow
from skimage.exposure import histogram

# from cleverhans.attacks.Attack import LBFGS

def main(my_trained):

    # attacks = [FGSM(), L2BasicIterativeAttack(), PGD(), L2CarliniWagnerAttack(steps=1000), L2DeepFoolAttack()]
    # attacks_names = ['FGSM', 'L2BasicIterativeAttack', 'PGD', 'L2CarliniWagnerAttack', 'L2DeepFoolAttack']
    info_class = Flowers()
    # info_class = Rice()
    # model = load_model('модели\\rice Accuracy  99.29777979850769\\' + info_class.model_file_name)
    model = load_model('модели\\flowers a = 75\\' + info_class.model_file_name)
    # model = tf.keras.applications.MobileNetV2(weights="imagenet")
    # keras.backend.set_learning_phase(0)
    fmodel = TensorFlowModel(model, bounds=(0, 1))
    attack = LinfinityBrendelBethgeAttack()
    attack_name = 'LinfinityBrendelBethgeAttack'
    epsilons = [0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.01, 0.1, 0.3, 0.5, 1.0]

    x_test = []
    y_test = []
    x_test_adv = []

    if not my_trained:
        name = model.name.split('_')[0]
        folder = f'{name}_pic'
        if not os.path.exists(folder):
            os.mkdir(folder)
        attack_folder = f'{folder}/{attack_name}'
        if not os.path.exists(attack_folder):
            os.mkdir(attack_folder)
        images, labels = foolbox.utils.samples(fmodel, dataset='imagenet', batchsize=16)
        print('acc before attack', foolbox.utils.accuracy(fmodel, images, labels) * 100)
        images = ep.astensor(images)
        labels = ep.astensor(labels)
        raw, advs, is_adv = attack(fmodel, images, labels, epsilons=0.03)
        print('acc after attack', foolbox.utils.accuracy(fmodel, advs, labels) * 100)
        foolbox.plot.images(images)
        plt.tight_layout()
        plt.savefig(f'{attack_folder}/img_{name}.jpg')
        plt.clf()
        foolbox.plot.images(advs)
        plt.tight_layout()
        plt.savefig(f'{attack_folder}/adv_{name}.jpg')
        plt.close()
        i = 1
        for img, adv in zip(images, advs):
            print(i)
            sh = list(img.shape)
            sh.insert(0, 1)
            img = img.reshape(sh)
            foolbox.plot.images(img)
            plt.tight_layout()
            plt.savefig(f'{attack_folder}/img_{name}_{i}.jpg')
            plt.close()
            hist, bins = histogram(img.raw[0].numpy())
            plt.plot(bins, hist)
            plt.tight_layout()
            plt.savefig(f'{attack_folder}/img_{name}_hist_{i}.jpg')
            plt.close()
            adv = adv.reshape(sh)
            foolbox.plot.images(adv)
            plt.tight_layout()
            plt.savefig(f'{attack_folder}/adv_{name}_{i}.jpg')
            plt.close()
            hist, bins = histogram(adv.raw[0].numpy())
            plt.plot(bins, hist)
            plt.tight_layout()
            plt.savefig(f'{attack_folder}/adv_{name}_hist_{i}.jpg')
            plt.close()
            i += 1
    else:

        folder = f'{info_class.name}_results'
        if not os.path.exists(folder):
            os.mkdir(folder)
        main_folder = f'{folder}/{attack_name}'
        if not os.path.exists(main_folder):
            os.mkdir(main_folder)
        logging.basicConfig(filename=f'{main_folder}/{info_class.name}_{attack_name}.log', filemode='w',
                            format='%(message)s',
                            level=logging.INFO)

        adv_pic_folder = f'{main_folder}/adversarial_pic'
        if not os.path.exists(adv_pic_folder):
            os.mkdir(adv_pic_folder)

        histogram_folder = f'{main_folder}/histogram_pic'
        if not os.path.exists(histogram_folder):
            os.mkdir(histogram_folder)

        best_adv_pic_folder = f'{main_folder}/best_adversarial_pic'
        if not os.path.exists(best_adv_pic_folder):
            os.mkdir(best_adv_pic_folder)

        for img_file in glob.glob(info_class.path_to_test_pic + '*'):
            # Подготовка тестового изображения
            size = imread(img_file).shape
            m, n = size[0], size[1]
            img = image.load_img(img_file, target_size=info_class.target_size)
            test_image = image.img_to_array(img)
            x_test.append(test_image)
            plt.clf()
            hist, bins = histogram(test_image)
            plt.plot(bins, hist)
            plt.savefig(f'{histogram_folder}/{Path(img_file).stem}_hist.jpg')
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0
            result = model.predict(test_image)
            predicted_class = result.argmax()
            probability = result[0][predicted_class] * 100
            logging.info(f'predicted before attack: {info_class.class_dict[predicted_class]}  with probability = '
                         f'{probability} % / real: {Path(img_file).stem.split()[0]}')

            label = np.array([info_class.class_dict_name_ind[Path(img_file).stem.split()[0]]])
            y_test.append(label[0])
            tensor_img = tf.convert_to_tensor(test_image)
            tensor_label = tf.convert_to_tensor(label)
            tensor_img, tensor_label = ep.astensors(tensor_img, tensor_label)
            raw_advs, clipped_advs, success = attack(fmodel, tensor_img, tensor_label, epsilons=epsilons)

            first_adv_ind = np.where(success.raw.numpy() == True)
            if first_adv_ind[0] != []:
                first_adv_ind = first_adv_ind[0][0]
            else:
                # x_test.pop(-1)
                # y_test.pop(-1)
                x_test_adv.append(x_test[-1])
                continue
            ind = 0

            for eps, raw_adversarial_image, clipped_adversarial_image in zip(epsilons, raw_advs, clipped_advs):
                result = model.predict(clipped_adversarial_image.raw)
                predicted_class = result.argmax()
                probability = result[0][predicted_class] * 100
                logging.info(
                    f'{eps} / predicted after attack: {info_class.class_dict[predicted_class]} with probability = '
                    f'{probability} % / real: {Path(img_file).stem.split()[0]}')

                if first_adv_ind == ind:
                    x_test_adv.append(clipped_adversarial_image.raw[0].numpy()*255)
                    plt.clf()
                    hist, bins = histogram(clipped_adversarial_image.raw[0].numpy()*255)
                    plt.plot(bins, hist)
                    plt.savefig(f'{histogram_folder}/{Path(img_file).stem}_hist_{eps}.jpg')

                adversarial_image = image.array_to_img(clipped_adversarial_image.raw[0].numpy())
                adversarial_image = adversarial_image.resize((n, m))
                adversarial_image.save(f'{adv_pic_folder}/{Path(img_file).stem}_{eps}.jpg')
                if first_adv_ind == ind:
                    adversarial_image.save(f'{best_adv_pic_folder}/{Path(img_file).stem}_{eps}.jpg')
                ind += 1

                # fig, ax = plt.subplots()
                # ax.imshow(adversarial_image)
                # plt.show()
                # ax.imshow(adversarial_image, cmap='gray')
                # plt.show()

                # foolbox.plot.images(clipped_adversarial_image)
                # plt.savefig(f'{adv_pic_folder}/{Path(img_file).stem}_{eps}.jpg')
                # img = image.load_img(f'{adv_pic_folder}/{Path(img_file).stem}_{eps}.jpg', target_size=(m, n))
                # image.save_img(f'{adv_pic_folder}/{Path(img_file).stem}_{eps}_img.jpg', img)
            logging.info('')
            logging.info(
                '________________________________________________________________________________________________')
            logging.info('')
            # print(img_file)

        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_test_adv = np.array(x_test_adv)
        x_test_adv = x_test_adv.astype('float32')
        x_test = x_test.astype('float32')
        x_test_adv = x_test_adv / 255.0
        x_test = x_test / 255.0
        y_test = np_utils.to_categorical(y_test)

        scores = model.evaluate(x_test, y_test, verbose=0)
        logging.info(f"Accuracy of the real images: {scores[1] * 100}")

        scores = model.evaluate(x_test_adv, y_test, verbose=0)
        logging.info(f"Accuracy of the adversarial images: {scores[1] * 100}")



if __name__ == '__main__':
    main(True)

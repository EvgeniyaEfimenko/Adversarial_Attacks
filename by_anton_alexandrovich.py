import glob
import os.path
import shutil
from pathlib import Path
from numpy import uint8
from PIL import Image

import eagerpy as ep
import numpy as np
import tensorflow as tf

from foolbox.attacks import PGD, FGSM, LinfBasicIterativeAttack
from foolbox.models import TensorFlowModel
from keras.preprocessing import image

import matplotlib.pyplot as plt
from skimage.exposure import histogram
from skimage.io import imread
from keras.utils.vis_utils import plot_model
import errno



def create_resnet_model():
    from keras.applications.resnet_v2 import ResNet152V2, preprocess_input, decode_predictions
    model = ResNet152V2(weights="imagenet")  # imagenet-mini: 0.687993882232985, imagenet: 0.6907
    target_size = (224, 224)
    return model, target_size, preprocess_input, decode_predictions, postprocess_tf


def create_mobilenet_model():
    from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    model = MobileNetV2(weights="imagenet")  # imagenet-mini: 0.6839, imagenet: 0.68048
    target_size = (224, 224)
    return model, target_size, preprocess_input, decode_predictions, postprocess_tf


def create_efficientnet_model():
    from keras.applications.efficientnet_v2 import EfficientNetV2L, preprocess_input, decode_predictions
    model = EfficientNetV2L(weights="imagenet")  # imagenet-mini: 0.84731, imagenet: 0.85256
    target_size = (480, 480)
    return model, target_size, preprocess_input, decode_predictions, postprocess_efficientnet


def create_davit_model():
    from keras_cv_attention_models import davit
    model = davit.DaViT_B(pretrained='models/davit_b_imagenet.h5')  # imagenet-mini: 0.817 / 0.8106, imagenet: 0.8233
    target_size = (224, 224)
    return model, target_size, preprocess_tf, model.decode_predictions, postprocess_tf


def create_model(model_name):
    """
    The following models are supported:
    <ul>
        <li>ResNet152V2</li>
        <li>MobileNetV2</li>
        <li>EfficientNetV2L</li>
        <li>DaViT_B</li>
    </ul>
    :param model_name:
    :return:
    """
    if model_name == 'ResNet152V2':
        return create_resnet_model()
    if model_name == 'MobileNetV2':
        return create_mobilenet_model()
    if model_name == 'EfficientNetV2L':
        return create_efficientnet_model()
    if model_name == 'DaViT_B':
        return create_davit_model()
    raise ValueError('Unexpected `model_name` value')


def imagenet_mini_reader():
    # path_to_the_dataset = 'datasets/imagenet-mini/val/'
    # path_to_the_dataset = '../imagenet-mini/val/'
    # path_to_the_dataset = 'smoothing/imagenet-mini/linear/'
    # path_to_the_dataset = 'smoothing/imagenet-mini/median/'
    path_to_the_dataset = 'smoothing/imagenet-mini/averaging/'
    # path_to_the_dataset = 'smoothing/imagenet-mini/gaussian/'
    for class_folder in glob.glob(path_to_the_dataset + '*'):
        class_name = Path(class_folder).name
        for image_file_path in glob.glob(os.path.join(class_folder, '*')):
            yield image_file_path, class_name


def imagenet_reader():
    # path_to_the_dataset = 'datasets/imagenet/data/val/'
    path_to_the_dataset = '../imagenet/val/'
    for image_file_path in glob.glob(path_to_the_dataset + '*'):
        image_file_name = Path(image_file_path).stem
        class_name = image_file_name.split('_')[-1]
        yield image_file_path, class_name


def create_imagenet_reader(dataset='imagenet_mini'):
    if dataset == 'imagenet_mini':
        return imagenet_mini_reader()
    if dataset == 'imagenet':
        return imagenet_reader()
    raise ValueError('Unexpected `dataset` value')


def create_attack_dataset_reader(model_name, attack_name):
    # path_to_the_dataset = f'attack_results/{model_name}/{attack_name}/'
    # path_to_the_dataset = f'best_attack_results/{model_name}/{attack_name}/'
    path_to_the_dataset = f'best_attack_results_max/{model_name}/{attack_name}/'
    # path_to_the_dataset = f'best_smoothing_attack_results_max/{model_name}/{attack_name}/linear/'
    # path_to_the_dataset = f'best_smoothing_attack_results_max/{model_name}/{attack_name}/median/'
    # path_to_the_dataset = f'best_smoothing_attack_results/{model_name}/{attack_name}/averaging/'
    # path_to_the_dataset = f'best_smoothing_attack_results/{model_name}/{attack_name}/gaussian/'
    for class_folder in glob.glob(path_to_the_dataset + '*'):
        class_name = Path(class_folder).name
        for image_file_path in glob.glob(os.path.join(class_folder, '*')):
            yield image_file_path, class_name


def evaluate_model_imagenet_dataset(model_name, dataset):
    images_reader = create_imagenet_reader(dataset)
    evaluate_model(model_name, images_reader)


# ResNet152V2 on MobileNetV2 0.6673
# MobileNetV2 on MobileNetV2 0.32959
# EfficientNetV2L on MobileNetV2 0.7932
# DaViT_B on MobileNetV2 0.79811
def evaluate_model_attacked_dataset(model_name, attacked_model_name, attack_name):
    images_reader = create_attack_dataset_reader(attacked_model_name, attack_name)
    return evaluate_model(model_name, images_reader)


def evaluate_model(model_name, images_reader):
    model, target_size, preprocess_input, decode_predictions, _ = create_model(model_name)

    count = 0
    true_count = 0
    for image_file_path, class_name in images_reader:
        test_image = load_preprocess_image(image_file_path, target_size, preprocess_input)
        result = model.predict(test_image)
        predicted_class_name = decode_predictions(result, top=1)[0][0][0]
        true_count += predicted_class_name == class_name
        count += 1
        print(count, true_count, round(true_count / count * 100, 2), class_name, predicted_class_name, image_file_path)
    print(true_count / count)
    return true_count / count


def attack_model(model_name, dataset, attack, attack_name):
    model, target_size, preprocess_input, decode_predictions, postprocess_output = create_model(model_name)

    # There may be different bounds, depends on the image preprocessing
    print("Preprocessing stage. Define model bounds")
    images_reader = create_imagenet_reader(dataset)
    bounds = None
    for image_file_path, _ in images_reader:
        test_image = load_preprocess_image(image_file_path, target_size, preprocess_input)
        if isinstance(test_image, np.ndarray):
            image_bounds = (np.min(test_image), np.max(test_image))
        else:
            image_bounds = (np.min(test_image.numpy()[0]), np.max(test_image.numpy()[0]))
        if not bounds:
            bounds = image_bounds
        else:
            bounds = (min(bounds[0], image_bounds[0]), max(bounds[1], image_bounds[1]))
    foolbox_model = TensorFlowModel(model, bounds=bounds)

    # epsilons = [0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.01, 0.1, 0.3, 0.5, 1.0]
    epsilons = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]  # DaViT
    epsilons = [x * 100 for x in epsilons]  # * 127.5 ?  EfficientNet

    attack_folder = create_folder_if_not_exists(f'attack_results/{model_name}/{attack_name}')

    counter = 0
    success_counter = 0
    images_reader = create_imagenet_reader(dataset)
    for image_file_path, class_name in images_reader:
        image_name = Path(image_file_path).stem
        image_folder = create_folder_if_not_exists(f'{attack_folder}/{class_name}')

        # predicted results
        test_image = load_preprocess_image(image_file_path, target_size, preprocess_input)
        result = model.predict(test_image)
        label = np.array([result.argmax()])

        # convert to eagerpy tensors and attack
        tensor_img = tf.convert_to_tensor(test_image)
        tensor_label = tf.convert_to_tensor(label)
        tensor_img, tensor_label = ep.astensors(tensor_img, tensor_label)
        raw_advs, clipped_advs, success = attack(foolbox_model, tensor_img, tensor_label, epsilons=epsilons)

        # save the best attacked image (if any) or copy the input image
        success_indexes = np.where(success.raw.numpy())
        if len(success_indexes[0]) > 0:
            # success_index = success_indexes[0][0]
            for success_index in success_indexes[0]:
                epsilon = epsilons[success_index]
                adv_image = clipped_advs[success_index]
                image_size = imread(image_file_path).shape[0:2]
                save_image(adv_image, f'{image_folder}/{image_name}_{epsilon}.jpeg', (image_size[1], image_size[0]),
                           postprocess_output)
            # save_histogram(adv, f'{attack_folder}/adv_{name}_hist_{i}.jpg')
            success_counter += 1
        else:
            shutil.copy(image_file_path, f'{image_folder}/{image_name}_-1.jpeg')
        counter += 1
        print(counter, success_counter)


def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def load_preprocess_image(image_file_path, target_size, preprocess_input_fn):
    img = image.load_img(image_file_path, target_size=target_size)
    processed_image = image.img_to_array(img)
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image = preprocess_input_fn(processed_image)
    return processed_image


def save_histogram(img, file_path):
    hist, bins = histogram(img.raw[0].numpy())
    plt.plot(bins, hist)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def save_image(img, filename, new_size=None, postprocess_output=None):
    ep_tensor: ep.Tensor = ep.astensor(img)
    np_image = ep_tensor.numpy()[0]
    if postprocess_output:
        np_image = postprocess_output(np_image)
    np_image = np_image.astype(uint8)
    pil_image = Image.fromarray(np_image)
    if new_size:
        pil_image = pil_image.resize(new_size)
    pil_image.save(filename)


def preprocess_tf(x):
    x /= 127.5
    x -= 1.
    return x


def postprocess_tf(x):
    return (x + 1) * 127.5


def postprocess_efficientnet(x):
    return x


if __name__ == '__main__':
    # TODO: read params from args
    model_name_arg = 'EfficientNetV2L'  # ResNet152V2, MobileNetV2, EfficientNetV2L, DaViT_B
    dataset_arg = 'imagenet_mini'
    evaluate_model_imagenet_dataset(model_name_arg, dataset_arg)
    attack_model(model_name_arg, dataset_arg, PGD(), 'PGD')
    evaluate_model_attacked_dataset('DaViT_B', model_name_arg, 'LinfBasicIterativeAttack')
    # file = open('Tables/acc.txt', 'a')
    # file = open('Tables/sm_acc_linear.txt', 'a')
    # file = open('Tables/sm_acc_median.txt', 'a')
    # file = open('Tables/sm_acc_ave_min.txt', 'a')
    # # file = open('Tables/sm_acc_gaus_min.txt', 'a')
    # file.write('PGD\n')
    # res = evaluate_model_attacked_dataset('DaViT_B', model_name_arg, 'PGD')
    # file.write(f'{res}\n')
    # file.write('\n')
    # file.write('LinfBasicIterativeAttack\n')
    # res = evaluate_model_attacked_dataset('DaViT_B', model_name_arg, 'LinfBasicIterativeAttack')
    # file.write(f'{res}\n')
    # file.write('\n')
    # file.write('FGSM\n')
    # res = evaluate_model_attacked_dataset('DaViT_B', model_name_arg, 'FGSM')
    # file.write(f'{res}\n')
    # file.write('\n')
    # file.close()

    # import os
    #
    # os.environ["PATH"] += os.pathsep + 'F:/Adversarial Attacks/Adversarial Attacks/plot_model/model_to_dot/Graphviz/bin/'
    #
    # model, _, _, _, _ = create_davit_model()
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # # print(model.summary())

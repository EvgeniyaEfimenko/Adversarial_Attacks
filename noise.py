from pathlib import Path
import glob
import os

import matplotlib.pyplot as plt
from keras.preprocessing import image
from skimage.io import imsave
from train import *
import foolbox

def main1():
    for folder in ['flowers_results', 'rice_results']:
        for attack in glob.glob(folder + '/*'):
            noise_folder = f'{attack}/noise'
            if not os.path.exists(noise_folder):
                os.mkdir(noise_folder)

            advs = glob.glob(f'{attack}/best_adversarial_pic' + '/*')
            i = 0
            for img in glob.glob(folder.split("_")[0] + '/*'):
                img_name = Path(img).stem
                if i == len(advs):
                    break
                if advs[i].find(img_name) != -1:
                    img = image.load_img(img)
                    img1 = image.load_img(advs[i])
                    img = image.img_to_array(img)
                    img1 = image.img_to_array(img1)
                    noise = img1 - img
                    imsave(f'{noise_folder}/{img_name}_noise_{Path(advs[i]).stem.split("_")[-1]}.jpg', noise)
                    i += 1

def main2():
    for attack in glob.glob('mobilenetv2_pic/*'):
        noise_folder = f'{attack}/noise'
        if not os.path.exists(noise_folder):
            os.mkdir(noise_folder)
        i = 1
        for img, adv in zip(glob.glob(f'{attack}/imgs' + '/*'), glob.glob(f'{attack}/adv' + '/*')):
                img = image.load_img(img)
                img1 = image.load_img(adv)
                img = image.img_to_array(img)
                img1 = image.img_to_array(img1)
                noise = img1 - img
                imsave(f'{noise_folder}/noise_{i}.jpg', noise)
                i += 1


def main3():
    for attack in glob.glob('mobilenetv2_pic/*'):
        noise = np.array(glob.glob(f'{attack}/noise/*'))
        arr = []
        for noise_img in noise:
            img = image.load_img(noise_img, target_size=(100,100))
            img = image.img_to_array(img)
            img = img / 255
            # img = np.expand_dims(img, axis=0)
            arr.append(img)
        arr = np.array(arr)
        # tensor_noise = tf.convert_to_tensor(noise)
        # tensor_img, tensor_label = ep.astensors(tensor_img, tensor_label)
        foolbox.plot.images(arr)
        plt.tight_layout()
        plt.savefig(f'{attack}/noise.jpg')

def main():
    info_class = Flowers()
    img_adv = image.load_img('flowers_results\\FGSM\\best_adversarial_pic\\tulip (17)_0.001.jpg', target_size=info_class.target_size)
    adv_image = image.img_to_array(img_adv)
    adv_image = np.expand_dims(adv_image, axis=0)
    adv_image = adv_image / 255.0

    img_real = image.load_img('flowers\\tulip (17).jpg', target_size=info_class.target_size)
    img_real = image.img_to_array(img_real)
    img_real = np.expand_dims(img_real, axis=0)
    img_real = img_real / 255.0

    model = load_model('модели\\flowers a = 75\\' + info_class.model_file_name)
    result = model.predict(img_real)
    predicted_class_real = result.argmax()  # np.where(result == 1)[1]
    probability = result[0][predicted_class_real] * 100
    print(f'predicted before attack: {info_class.class_dict[predicted_class_real]}  with probability = {probability} %')

    result = model.predict(adv_image)
    predicted_class_real = result.argmax()  # np.where(result == 1)[1]
    probability = result[0][predicted_class_real] * 100
    print(f'predicted after attack: {info_class.class_dict[predicted_class_real]}  with probability = {probability} %')


if __name__ == '__main__':
    # info_class = Rice()
    # # model = load_model('модели\\flowers a = 75\\' + info_class.model_file_name)
    # model = load_model('модели\\rice Accuracy  99.29777979850769\\' + info_class.model_file_name)
    # print(model.summary())
    main()

import numpy as np
from keras.models import load_model
from train import *  # CIFAR10, Rice, Flowers
from keras.preprocessing import image
from pathlib import Path
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def main():
    # info_class = Rice()
    # model = load_model('модели\\rice Accuracy  99.29777979850769\\' + info_class.model_file_name)
    info_class = Flowers()
    model = load_model('F:\\Adversarial Attacks\\Adversarial Attacks\\модели\\flowers a = 75\\'+info_class.model_file_name)
    count = 0

    # for img_file in glob.glob(info_class.path_to_test_pic+'*'):
    for img_file in glob.glob('flowers_results\\FGSM\\best_adversarial_pic'+'/*'):
        # Подготовка тестового изображения
        # image_path = 'F:\\Adversarial Attacks\\Adversarial Attacks\\airplane.jpg'
        # img = image.load_img(img_file, target_size=(32, 32))
        print(img_file)
        img = image.load_img(img_file, target_size=info_class.target_size)
        test_image = image.img_to_array(img)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0
        result = model.predict(test_image)
        print(result)
        predicted_class = result.argmax()   # np.where(result == 1)[1]
        print('probability = ', result[0][predicted_class] * 100)

        # if len(predicted_class) != 0:
        print('the predicted class is - ', info_class.class_dict[predicted_class])
        # else:
        #     print('the predicted class is - could not recognise')
        print('the real class is - ', Path(img_file).stem.split()[0])
        # tmp_dict = {value: key for key, value in info_class.class_dict.items()}
        y = []
        for i in range(0, len(info_class.class_dict)):
            if info_class.class_dict_name_ind[Path(img_file).stem.split()[0]] == i:
                y.append(1.0)
            else:
                y.append(0.0)
        scores = model.evaluate(test_image, np.array([y]), verbose=0)
        print("Accuracy: ", scores[1] * 100)
        print()
        if scores[1] * 100 > 0.0:
            count += 1
    print()
    print('number of right prediction = ', count)


if __name__ == '__main__':
    main()
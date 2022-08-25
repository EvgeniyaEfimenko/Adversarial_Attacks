from pathlib import Path
import os
import glob
import shutil


def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def choose_best_pic(dataset_path, new_folder):
    for img_folder in glob.glob(dataset_path):
        class_name = Path(img_folder).name
        tmp_dict = dict()
        for img_file in glob.glob(img_folder+'/*'):
            img_name = Path(img_file).stem
            split_res = img_name.rsplit('_', 1)
            img_orig_name = split_res[0]
            eps = float(split_res[1])
            if eps == -1.0:
                eps = -1

            if img_orig_name in tmp_dict:
                tmp_dict[img_orig_name].append(eps)
            else:
                tmp_dict[img_orig_name] = [eps]

        for name in tmp_dict:
            e = min(tmp_dict[name])
            dst = create_folder_if_not_exists(f'{new_folder}/{class_name}/')
            src = f'{img_folder}/{name}_{e}.jpeg'
            shutil.copy(src, dst)


def count_of_success(datafolder):
    all_count = 0
    success_count = 0
    for img_folder in glob.glob(datafolder):
        for img_file in glob.glob(img_folder + '/*'):
            img_name = Path(img_file).stem
            eps = img_name.rsplit('_', 1)[1]
            all_count += 1
            if eps != '-1':
                success_count += 1
    return success_count, all_count


def main2():
    filepath = create_folder_if_not_exists('Tables/') + 'count_of_success_attacks.txt'
    file = open(filepath, 'a')
    for model_name in glob.glob('best_attack_results/*'):
        for attack_name in glob.glob(f'{model_name}/*'):
            adversarial_imgs_path = f'{attack_name}/*'
            success_count, all_count = count_of_success(adversarial_imgs_path)
            file.write(f'{Path(model_name).stem}   {Path(attack_name).stem}  {success_count} / {all_count} \n')

def main3():
    for model_name in glob.glob('smoothing/attacked/*'):
        for attack_name in glob.glob(f'{model_name}/*'):
            for filter in glob.glob(f'{attack_name}/*'):
                if Path(filter).stem == 'linear' or Path(filter).stem == 'median':
                    continue
                adversarial_imgs_path = f'{filter}/*'
                new_folder = f'best_smoothing_attack_results/{Path(model_name).stem}/{Path(attack_name).stem}/{Path(filter).stem}/'
                choose_best_pic(adversarial_imgs_path, new_folder)

def main():
    for model_name in glob.glob('attack_results/*'):
        for attack_name in glob.glob(f'{model_name}/*'):
            adversarial_imgs_path = f'{attack_name}/*'
            new_folder = f'best_attack_results_max/{Path(model_name).stem}/{Path(attack_name).stem}/'
            choose_best_pic(adversarial_imgs_path, new_folder)


if __name__ == '__main__':
    # main()
    # main2()
    main3()


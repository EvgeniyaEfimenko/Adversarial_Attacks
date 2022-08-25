from pathlib import Path
import glob
import pandas as pd
import csv
import os

epsilons = [0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.01, 0.1, 0.3, 0.5, 1.0]

def main():
    for folder in ['flowers_results', 'rice_results']:
        amount_eps = pd.DataFrame({'epsilons': [0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015,
                                                0.002, 0.003, 0.01, 0.1, 0.3, 0.5, 1.0, 'success']})
        for attack in glob.glob(folder + '/*'):
            attack_name = Path(attack).stem
            eps_dict = {}
            count = 0
            for img in glob.glob(f'{attack}/best_adversarial_pic'+'/*'):
                eps = float(Path(img).stem.split('_')[-1])
                if eps not in eps_dict:
                    eps_dict[eps] = 1
                else:
                    eps_dict[eps] += 1
                if eps != 0.0:
                    count += 1
                    for higher_eps in epsilons[epsilons.index(eps)+1:]:
                        if higher_eps not in eps_dict:
                            eps_dict[higher_eps] = 1
                        else:
                            eps_dict[higher_eps] += 1
            for eps in epsilons:
                if eps not in eps_dict:
                    eps_dict[eps] = 0

            eps_dict = dict(sorted(eps_dict.items(), key=lambda x: x[0]))
            column = [val for val in eps_dict.values()]
            column.append(count)
            amount_eps[attack_name] = column

        if not os.path.exists('Tables'):
            os.mkdir('Tables')
        amount_eps.to_excel(f'Tables/{folder}.xlsx')
        
def main2():
        amount_eps = pd.DataFrame({'epsilons': [0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015,
                                                0.002, 0.003, 0.01, 0.1, 0.3, 0.5, 1.0, 'success']})
        for attack in glob.glob(folder + '/*'):
            attack_name = Path(attack).stem
            eps_dict = {}
            count = 0
            for img in glob.glob(f'{attack}/best_adversarial_pic'+'/*'):
                eps = float(Path(img).stem.split('_')[-1])
                if eps not in eps_dict:
                    eps_dict[eps] = 1
                else:
                    eps_dict[eps] += 1
                if eps != 0.0:
                    count += 1
                    for higher_eps in epsilons[epsilons.index(eps)+1:]:
                        if higher_eps not in eps_dict:
                            eps_dict[higher_eps] = 1
                        else:
                            eps_dict[higher_eps] += 1
            for eps in epsilons:
                if eps not in eps_dict:
                    eps_dict[eps] = 0

            eps_dict = dict(sorted(eps_dict.items(), key=lambda x: x[0]))
            column = [val for val in eps_dict.values()]
            column.append(count)
            amount_eps[attack_name] = column

        if not os.path.exists('Tables'):
            os.mkdir('Tables')
        amount_eps.to_excel(f'Tables/{folder}.xlsx')


if __name__ == '__main__':
    main()
    main2()

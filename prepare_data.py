import os
import shutil
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
import fire

def prepare_cars():
    PATH = Path('data/cars')
    annots = loadmat(PATH/'cars_annos.mat')
    trn_ids, trn_classes, val_ids, val_classes = [], [], [], []
    for annot in annots['annotations'][0]:
        if int(annot[6]) == 1:
            val_classes.append(int(annot[5]))
            val_ids.append(annot[0][0])
        else:
            trn_classes.append(int(annot[5]))
            trn_ids.append(annot[0][0])
    df_trn = pd.DataFrame({'fname': trn_ids, 'class': trn_classes}, columns=['fname', 'class'])
    df_val = pd.DataFrame({'fname': val_ids, 'class': val_classes}, columns=['fname', 'class'])
    combined = df_trn.append(df_val)
    combined.reset_index(inplace=True)
    combined.drop(['index'], 1, inplace=True)
    combined.to_csv(PATH/'annots.csv', index=False)

def prepare_cifar10():
    PATH = Path('data/cifar10')
    TMP_PATH = PATH/'cifar'
    shutil.move(TMP_PATH/'train', PATH/'train')
    shutil.move(TMP_PATH/'test', PATH/'test')
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for d in ['train', 'test']:
        for clas in classes:
            os.mkdir(PATH/d/clas)
            fnames = list(PATH.glob(f'{d}/*{clas}.png'))
            for fname in fnames: shutil.move(fname, PATH/d/clas/str(fname)[len(str(PATH/d))+1:])
    shutil.rmtree(TMP_PATH)

def prepare_wt2():
    PATH = Path('data/wikitext')
    TMP_PATH = PATH/'wikitext-2'
    for name in ['train', 'valid', 'test']:
        shutil.move(TMP_PATH/f'wiki.{name}.tokens', PATH/f'wiki.{name}.tokens')
    shutil.rmtree(TMP_PATH)
    

def prepare_data():
    prepare_cars()
    prepare_cifar10()
    prepare_wt2()

if __name__ == '__main__': fire.Fire(prepare_data)
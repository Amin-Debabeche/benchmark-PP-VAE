import os
import numpy as np
import pandas as pd
from featurizer import OneHotFeaturizer

task = 'zinc'
prop = 'logP'

for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'data/{task}_{split}.csv')
    smiles = list(df.smiles)
    gaps = np.array(list(df[prop]))

    ohf = OneHotFeaturizer()
    oh_smiles = ohf.featurize(smiles)

    print(oh_smiles.shape)
    np.savez_compressed(f'./data/{task}_smiles_{split}.npz', arr=oh_smiles)
    np.savez_compressed(f'./data/{task}_{prop}_{split}.npz', arr=gaps)

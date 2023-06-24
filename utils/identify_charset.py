import pandas as pd

df_train = pd.read_csv('data/train_tki/data-dft_train.csv')
df_val = pd.read_csv('data/val_tki/data-dft_val.csv')
df_test = pd.read_csv('data/test_tki/data-dft_test.csv')

smiles = list(df_train.smiles) + list(df_val.smiles) + list(df_test.smiles)

charset = []
for smi in smiles:
    for char in smi:
        if char not in charset:
            charset.append(char)

print(charset, len(charset))

max_len = 0
for smi in smiles:
    if len(smi) > max_len:
        max_len = len(smi)

print(max_len)

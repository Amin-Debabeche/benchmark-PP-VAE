import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split_data(data, test_ratio, val_ratio, output_folder):
    train, test = train_test_split(data, test_size=test_ratio, random_state=42)
    val, test = train_test_split(test, test_size=val_ratio, random_state=42)

    train.to_csv(os.path.join(output_folder, 'zinc_train.csv'), index=False)
    val.to_csv(os.path.join(output_folder, 'zinc_val.csv'), index=False)
    test.to_csv(os.path.join(output_folder, 'zinc_test.csv'), index=False)


data = pd.read_csv('./data/250k_rndm_zinc_drugs_clean_3.csv')
split_data(data, 0.2, 0.5, './data')

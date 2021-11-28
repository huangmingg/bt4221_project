from typing import Tuple
import numpy as np
import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import SMOTE


def parse_input_file(df_filepath: os.path, img_directory: os.path, label_filepath: os.path) -> pd.DataFrame:
    df = pd.read_table(df_filepath, skiprows=1, delim_whitespace=True)
    label_classes = pd.read_csv(label_filepath)
    df['image_name'] = df['image_name'].apply(lambda x: os.path.join(img_directory, x.split('/')[1], x.split('/')[2]))
    df.columns = ['filename', 'class']
    df = df.merge(label_classes, on='class', how='left')
    return df


def transform_df(df: pd.DataFrame, use_coarse: bool = True, sampling_strategy: str = None, random_state: int=4221) -> pd.DataFrame:
    class_to_use = 'class' if not use_coarse else 'coarse_class'
    df[class_to_use] = df[class_to_use].astype(str)
    df = df[['filename', class_to_use]]
    if not sampling_strategy:
        x_res, y_res = df['filename'].values.reshape(-1, 1), df[class_to_use].values
    elif sampling_strategy == 'under':
        rus = RandomUnderSampler(sampling_strategy='not minority', random_state=random_state)
        x_res, y_res = rus.fit_resample(df['filename'].values.reshape(-1, 1), df[class_to_use].values)
    elif sampling_strategy == 'over':
        smote = SMOTE(sampling_strategy='not majority', random_state=random_state)
        x_res, y_res = smote.fit_resample(df['filename'].values.reshape(-1, 1), df[class_to_use].values)
    else:
        print(f'sampling strategy {sampling_strategy} invalid')
        return pd.DataFrame()
    x = pd.DataFrame(x_res, columns=['filename'])       
    y = pd.DataFrame(y_res, columns=[class_to_use])
    df = x.join(y)
    df.columns = ['filename', 'class']    
    return df


def train_val_test_split(df: pd.DataFrame, label: str='class', val_size: float=0.2, test_size: float=0.2, 
    stratify: bool=True, random_state: int=4221) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(random_state)
    if val_size + test_size >= 1:
        print("invalid parameters")
        return
    train_size = 1 - val_size - test_size
    if not stratify:
        indices = np.array(df.index)
        c = len(indices)
        np.random.shuffle(indices)
        fti, sti = int(train_size*c), int((train_size+val_size)*c)
        train_idx, val_idx, test_idx = indices[:fti], indices[fti: sti], indices[sti:]
        return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

    else:
        for l in df[label].value_counts().index.tolist():
            indices = np.array(df.loc[df[label] == str(l)].index)
            c = len(indices)
            np.random.shuffle(indices)
            fti, sti = int(train_size*c), int((train_size+val_size)*c)
            train_idx, val_idx, test_idx = indices[:fti], indices[fti: sti], indices[sti:]
            df.loc[train_idx, 'split'] = 1
            df.loc[val_idx, 'split'] = 2
            df.loc[test_idx, 'split'] = 3
        return df.loc[df['split'] == 1][['filename', 'class']], df.loc[df['split'] == 2][['filename', 'class']], df.loc[df['split'] == 3][['filename', 'class']]

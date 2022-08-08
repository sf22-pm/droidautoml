import pandas as pd

from argparse import ArgumentParser
import sys
from methods.utils import get_base_parser, get_dataset, get_X_y, get_filename
def run_data_cleaner(dataset_df):

    try:
        dataset_dirty = dataset_df
        
    except BaseException as e:
        print('Exception: {}'.format(e))
        exit(1)
    
    print("remove irrelevant coluns and duplicates values")
    for col in dataset_dirty.columns:
        if len(dataset_dirty[col].unique()) == 0:
            dataset_dirty.drop(col,inplace=True,axis=1)
            dataset_dirty = dataset_dirty.drop_duplicates(keep='first')
    return dataset_dirty

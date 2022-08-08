import pandas as pd
import numpy  as np
import argparse
from utils import get_base_parser, get_dataset
def cleaner(args):

    try:
        dataset_dirty = get_dataset(args)
        
    except BaseException as e:
        print('Exception: {}'.format(e))
        exit(1)
    
    print("remove irrelevant coluns and duplicates values")
    for col in dataset_dirty.columns:
        if len(dataset_dirty[col].unique()) == 0:
            dataset_dirty.drop(col,inplace=True,axis=1)
            dataset_dirty = dataset_dirty.drop_duplicates(keep='first')
    return dataset_dirty
if __name__ == '__main__':
    cleaner()

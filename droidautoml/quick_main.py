from droidautoml.main import make_classifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pandas as pd
import timeit
from os.path import exists, basename
import sys
from datetime import datetime

import argparse
from lib.help import *
import subprocess
from termcolor import colored
import pickle


def get_current_datetime(format="%Y%m%d%H%M%S"):
    return datetime.now().strftime(format)
    
def get_output_file(args):
        if args.output_sigpid==True:
            output_file = args.output_sigpid
            print(output_file)

        elif args.output_rfg==True:
            output_file = args.output_sigpid
            print(output_file)
        else:
            args.output_file
        return args.output_file

def get_quick(args,dataset_df):

    
    try:
        dataset_file_path = args.dataset
        dataset_name = basename(dataset_file_path)
        start_time = timeit.default_timer()     	
        dataset_df = pd.read_csv(dataset_file_path, encoding='utf8')
        print("dataset utilizado",dataset_df.shape)
    except BaseException as e:
        print('Exception: {}'.format(e))
        exit(1)
        
    print(colored("Selecting best algorithms and Hyperparams Optimizer...", 'blue'))
    start_time = timeit.default_timer()
    estimator = make_classifier()
    data = estimator.prepare_data(dataset_df)

    y = data['class']
    X = data.drop(['class'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    estimator.fit(X_train, y_train)
    print(colored("Best_Model", 'blue'))
    print(colored(estimator.best_model, 'blue'))
    predictions = estimator.predict(X_test)
    print(colored("Salve_Model", 'blue'))
    pickle.dump(estimator, open(f"{args.output_file}_model_trained_{get_current_datetime()}_{dataset_name}.pkl", 'wb'))
    m, s = divmod(timeit.default_timer() - start_time, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)

    pd.DataFrame({
        "best_model": estimator.best_model,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "dataset" : dataset_name,
        "execution_time" : time_str
    }, index=[0]).to_csv(f"{args.output_file}_quickautoml_{get_current_datetime()}_{dataset_name}", index=False)

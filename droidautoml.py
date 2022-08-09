#!/usr/bin/python3

from droidautoml.main import make_classifier

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pandas as pd
import timeit
from os.path import exists, basename
import sys
from datetime import datetime

from lib.Log import *
import argparse
import methods.datacleaner as cleaner
import methods.sigpid.sigpid_main as sigpid
import methods.rfg.rfg as rfg
import methods.jowmdroid.jowmdroid as jowmdroid
from lib.help import *
import subprocess
from termcolor import colored
import pickle
import warnings
warnings.filterwarnings("ignore")

def parse_args(argv):
    parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                    usage="python3 quick.py --dataset <Dataset> [opção]", add_help=False)
    pos_opt = parse.add_argument_group("Opções")
    pos_opt.add_argument("--about",action="store_true",help="Tool information")
    pos_opt.add_argument("--help", action="store_true", default=False, help="Show usage parameters")
    pos_opt.add_argument("--dataset", metavar="", help="dataset (e.g. datasets/DrebinDatasetPermissoes.csv)")
    pos_opt.add_argument(
        '--use-select-features', metavar='FEATURE TYPE: permissions or api-calls or mult-features',
        help="--use-select-features permissions",
        choices=['permissions', 'api-calls', 'mult-features'], type = str)

    # PADRAO PARA TODOS
    pos_opt.add_argument( '--sep', metavar = 'SEPARATOR', type = str, default = ',',
        help = 'Dataset feature separator. Default: ","')
    pos_opt.add_argument('--class-column', type = str, default="class", metavar = 'CLASS_COLUMN',
        help = 'Name of the class column. Default: "class"')
    pos_opt.add_argument('--n-samples', type=int,
        help = 'Use a subset of n samples from the dataset. By default, all samples are used.')
    pos_opt.add_argument('--output-results', metavar = 'OUTPUT_FILE', type = str, default = 'droidautoml_results.csv',
        help = 'Output metrics (e.g. acuracy, recall,time) Default: quick_results.csv')
    pos_opt.add_argument('--output-model', metavar = 'OUTPUT_FILE', type = str, default = 'model_serializable',
        help = 'Output model ML serializable. Default: model_serializable')
        
    
    feature_selection_args =  any([x == 'permissions' for x in argv])
   
    group_sigpid = parse.add_argument_group('Additional Parameters for SigPID')
    if feature_selection_args:
        group_sigpid.add_argument('--output-sigpid', metavar = 'OUTPUT_FILE', type = str, default = 'subset_sigpid.csv',
        help = 'Output file name. Default: subset_sigpid.csv')
        
        #mudar output-file no sigpid para output-sigpid
        
        
    feature_selection_args =  any([x == 'api-calls' for x in argv])
   
    group_rfg = parse.add_argument_group('Additional Parameters for RFG')
    if feature_selection_args:
        group_rfg.add_argument('--output-rfg', metavar = 'OUTPUT_FILE', type = str, default = 'subset_rfg.csv',
        help = 'Output file name. Default: subset_rfg.csv')
        
        #RFG  
        group_rfg.add_argument(
        '-i', '--increment', 
        help = 'Increment. Default: 20',
        type = int, 
        default = 20)
        group_rfg.add_argument(
        '-f',
        metavar = 'LIST',
        help = 'List of number of features to select. If provided, Increment is ignored. Usage example: -f="10,50,150,400"',
        type = str, 
        default = "")
        group_rfg.add_argument(
        '-k', '--n-folds',
        help = 'Number of folds to use in k-fold cross validation. Default: 10.',
        type = int, 
        default = 10)
        group_rfg.add_argument('--feature-selection-only', action='store_true',
        help="If set, the experiment is constrained to the feature selection phase only. The program always returns the best K features, where K is the maximum value in the features list.")  
 
 
 
    feature_selection_args =  any([x == 'mult-features' for x in argv])
   
    group_jowmdroid = parse.add_argument_group('Additional Parameters for JOWMDROID')
    if feature_selection_args:
        group_jowmdroid.add_argument('--output-jowmdroid', metavar = 'OUTPUT_FILE', type = str, default = 'subset_jowmdroid.csv',
        help = 'Output file name. Default: subset_jowmdroid.csv')
        group_jowmdroid.add_argument('--exclude-hyperparameter', action='store_false',
        help="If set, the ML hyperparameter will be excluded in the Differential Evolution. By default it's included")
        group_jowmdroid.add_argument( '-m', '--mapping-functions', metavar = 'LIST', type = str,
        default = "power, exponential, logarithmic, hyperbolic, S_curve",
        help = 'List of mapping functions to use. Default: "power, exponential, logarithmic, hyperbolic, S_curve"')
        group_jowmdroid.add_argument( '-t', '--mi-threshold', type = float, default = 0.2,
        help = 'Threshold to select features with Mutual Information. Default: 0.2. Only features with score greater than or equal to this value will be selected')
        group_jowmdroid.add_argument('--train-size', type = float, default = 0.8,
        help = 'Proportion of samples to use for train. Default: 0.8')
        group_jowmdroid.add_argument('--cv', metavar = 'INT', type = int, default = 5,
        help="Number of folds to use in cross validation. Default: 5")
        group_jowmdroid.add_argument('--feature-selection-only', action='store_false',
        help="If set, the experiment is constrained to the feature selection phase only.")
        
    print(colored(logo, 'green'))
    
    try:
       getopt = parse.parse_args(argv)
    except:
        parse.print_help()
        sys.exit(1)
    return getopt

def get_current_datetime(format="%Y%m%d%H%M%S"):
    return datetime.now().strftime(format)

def show_about():
    print("DrodAutoML v0.1")

def cleaner(dataset):
    start_time = timeit.default_timer() 
   
    Log.info("STAGE 1: DATA CLEANING ...")
    #print(colored("APPLYING DATA CLEANER...", 'blue', attrs=['bold']))
    dataset_df = pd.read_csv(dataset, encoding='utf8')
    linhas = dataset_df.shape[0]
    colunas= dataset_df.shape[1]-1
    print("Dataset Size: ",linhas,",",colunas)
    print("Removing irrelevant columns")
    for col in dataset_df.columns:
        if len(dataset_df[col].unique()) == 0:
            dataset_df.drop(col,inplace=True,axis=1)
        
    #print("Removing duplicates values")
    #dataset_df=dataset_df.drop_duplicates(keep='first')
   
    print("Removing NaN and Null values")
    dataset_df.dropna(axis=1)
   
    print("There is NaN data?->",dataset_df.isna().values.any())
    print("There is null data?->",dataset_df.isnull().values.any())
    
    dataset_df.shape
    m, s = divmod(timeit.default_timer() - start_time, 60)
    h, m = divmod(m, 60)
    time_str_cleaner = "%02d:%02d:%02d" % (h, m, s)
    print("Elapsed Time: ",time_str_cleaner)     
    return dataset_df 

if __name__ == "__main__":
    getopt = parse_args(sys.argv[1:])
    start_time_geral = timeit.default_timer() 
    if len(sys.argv) < 2:
        print ("Usage: " + sys.argv[0] + " -h")
        exit(1)
    if getopt.about:
        show_about()
        exit(1)
    try:
        dataset_file_path = getopt.dataset
        dataset_name = basename(dataset_file_path)
        dataset_df = cleaner(getopt.dataset)#pd.read_csv(dataset_file_path, encoding='utf8')
    except BaseException as e:
        Log.high("Error", e)
        exit(1)
    start_time = timeit.default_timer() 
    if getopt.use_select_features == 'permissions':
        Log.info("STAGE 2: FEATURE ENGINEERING (FEATURE SELECTION - PERMISSIONS)")
        dataset_df = sigpid.run(getopt)
        m, s = divmod(timeit.default_timer() - start_time, 60)
        h, m = divmod(m, 60)
        time_str_features = "%02d:%02d:%02d" % (h, m, s)
        print("Elapsed Time: ",time_str_features)
    elif getopt.use_select_features == 'api-calls':
        Log.info("STAGE 2: FEATURE ENGINEERING (FEATURE SELECTION - API_CALLS)")
        dataset_df = rfg.rfg(getopt)
        m, s = divmod(timeit.default_timer() - start_time, 60)
        h, m = divmod(m, 60)
        time_str_features = "%02d:%02d:%02d" % (h, m, s)
        print("Elapsed Time: ",time_str_features)
    elif getopt.use_select_features == 'mult-features':
        Log.info("STAGE 2: FEATURE ENGINEERING (FEATURE SELECTION - MULT-FEATURES)")
        dataset_df = jowmdroid.jowmdroid(getopt)
        m, s = divmod(timeit.default_timer() - start_time, 60)
        h, m = divmod(m, 60)
        time_str_features = "%02d:%02d:%02d" % (h, m, s)
        print("Elapsed Time: ",time_str_features)
        

    Log.info("STAGE 3: SELECTING ALGORITHMS AND OPTIMIZING HYPER-PARAMETERS")
    start_time = timeit.default_timer()
    estimator = make_classifier()
    selected_linhas = dataset_df.shape[0]
    selected_colunas= dataset_df.shape[1]-1
    print("Dataset Size: ",selected_linhas,",",selected_colunas)
    data = estimator.prepare_data(dataset_df)

    y = data[getopt.class_column]
    X = data.drop([getopt.class_column], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    estimator.fit(X_train, y_train)
    Log.info("STATE 4: EVALUATION AND BEST MODEL SELECTION")
    print(estimator.best_model)
    predictions = estimator.predict(X_test)
    print(f"{getopt.output_model}_trained_{get_current_datetime()}_{dataset_name}.pkl")
    #Log.info("BEST MODEL CREATED SUCCESSFULLY IN PKL FILE.")
    pickle.dump(estimator, open(f"{getopt.output_model}_trained_{get_current_datetime()}_{dataset_name}.pkl", 'wb'))
   
    m, s = divmod(timeit.default_timer() - start_time, 60)
    h, m = divmod(m, 60)
    time_str_model = "%02d:%02d:%02d" % (h, m, s)
 
    m, s = divmod(timeit.default_timer() - start_time_geral, 60)
    h, m = divmod(m, 60)
    time_str_geral = "%02d:%02d:%02d" % (h, m, s)
    
    pd.DataFrame({
        "best_model": estimator.best_model,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "dataset" : dataset_name,
        "execution_time" : time_str_geral
    }, index=[0]).to_csv(f"{getopt.output_results}_droidautoml_{get_current_datetime()}_{dataset_name}", index=False)

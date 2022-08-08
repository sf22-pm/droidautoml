from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from argparse import ArgumentParser
import sys
from methods.utils import get_base_parser, get_dataset, get_X_y

def parse_args(argv):
    base_parser = get_base_parser()
    parser = ArgumentParser(parents=[base_parser])
    parser.add_argument(
        '-i', '--increment', 
        help = 'Increment. Default: 20',
        type = int, 
        default = 20)
    parser.add_argument(
        '-f',
        metavar = 'LIST',
        help = 'List of number of features to select. If provided, Increment is ignored. Usage example: -f="10,50,150,400"',
        type = str, 
        default = "")
    parser.add_argument(
        '-k', '--n-folds',
        help = 'Number of folds to use in k-fold cross validation. Default: 10.',
        type = int, 
        default = 10)
    parser.add_argument('--feature-selection-only', action='store_true',
        help="If set, the experiment is constrained to the feature selection phase only. The program always returns the best K features, where K is the maximum value in the features list.")
    args = parser.parse_args(argv)
    return args

def run_experiment(X, y, classifiers, is_feature_selection_only = False,
                   score_functions=[chi2, f_classif], 
                   n_folds=10,
                   k_increment=20,
                   k_list=[]):
    """
    Esta função implementa um experimento de classificação binária usando validação cruzada e seleção de características. 
    Os "classifiers" devem implementar as funções "fit" e "predict", como as funções do Scikit-learn.
    Se o parâmetro "k_list" for uma lista não vazia, então ele será usado como a lista das quantidades de características a serem selecionadas. 
    """
    results = []
    best_features = []
    if(len(k_list) > 0):
        k_values = k_list
    else:
        k_values = range(1, X.shape[1], k_increment)
    for k in k_values:
        if(k > X.shape[1]):
            print(f"Warning: skipping K = {k}, since it's greater than the number of features available ({X.shape[1]})")
            continue
        print("K =", k)
        for score_function in score_functions:
            if(k == max(k_values)): 
                selector = SelectKBest(score_func=score_function, k=k).fit(X, y)
                X_selected = X.iloc[:, selector.get_support(indices=True)].copy()
                feature_scores_sorted = pd.DataFrame(list(zip(X_selected.columns.values.tolist(), selector.scores_)), columns= ['features','score']).sort_values(by = ['score'], ascending=False)
                X_selected_sorted = X_selected.loc[:, list(feature_scores_sorted['features'])]
                X_selected_sorted['class'] = y

                best_features.append({
                    'score_function' : score_function.__name__, 
                    'selected_dataset' : X_selected_sorted,
                    'k': k 
                })
                if(X_selected.shape[1] == 1):
                    print("AVISO: 0 features selecionadas")
            if(is_feature_selection_only):
                continue
            X_selected = SelectKBest(score_func=score_function, k=k).fit_transform(X, y)
            kf = KFold(n_splits=n_folds, random_state=256, shuffle=True)
            fold = 0
            for train_index, test_index in kf.split(X_selected):
                X_train, X_test = X_selected[train_index], X_selected[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                for classifier_name, classifier in classifiers.items():
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    results.append({'n_fold': fold,
                                    'k': k,
                                    'score_function':score_function.__name__,
                                    'algorithm': classifier_name,
                                    'accuracy': report['accuracy'],
                                    'precision': report['macro avg']['precision'], 
                                    'recall': report['macro avg']['recall'],
                                    'f-measure': report['macro avg']['f1-score']
                                })
                fold += 1
            
    return pd.DataFrame(results), best_features

def rfg(args):    
    parsed_args = args
    X, y = get_X_y(parsed_args, get_dataset(parsed_args))
    k_list = [int(value) for value in parsed_args.f.split(",")] if parsed_args.f != "" else []

    classifiers = {
        'RandomForest': RandomForestClassifier(),
    }

    results, best_features = run_experiment(
        X, y, 
        classifiers, 
        n_folds = parsed_args.n_folds, 
        k_increment = parsed_args.increment, 
        k_list=k_list, 
        is_feature_selection_only=parsed_args.feature_selection_only
    )

    if(not parsed_args.feature_selection_only):
        results.to_csv(parsed_args.output_rfg, index=False)
    
    for best_feature in best_features:
        k = best_feature['k']
        score_function = best_feature['score_function']
        file_name = f"top_{k}_features_with_{score_function}_{parsed_args.output_rfg}.csv"
        best_feature['selected_dataset'].to_csv(file_name, index=False)
    print("done")
    return best_feature['selected_dataset']
    exit(0)

if __name__ == '__main__':
    rfg()

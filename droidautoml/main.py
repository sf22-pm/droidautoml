from droidautoml.adapters import SKLearnModelsSupplier
from droidautoml.estimators import Classifier
from droidautoml.feature_engineering import PandasFeatureEngineer
from droidautoml.hyperparameter_optimizer import OptunaHyperparamsOptimizer
from droidautoml.preprocessors import PandasDataPreprocessor
from droidautoml.entities import NaiveModel, Hyperparameter
#from quickautoml.methods.SigPID.sigpid import NaiveModel, Hyperparameter

import pandas as pd


def make_classifier():
  data_preprocessor = PandasDataPreprocessor()
  feature_engineer = PandasFeatureEngineer()
  hyperparameter_optimizer = OptunaHyperparamsOptimizer('accuracy')
  models_supplier = SKLearnModelsSupplier()
  return Classifier(data_preprocessor, feature_engineer, models_supplier, hyperparameter_optimizer)

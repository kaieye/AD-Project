import sklearn
import os
import csv
import collections
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.calibration import CalibratedClassifierCV
from utils import COG_thresholding, ADD_thresholding,read_json
from model_wrappers import Multask_Wrapper
from nonImg_model_wrappers import NonImg_Model_Wrapper, Fusion_Model_Wrapper
from utils import read_json, plot_shap_bar, plot_shap_heatmap, plot_shap_beeswarm
from performance_eval import whole_eval_package
from multiprocessing import Process

class GWAS_Task_Wrapper:
    def __init__(self, tasks, main_config, task_config, seed):

        # --------------------------------------------------------------------------------------------------------------
        # some constants
        self.seed = seed  # random seed number
        self.model_name = main_config['model_name']  # user assigned model_name, will create folder using model_name to log
        self.csv_dir = main_config['csv_dir']  # data will be loaded from the csv files specified in this directory
        self.config = task_config  # task_config contains task specific info
        self.n_tasks = len(tasks)  # number of tasks will be trained
        self.tasks = tasks  # a list of tasks names to be trained
        self.features = task_config['features'] # a list of features

        # --------------------------------------------------------------------------------------------------------------
        # folders preparation to save checkpoints of model weights *.pth
        self.checkpoint_dir = './checkpoint_dir/{}/'.format(self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        # folders preparation to save tensorboard and other logs
        self.tb_log_dir = './tb_log/{}/'.format(self.model_name)
        if not os.path.exists(self.tb_log_dir):
            os.mkdir(self.tb_log_dir)

        # --------------------------------------------------------------------------------------------------------------
        # initialize models
        self.models = []  # note: self.models[i] is for the i th task
        self.init_models([task_config[t]['name'] for t in tasks])

        # --------------------------------------------------------------------------------------------------------------
        # initialize data
        self.train_data = []                      # note: self.train_data[i] contains the
        self.imputer = None
        self.load_preprocess_data()               #       features and labels for the i th task

    def train(self):
        for i, task in enumerate(self.tasks):
            X, Y = self.train_data[i].drop([task], axis=1), self.train_data[i][task]
            self.models[i].fit(X, Y)
            print(task + ' model training is done!')

    def get_optimal_thres(self, csv_name='GWAS_valid'):
        self.gen_score(stages=[csv_name])
        thres = {}
        for i, task in enumerate(self.tasks):
            if task == 'COG' and self.config['COG']['type'] == 'reg':
                thres['NC'], thres['DE'] = COG_thresholding(self.tb_log_dir + csv_name + '_eval.csv')
            elif task == 'ADD':
                thres[task] = ADD_thresholding(self.tb_log_dir + csv_name + '_eval.csv')
            else:
                print("optimal for the task {} is not supported yet".format(task))
        return thres

    def gen_score(self, stages=['GWAS_train', 'GWAS_valid', 'GWAS_test', 'OASIS'], thres={'ADD':0.5, 'NC':0.5, 'DE':1.5}):
        for stage in stages:
            data = pd.read_csv(self.csv_dir + stage + '.csv')[self.features + self.tasks + ['index']]
            data = self.drop_cases_without_label(data, 'COG')
            COG_data = self.preprocess_pipeline(data[self.features+['COG']], 'COG') # treat it as COG data to do the preprocessing
            features = COG_data.drop(['COG'], axis=1)
            labels = data[self.tasks]
            filenames = data['index']

            # make sure the features and labels has the same number of rows
            if len(features.index) != len(labels.index):
                raise ValueError('number of rows between features and labels have to be the same')

            predicts = []
            for i, task in enumerate(self.tasks):
                if task == 'COG':
                    predicts.append(self.models[i].predict(features))
                    print("the shape of prediction for COG task is ", predicts[-1].shape)

            content = []
            for i in range(len(features.index)):
                label = labels.iloc[i] # the feature and label are for the i th subject
                filename = filenames.iloc[i]
                case = {'filename': filename}
                for j, task in enumerate(self.tasks): # j is the task index
                    case[task] = "" if np.isnan(label[task]) else int(label[task])
                    if task == 'COG':
                        case[task+'_score'] = predicts[j][i]
                        if case[task+'_score'] < thres['NC']:
                            case[task + '_pred'] = 0
                        elif thres['NC'] <= case[task+'_score'] <= thres['DE']:
                            case[task + '_pred'] = 1
                        else:
                            case[task + '_pred'] = 2
                    elif task == 'ADD':
                        case[task + '_score'] = predicts[j][i, 1]
                        if case[task+'_score'] < thres['ADD']:
                            case[task + '_pred'] = 0
                        else:
                            case[task + '_pred'] = 1
                content.append(case)

            with open(self.tb_log_dir + stage + '_eval.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
                writer.writeheader()
                for case in content:
                    writer.writerow(case)


    def init_models(self, task_models):

        for name in task_models:
            # random forest model tested, function well
            if name == 'RandomForestCla':
                model = RandomForestClassifier()
            elif name == 'RandomForestReg':
                model = RandomForestRegressor()

            # xgboost model tested, function well
            elif name == 'XGBoostCla':
                model = xgb.XGBClassifier(use_label_encoder=False)
            elif name == 'XGBoostReg':
                model = xgb.XGBRegressor()

            # catboost model tested, function well
            elif name == 'CatBoostCla':
                model = CatBoostClassifier()
            elif name == 'CatBoostReg':
                model = CatBoostRegressor()

            # mlp model tested, function well
            elif name == 'PerceptronCla':
                model = MLPClassifier(max_iter=1000)
            elif name == 'PerceptronReg':
                model = MLPRegressor(max_iter=1000)

            # support vector model tested, function well
            elif name == 'SupportVectorCla':
                model = SVC(probability=True)
            elif name == 'SupportVectorReg':
                model = SVR()

            self.models.append(model)

    def init_imputer(self, data):
        """
        since cases with ADD labels is only a subset of the cases with COG label
        in this function, we will initialize a single imputer
        and fit the imputer based on the COG cases from the training part
        """
        imputation_method = self.config['impute_method']
        if imputation_method == 'mean':
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif imputation_method == 'median':
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
        elif imputation_method == 'most_frequent':
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        elif imputation_method == 'constant':
            imp = SimpleImputer(missing_values=np.nan, strategy='constant')
        elif imputation_method == 'KNN':
            imp = KNNImputer(n_neighbors=20)
        elif imputation_method == 'Multivariate':
            imp = IterativeImputer(max_iter=1000)
        else:
            raise NameError('method for imputation not supported')
        imp.fit(data)
        return imp

    def load_preprocess_data(self):
        data_train = pd.read_csv(self.csv_dir + 'GWAS_train.csv')
        for task in self.tasks:
            self.train_data.append(data_train[self.features + [task]])
        for i, task in enumerate(self.tasks):
            self.train_data[i] = self.preprocess_pipeline(self.train_data[i], task)
            print('after preprocess pipeline, the data frame for the {} task is'.format(task))
            print(self.train_data[i])
            print('\n' * 2)

    def preprocess_pipeline(self, data, task):
        """
        Cathy, we need to remove cases with too much missing non-imaging features, please consider adding the step
        """
        # data contains features + task columns
        data = self.drop_cases_without_label(data, task)
        data = self.transform_categorical_variables(data)
        features = data.drop([task], axis=1) # drop the task columns to get all features
        features = self.imputation(features) # do imputation merely on features
        features = self.normalize(features)  # normalize features
        features[task] = data[task]          # adding the task column back
        return features                      # return the complete data

    def drop_cases_without_label(self, data, label):
        # data = data.dropna(axis=0, how='any', thresh=None, subset=[label], inplace=False)
        data = data.dropna(axis=0, how='any',  subset=[label], inplace=False)
        return data.reset_index(drop=True)

    def transform_categorical_variables(self, data):
        if 'gender' in data:
            return data.replace({'male': 0, 'female': 1})
        else:
            return data
        # return pd.get_dummies(data, columns=['gender'])

    def imputation(self, data):
        columns = data.columns
        if self.imputer == None:
            self.imputer = self.init_imputer(data)
        data = self.imputer.transform(data)
        return pd.DataFrame(data, columns=columns)

    def normalize(self, data):
        df_std = data.copy()
        for column in df_std.columns:
            if data[column].std(): # normalize only when std != 0
                df_std[column] = (data[column] - data[column].mean()) / data[column].std()
        return df_std

def crossValid_GWAS(main_config, task_config, tasks, shap_analysis=True):
    model_name = main_config['model_name']
    csv_dir = main_config['csv_dir']
    shap, data = [], []
    for i in range(5):
        main_config['csv_dir'] = csv_dir + 'cross{}/'.format(i)
        main_config['model_name'] = model_name + '_cross{}'.format(i)
        model = GWAS_Task_Wrapper(tasks=tasks,
                                     main_config=main_config,
                                     task_config=task_config,
                                     seed=1000)
        model.train()
        thres = model.get_optimal_thres()
        model.gen_score(['GWAS_test'], thres)
        model.gen_score(['GWAS_train'], thres)
    whole_eval_package(model_name, 'GWAS_test','tb_log/{}_cross0/test_perform'.format(model_name))
    


crossValid_GWAS(tasks = ['COG'],
           main_config = read_json('config/GWAS.json'),
           task_config = read_json('config/GWAS.json'))
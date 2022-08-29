from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import pickle
import os, sys
from collections import defaultdict

# make dict from sys.argv
argv_dict = defaultdict(str)
for k, v in (elem.split('=') for elem in sys.argv[1:] if '=' in elem):
    argv_dict[k] = v

print(argv_dict)
dynamic_wf_name_list = argv_dict['dynamic_wf_name'].split(',')
print("dynamic_wf_name_list: {}".format(dynamic_wf_name_list))
dynamic_wf_step = argv_dict['dynamic_wf_step']
dynamic_wf_target= argv_dict['dynamic_wf_target']
dynamic_wf_model_name_list = argv_dict['dynamic_wf_model_name'].split(',')
print("dynamic_wf_model_name_list: {}".format(dynamic_wf_model_name_list))

home = os.environ['project_home']

workflow_history_path = os.environ['workflow_history_path']
step = 'parameter'
parameter_file = os.path.join(workflow_history_path, step, 'parameter.csv') # should be

model_path = os.path.join(home, 'model')
data_path = os.path.join(home, 'data_in')
df = pd.read_csv(os.path.join(data_path, 'boston_contest.csv')) # should be

print(df.head())

# fill N/A
df.isna().sum()
df['ZN'] = df['ZN'].fillna(0)
df['NOX'] = df['NOX'].fillna(df.NOX.mean())

# Feature drop
df = df.drop(['CHAS'],axis=1)

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

steps = [
    ('scalar', MinMaxScaler()),
    ('poly', PolynomialFeatures(degree=1)),
    ('model', LinearRegression())
]
lr_pipe = Pipeline(steps)
lr_pipe.fit(X_train, Y_train)
Y_prediction = lr_pipe.predict(X_test)

rmse = (np.sqrt(mean_squared_error(Y_test, Y_prediction)))
r2 = r2_score(Y_test, Y_prediction)
print("\nRMSE is {}".format(rmse))
print("R2 Score is {}".format(r2))

parameter_row_data = list()

if rmse < 5.0:
    for model_name in dynamic_wf_model_name_list:
        parameter_row_data.append([dynamic_wf_name_list[0], dynamic_wf_step, dynamic_wf_target, model_name])
    parameter_df = pd.DataFrame(parameter_row_data,
                                columns = ['workflow', 'step', 'target', 'args'])

    parameter_df.to_csv(parameter_file,index= False)
    print("\n{}".format(parameter_df))
else:
    parameter_row_data.append([dynamic_wf_name_list[1]])
    parameter_df = pd.DataFrame(parameter_row_data,
                                columns = ['workflow'])

    parameter_df.to_csv(parameter_file,index= False)
    print("\n{}".format(parameter_df))
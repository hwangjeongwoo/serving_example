from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import pickle
import os, sys
import importlib

home = os.environ['project_home']
workflow_history_path = os.environ['workflow_history_path']
step = os.environ['step_type']
target_path = os.environ['target_path']
seq = os.environ.get('seq', '0')

model_path = os.path.join(home, 'model')
data_path = os.path.join(home, 'data_in')

train_file = os.path.join(workflow_history_path, step, target_path, seq, 'train.csv')
target = os.environ['target']
args = sys.argv
# Please refer to 'wf-ex03-select-best-model' > select 'Model Training' > select Args tab > 'Arg' Column
# Assign a value of 'Arg' Column
train_type = args[1]

if not os.path.exists(model_path):
    os.mkdir(model_path)

df = pd.read_csv(os.path.join(data_path, 'boston_contest.csv'))

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

# [train_type].py would be loaded via 'import_module()'
model_module = importlib.import_module(train_type)
model = model_module.get_model(X_train, Y_train)
Y_prediction = model.predict(X_test)

rmse = (np.sqrt(mean_squared_error(Y_test, Y_prediction)))
r2 = r2_score(Y_test, Y_prediction)
print("\nRMSE is {}".format(rmse))
print("R2 Score is {}".format(r2))

model_name = train_type + '.pkl'
with open(os.path.join(model_path, model_name), 'wb') as f:
    pickle.dump(model, f)
print("\nmodel saved, path :%s" % os.path.join(model_path, model_name))

train_col = ['target', 'path', 'score', 'args']
train_df = pd.DataFrame(data=[[target, os.path.join(model_path, model_name), r2, 1]], columns=train_col)
train_df.to_csv(train_file, index = False)
print("\n{}".format(train_df))

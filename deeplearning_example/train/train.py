from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
import tensorflow as tf
import os, sys

home = os.environ['project_home']
workflow_history_path = os.environ['workflow_history_path']
step = 'models'
target_path = os.environ['target_path']
seq = os.environ.get('seq', '0')

model_path = os.path.join(home, 'model')
data_path = os.path.join(home, 'data_in')

train_file = os.path.join(workflow_history_path, step, target_path, seq, 'train.csv')
target = os.environ['target']

def create_df (col_list = []) : 
    df = pd.DataFrame(columns = col_list)
    return df

if not os.path.exists(model_path):
    os.mkdir(model_path)

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv(os.path.join(data_path, 'housing.csv'), delim_whitespace=True, header=None)

print(df.head())

dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer='adam')

model.fit(X_train, Y_train, epochs=200, batch_size=10)

# test set에 대한 모델 평가
Y_prediction = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, Y_prediction)))
r2 = r2_score(Y_test, Y_prediction)
print("\nR2 Score is {}".format(r2))

model_name = 'boston-housing'
model.save(os.path.join(model_path, model_name), 'a')
print("model saved, path :%s" % os.path.join(model_path, model_name))

train_col = ['target', 'path', 'score', 'args']
train_df = pd.DataFrame(data=[[target, os.path.join(model_path, model_name), r2, 1]], columns=train_col)
print(train_df)

train_df.to_csv(train_file, index = False)


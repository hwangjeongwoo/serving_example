from tensorflow import keras
import numpy
import json
import os, sys, re

def handler(data):

    model_name = 'boston-housing'
    home_dir = os.environ['project_home']
    current_path = os.path.join(home_dir, 'model')
    print(current_path)

    event_body = data
    print(event_body)

    rows = 1
    values_list = [[float(v) for v in event_body.values()] for i in range(rows)]

    print(values_list)
    np_arr_X = numpy.array(values_list)

    reconstructed_model = keras.models.load_model(os.path.join(current_path, model_name))
    ret = reconstructed_model.predict(np_arr_X).flatten()

    return {
        'statusCode': 200,
        'body': ret.tolist()[0]
    }

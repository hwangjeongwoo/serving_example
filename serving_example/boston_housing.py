from tensorflow import keras
import numpy
import json
import os, sys, re

def handler(data):

    print(os.getcwd())
    os.system("ls /mm/project -al")

    model_name = 'boston-housing'
    os.chdir('/home/splunk/project/model')
    current_path = os.getcwd()

    event_body = data
    print(event_body)
    os.system("ls /home/splunk/project -al")
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

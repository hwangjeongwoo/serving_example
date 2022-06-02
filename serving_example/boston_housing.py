from tensorflow import keras
import numpy
import json
import os, sys, re

def handler(data):

    print("[start] : " + os.getcwd())
    os.system("ls /mm/project -al")

    try:
        root_dir = "/mm/project"
        for (root, dirs, files) in os.walk(root_dir):
            print("# root : " + root)
            if len(dirs) > 0:
                for dir_name in dirs:
                    print("dir: " + dir_name)

            if len(files) > 0:
                for file_name in files:
                    print("file: " + file_name)
    except PermissionError:
        pass
    model_name = 'boston-housing'
    os.chdir('/mm/project/model')
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

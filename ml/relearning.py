import numpy as np
import pandas as pd
from threading import Thread

from ml.model.train_new_model import train_new_model

import tensorflow as tf



def parse_config_properties():

    props = {}
    with open('ml/configuration.properties', 'r') as f:
        for line in f:
            line = line.rstrip() #removes trailing whitespace and '\n' chars

            if "=" not in line: continue #skips blanks and comments w/o =
            if line.startswith("#"): continue #skips comments which contain =

            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            props[k] = v

    return props



def change_property(property, value):

    lines = []
    with open('ml/configuration.properties', 'r') as f:
        for line in f:
            if not property in line:
                lines.append(line)
            else:
                lines.append(f'{property} = {value}\n')

    with open('ml/configuration.properties', 'w') as f:
        f.writelines(lines)



def read_model(model_filename):
    return tf.keras.models.load_model(f'ml/model/saved_models/{model_filename}')



def import_data():

    id_to_color_mapping = {
        0: 'Grey',
        1: 'Blue',
        2: 'Brown',
        3: 'Green',
        4: 'Orange',
        5: 'Red',
        6: 'Violet',
        7: 'Beige',
        8: 'Yellow'
    }
    color_to_id_mapping = {}
    for p in id_to_color_mapping:
        color_to_id_mapping[id_to_color_mapping[p]] = p

    df = pd.read_csv('ml/data/colors.csv')
    X = df.loc[:,['R','G','B']].values.reshape(-1, 3)
    y = df['Color'].values
    #input -> 
    #   X - > [[R,G,B], ...] -> list of list of values
    #   y - > [Color, ...] -> list of values

    #converting X to numpy array of numpy arrays
    X = np.array([np.array(xi) for xi in X])

    #reshaping y to desired format and being numpy array of numpy arrays
    new_y = []
    for c in y:
        new_c = np.zeros(len(color_to_id_mapping))
        new_c[color_to_id_mapping[c]] = 1
        new_y.append(new_c)
    
    y = np.asarray(new_y)

    return X, y



def check_current_accuracy(model_name, X, y):
    model = read_model(model_name)
    model_accuracy = model.evaluate(X, y)
    
    return model_accuracy[1]



def order_retraining():
    
    X, y = import_data()

    props = parse_config_properties()
    curr_model_name = props['name_of_current_model']
    curr_accuracy = check_current_accuracy(curr_model_name, X, y)
    new_accuracy, new_model_name = train_new_model(swap_to_new_model = False)

    if new_accuracy > curr_accuracy:
        change_property('name_of_current_model', new_model_name)



def relearn(r, g, b, color):

    df = pd.read_csv('ml/data/colors.csv')

    df.loc[len(df.index)] = [r,g,b,color]

    df.to_csv('ml/data/colors.csv', index = False)

    props = parse_config_properties()

    dp_threshold = int(props['new_dp_treshold_to_retrain'])

    if (len(df.index) + 1) % dp_threshold == 0:

        Thread(target = order_retraining).start()

    pass
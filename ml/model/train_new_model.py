#Additional prerequisites -> Tensorflow 2.2.0 and Keras 2.3.1
# TO BE RUN FROM MAIN PROJECT DIR

import time
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.model_selection import train_test_split



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
    pass #TODO



def compile_neural_network(input_dimentions, number_of_classes):

    model = Sequential()
    model.add(Dense(input_dimentions, activation='relu', use_bias = True))
    model.add(Dense(192, activation='relu', use_bias = True))
    model.add(Dense(96, activation='relu', use_bias = True))
    model.add(Dense(48, activation='relu', use_bias = True))
    model.add(Dense(24, activation='relu', use_bias = True))
    model.add(Dense(12, activation='relu', use_bias = True))
    model.add(Dense(number_of_classes, activation='softmax', use_bias =False))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def train_neural_network(X_train, y_train, X_test, y_test, model):

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    history = model.fit(X_train, y_train, use_multiprocessing=True, workers = 2, epochs = 1000, validation_data=(X_test, y_test), callbacks=[early_stop])
    return history, model



def import_data(color_to_id_mapping):
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



def split_data(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    return X_train, X_test, y_train, y_test



def main():

    config_properties = parse_config_properties()

    time_format = config_properties['time_format']

    #need to reshape y to [[0,0,0,0,0,0,0,0,0,1], ....]
    #mapping both ways to encode/decode color values
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

    X, y = import_data(color_to_id_mapping)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = compile_neural_network(X.shape[1], len(color_to_id_mapping))

    history, model = train_neural_network(X_train, y_train, X_test, y_test, model)
    model.summary()

    timestamp = time.strftime(time_format, time.localtime())
    model_filename = f'neural_network{timestamp}.pb'
    model.save(f'ml/model/saved_models/{model_filename}')

if __name__ == '__main__':
    main()

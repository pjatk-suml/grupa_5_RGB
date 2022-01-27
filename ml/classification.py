#Additional prerequisites -> Tensorflow 2.2.0 and Keras 2.3.1
# TO BE RUN FROM MAIN PROJECT DIR


import tensorflow as tf
import numpy as np



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



def read_model(model_filename):
    return tf.keras.models.load_model(f'ml/model/saved_models/{model_filename}')



def classify (r, g, b):
    
    current_model = parse_config_properties()['name_of_current_model']

    model = read_model(current_model)


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

    X = np.array([r, g, b])
    X = np.array([X])

    color = id_to_color_mapping[np.argmax(model.predict(X)[0])]

    return color
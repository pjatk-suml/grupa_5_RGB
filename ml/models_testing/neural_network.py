#Additional prerequisites -> Tensorflow 2.2.0 and Keras 2.3.1, matplotlib for visualization

import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

def compile_neural_network(input_dimentions, number_of_classes):

    model = Sequential()
    model.add(Dense(input_dimentions, activation='relu', use_bias = True))
    model.add(Dense(6, activation='relu', use_bias = True))
    model.add(Dense(number_of_classes, activation='softmax', use_bias =False))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def train_neural_network(X_train, y_train, X_test, y_test, model):

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X_train, y_train, batch_size=32, epochs = 1000, validation_data=(X_test, y_test), callbacks=[early_stop])
    return history, model



def perform_test(X, y):

    #input -> 
    #   X - > [[R,G,B], ...] -> list of list of values
    #   y - > [Color, ...] -> list of values

    #converting X to numpy array of numpy arrays
    X = np.array([np.array(xi) for xi in X])

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

    #reshaping y to desired format and being numpy array of numpy arrays
    new_y = []
    for c in y:
        new_c = np.zeros(len(color_to_id_mapping))
        new_c[color_to_id_mapping[c]] = 1
        new_y.append(new_c)
    
    y = np.asarray(new_y)

    
    #splitting to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    #data -> 
    #   X_train - > [[R,G,B], ...] -> list of list of values | train set
    #   Y_train - > [[0,0,0,0,0,0,0,0,0,1], ....]-> list of list of 0 and 1 -> index of value '1' representing a chosen color
    #   X_test - > [[R,G,B], ...] -> list of list of values | test set
    #   Y_test - > [[0,0,0,0,0,0,0,0,0,1], ....]-> list of list of 0 and 1 -> index of value '1' representing a chosen color

    print(f'X shape -> {X.shape}\ny shape -> {y.shape}')

    model = compile_neural_network(X.shape[1], len(color_to_id_mapping))

    history, model = train_neural_network(X_train, y_train, X_test, y_test, model)
    model.summary()

    #visualisation of learning 
    plt.plot(history.history['accuracy'], label='train set')
    plt.plot(history.history['val_accuracy'], label='test set')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


    #testing myself for accuracy in the whole dataset

    good_predictions = 0.0
    y_predicted = model.predict(X)

    for p, t in zip(y_predicted, y):
        if np.argmax(p) == np.argmax(t):
            good_predictions += 1.0

    accuracy = good_predictions / y.shape[0]

    return accuracy
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

def classify (r, g, b):
    #print('classification in process....')

    df = pd.read_csv('ml/data/colors.csv')

    X = df.loc[:,['R','G','B']].values.reshape(-1,3)
    #print (X)

    y = df['Color'].values
    #print (y)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X,y)

    return classifier.predict([[r,g,b]])[0]
    
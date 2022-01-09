import numpy as np
import pandas as pd

def relearn(r, g, b, color):
    #print('relearning in process....')

    df = pd.read_csv('ml/data/colors.csv')

    df.loc[len(df.index)] = [r,g,b,color]

    df.to_csv('ml/data/colors.csv', index = False)

    pass
from k_nearest_neighbours import perform_test as KNN_perform_test

from sklearn.model_selection import train_test_split
import pandas as pd



def main():

    #data from .csv import
    df = pd.read_csv('data/colors.csv')
    X = df.loc[:,['R','G','B']].values.reshape(-1, 3)
    y = df['Color'].values
    
    
    #testing KNN
    print('\n========= Testing KNN ============\n')

    min_n = 3
    max_n = 15
    iter_per_n = 100
    print(f'''Testing specs:\n
    Minimal number of neighbours = {min_n}\n
    Maximal number of neighbours = {max_n}\n
    Number of test per each number of neughbours = {iter_per_n}\n\n''')

    KNN_result = KNN_perform_test(X, y, min_n, max_n, iter_per_n)
    best_n = min_n

    for key in KNN_result:
        if KNN_result[key] > KNN_result[best_n]:
            best_n = key
        print(f'Average result per {key} neighbours = {KNN_result[key]}')

    print(f'\n\nBest result: {best_n} neighbours with average accuracy = {KNN_result[best_n]}')
    
    print('\n==================================\n\n')



if __name__ == '__main__':
    main()
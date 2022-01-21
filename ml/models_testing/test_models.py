from k_nearest_neighbours import perform_test as KNN_perform_test
from naive_bayes import perform_test as NB_perform_test

from sklearn.model_selection import train_test_split
import pandas as pd

# TO Run this script, run it from 'project_dir/ml/model_testing'
def main():

    #data from .csv import
    df = pd.read_csv('data/colors.csv')
    X = df.loc[:,['R','G','B']].values.reshape(-1, 3)
    y = df['Color'].values
    
    
    #testing KNN
    print('\n========= Testing KNN ============\n')

    print('Set minimal number of neighbors: [int]')
    min_n = int(input())
    print('Set maximal number of neighbors: [int]')
    max_n = int(input())
    print('Set number of tests per each number of neughbours - each test is new data set split: [int]')
    iter_per_n = int(input())
    print(f'''\nTesting specs:\n
    Minimal number of neighbours = {min_n}\n
    Maximal number of neighbours = {max_n}\n
    Number of tests per each number of neughbours = {iter_per_n}\n\n''')

    KNN_result = KNN_perform_test(X, y, min_n, max_n, iter_per_n)
    best_n = min_n

    for key in KNN_result:
        if KNN_result[key] > KNN_result[best_n]:
            best_n = key
        print(f'Average result per {key} neighbours = {KNN_result[key]}')

    print(f'\n\nBest result: {best_n} neighbours with average accuracy = {KNN_result[best_n]}')
    
    print('\n==================================\n\n')
    

    #testing Naive Bayes
    print('\n========= Testing Naive Bayes ============\n')

    print('Set number of tests - each test is new data set split: [int]')
    iter= int(input())
    print(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')

    NB_average_result, NB_results = NB_perform_test(X, y, iter)

    print(f'''List of results (accuracies):\n''')
    for i, res in enumerate(NB_results):
        print(f'Data split {i+1} -> accuracy = {res}')

    print(f'''\nAverage accuracy = {NB_average_result}''')
    

    print('\n==================================\n\n')


if __name__ == '__main__':
    main()
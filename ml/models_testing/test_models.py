from k_nearest_neighbours import perform_test as KNN_perform_test
from naive_bayes import perform_test as NB_perform_test
from support_vector_machine import perform_test as SVM_perform_test
from logistic_regression import perform_test as LR_perform_test
from random_forest import perform_test as RF_perform_test

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



    #testing Naive Bayes
    print('\n========= Testing Support Vector Machine ============\n')

    print('Set number of tests - each test is new data set split: [int]')
    iter= int(input())
    print(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')

    SVM_average_result, SVM_results = SVM_perform_test(X, y, iter)

    print(f'''List of results (accuracies):\n''')
    for i, res in enumerate(SVM_results):
        print(f'Data split {i+1} -> accuracy = {res}')

    print(f'''\nAverage accuracy = {SVM_average_result}''')

    print('\n==================================\n\n')



    #testing Logistic Regression
    print('\n========= Testing Logistic Regression ============\n')

    print('Set number of tests - each test is new data set split: [int]')
    iter= int(input())
    print(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')

    LR_average_result, LR_results = LR_perform_test(X, y, iter)

    print(f'''List of results (accuracies):\n''')
    for i, res in enumerate(LR_results):
        print(f'Data split {i+1} -> accuracy = {res}')

    print(f'''\nAverage accuracy = {LR_average_result}''')

    print('\n==================================\n\n')



    #testing Random Forest
    print('\n========= Testing Random Forest ============\n')

    print('Set number of tests - each test is new data set split: [int]')
    iter= int(input())
    print(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')

    RF_average_result, RF_results = LR_perform_test(X, y, iter)

    print(f'''List of results (accuracies):\n''')
    for i, res in enumerate(RF_results):
        print(f'Data split {i+1} -> accuracy = {res}')

    print(f'''\nAverage accuracy = {RF_average_result}''')

    print('\n==================================\n\n')


if __name__ == '__main__':
    main()
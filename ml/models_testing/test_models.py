from k_nearest_neighbours import perform_test as KNN_perform_test
from naive_bayes import perform_test as NB_perform_test
from support_vector_machine import perform_test as SVM_perform_test
from logistic_regression import perform_test as LR_perform_test
from random_forest import perform_test as RF_perform_test
from neural_network import perform_test as NN_perform_test


import pandas as pd

import time

# TO Run this script, run it from 'project_dir/ml/model_testing'
def main():

    #data from .csv import
    df = pd.read_csv('data/colors.csv')
    X = df.loc[:,['R','G','B']].values.reshape(-1, 3)
    y = df['Color'].values
    
    #create a new log file and prepare to log values
    time_format_filename = '__%Y_%m_%d__%H_%M_%S'
    time_format_raport_log = '%Y/%m/%d %H:%M:%S'

    creation_timestamp = time.strftime(time_format_filename, time.localtime())
    raport_filename = f'raports/Raport{creation_timestamp}.txt'

    raport = open(raport_filename, 'a')
    

    #testing KNN
    print('\n========= Testing KNN ============\n')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'\n========= {timestamp} Testing KNN ============\n')

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
    raport.write(f'''\nTesting specs:\n
    Minimal number of neighbours = {min_n}\n
    Maximal number of neighbours = {max_n}\n
    Number of tests per each number of neughbours = {iter_per_n}\n\n''')

    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'{timestamp} Computing started')
    timestamp_computing_started = time.time()*1000.0 

    KNN_result = KNN_perform_test(X, y, min_n, max_n, iter_per_n)
    best_n = min_n

    for key in KNN_result:
        if KNN_result[key] > KNN_result[best_n]:
            best_n = key
        print(f'Average result per {key} neighbours = {KNN_result[key]}')

    print(f'\n\nBest result: {best_n} neighbours with average accuracy = {KNN_result[best_n]}')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    time_elapsed = time.time()*1000 - timestamp_computing_started
    raport.write(f'\n\nFinish time: {timestamp}\nTime elapsed: {time_elapsed/1000.0}s\nBest result: {best_n} neighbours with average accuracy = {KNN_result[best_n]}')

    print('\n==================================\n\n')
    raport.write('\n==================================\n\n')
    


    #testing Naive Bayes
    print('\n========= Testing Naive Bayes ============\n')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'\n========= {timestamp} Testing Naive Bayes ============\n')

    print('Set number of tests - each test is new data set split: [int]')
    iter= int(input())

    print(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')
    raport.write(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')

    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'{timestamp} Computing started')
    timestamp_computing_started = time.time()*1000.0 

    NB_average_result, NB_results = NB_perform_test(X, y, iter)

    print(f'''List of results (accuracies):\n''')
    for i, res in enumerate(NB_results):
        print(f'Data split {i+1} -> accuracy = {res}')

    print(f'''\nAverage accuracy = {NB_average_result}''')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    time_elapsed = time.time()*1000 - timestamp_computing_started
    raport.write(f'\n\nFinish time: {timestamp}\nTime elapsed: {time_elapsed/1000.0}s\nAverage accuracy = {NB_average_result}')
    

    print('\n==================================\n\n')
    raport.write('\n==================================\n\n')



    #testing Naive Bayes
    print('\n========= Testing Support Vector Machine ============\n')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'\n========= {timestamp} Testing Support Vector Machine ============\n')

    print('Set number of tests - each test is new data set split: [int]')
    iter= int(input())

    print(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')
    raport.write(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')
    
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'{timestamp} Computing started')
    timestamp_computing_started = time.time()*1000.0 

    SVM_average_result, SVM_results = SVM_perform_test(X, y, iter)

    print(f'''List of results (accuracies):\n''')
    for i, res in enumerate(SVM_results):
        print(f'Data split {i+1} -> accuracy = {res}')

    print(f'''\nAverage accuracy = {SVM_average_result}''')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    time_elapsed = time.time()*1000 - timestamp_computing_started
    raport.write(f'\n\nFinish time: {timestamp}\nTime elapsed: {time_elapsed/1000.0}s\nAverage accuracy = {SVM_average_result}')

    print('\n==================================\n\n')
    raport.write('\n==================================\n\n')



    #testing Logistic Regression
    print('\n========= Testing Logistic Regression ============\n')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'\n========= {timestamp} Testing Logistic Regression ============\n')

    print('Set number of tests - each test is new data set split: [int]')
    iter= int(input())

    print(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')
    raport.write(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')
    
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'{timestamp} Computing started')
    timestamp_computing_started = time.time()*1000.0 

    LR_average_result, LR_results = LR_perform_test(X, y, iter)

    print(f'''List of results (accuracies):\n''')
    for i, res in enumerate(LR_results):
        print(f'Data split {i+1} -> accuracy = {res}')

    print(f'''\nAverage accuracy = {LR_average_result}''')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    time_elapsed = time.time()*1000 - timestamp_computing_started
    raport.write(f'\n\nFinish time: {timestamp}\nTime elapsed: {time_elapsed/1000.0}s\nAverage accuracy = {LR_average_result}')

    print('\n==================================\n\n')
    raport.write('\n==================================\n\n')



    #testing Random Forest
    print('\n========= Testing Random Forest ============\n')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'\n========= {timestamp} Testing Random Forest ============\n')

    print('Set number of tests - each test is new data set split: [int]')
    iter= int(input())

    print(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')
    raport.write(f'''\nTesting specs:\n
    Number of tests = {iter}\n\n''')
    
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'{timestamp} Computing started')
    timestamp_computing_started = time.time()*1000.0 

    RF_average_result, RF_results = RF_perform_test(X, y, iter)

    print(f'''List of results (accuracies):\n''')
    for i, res in enumerate(RF_results):
        print(f'Data split {i+1} -> accuracy = {res}')

    print(f'''\nAverage accuracy = {RF_average_result}''')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    time_elapsed = time.time()*1000 - timestamp_computing_started
    raport.write(f'\n\nFinish time: {timestamp}\nTime elapsed: {time_elapsed/1000.0}s\nAverage accuracy = {RF_average_result}')

    print('\n==================================\n\n')
    raport.write('\n==================================\n\n')



    #testing Neural Network
    print('\n========= Testing Neural Network ============\n')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'\n========= {timestamp} Testing Neural Network ============\n')

    print('Do you want to see charts after fitting model?')
    print('[y/n]')
    key = input()[0]
    if key == 'y':
        charts = True
    elif key == 'n':
        charts = False
    else:
        print('Wrong input - will not be showing charts')
        chart = False

    print(f'''\nTesting specs:\n
    Show charts = {charts}\n\n''')
    raport.write(f'''\nTesting specs:\n
    Show charts = {charts}\n\n''')
    
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'{timestamp} Computing started')
    timestamp_computing_started = time.time()*1000.0 


    NN_average_result = NN_perform_test(X, y, charts)

    print(f'''\nAverage accuracy = {NN_average_result}''')
    timestamp = time.strftime(time_format_raport_log, time.localtime())
    time_elapsed = time.time()*1000 - timestamp_computing_started
    raport.write(f'\n\nFinish time: {timestamp}\nTime elapsed: {time_elapsed/1000.0}s\nAverage accuracy = {NN_average_result}')

    print('\n==================================\n\n')
    raport.write('\n==================================\n\n')



    timestamp = time.strftime(time_format_raport_log, time.localtime())
    raport.write(f'\n\n{timestamp} END OF SESSION')

    raport.close()


if __name__ == '__main__':
    main()
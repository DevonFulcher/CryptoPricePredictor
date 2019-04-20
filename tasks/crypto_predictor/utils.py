import csv
import numpy as np
import math


def data_generator():
    #this is a constant for now
    #the number of observations before prediction
    length_of_convolution = 50
    #the portion of the total data set that is dedicated to training
    portion_training_set = .8

    with open('bitcoin prices 4-3-18 to 4-3-19.csv', newline='') as csvfile:
        data = csv.reader(csvfile)
        prices = []
        for datum in data:
            prices.append(datum[1])
        train_set_size = math.floor(portion_training_set * len(prices))
        training_set = prices[:train_set_size]
        test_set = prices[train_set_size:]
        (x_train, y_train) = generate_x_and_y(training_set, length_of_convolution)
        (x_test, y_test) = generate_x_and_y(test_set, length_of_convolution)

    return (x_train, y_train), (x_test, y_test)


def generate_x_and_y(observations_set, length_of_convolution):
    x = np.array([])
    y = np.array([])
    for i in range(len(observations_set) - length_of_convolution - 1):
        j = 0
        for _ in range(length_of_convolution):
            x = np.append(x, observations_set[i + j])
            j += 1
        y = np.append(y, observations_set[i + j + 1])
    return x, y

if __name__ == '__main__':
    print(data_generator())

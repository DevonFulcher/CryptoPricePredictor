import csv
import numpy as np
import math


def data_generator(inputFile):
    #this is a constant for now
    #the number of observations before prediction
    length_of_convolution = 3
    #the portion of the total data set that is dedicated to training
    portion_training_set = .8
    #data_generator to take general csvInput
    with open(inputFile, newline='') as csvfile:
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
    assert len(observations_set) >= length_of_convolution, \
        "len(observations_set): " + str(len(observations_set)) + " length_of_convolution: " + str(length_of_convolution)
    num_training_examples = len(observations_set) - length_of_convolution
    x = np.empty([num_training_examples, length_of_convolution, 1])
    y = np.empty([num_training_examples, 1, 1])
    for i in range(len(x)):
        j = 0
        for _ in range(len(x[0])):
            x[i][j][0] = observations_set[i + j]
            j += 1
        y[i] = observations_set[i + j]
    return x, y


if __name__ == '__main__':
    print(data_generator())

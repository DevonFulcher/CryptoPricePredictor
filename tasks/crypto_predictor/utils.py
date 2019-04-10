import csv
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
'''
ops.reset_default_graph()

sess = tf.Session()
list_tensor = []

with open('bitcoin prices 4-3-18 to 4-3-19.csv', newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        print(row[0], row[1])
        list_tensor.append(row)

something = tf.constant(list_tensor)
print(something)

# for data in list_tensor:
#print((data[0],data[1]))'''

def data_generator():
    #this is a constant for now
    #the number of observations before prediction
    length_of_convolution = 50
    #the portion of the total data set that is dedicated to training
    portion_training_set = .8

    with open('bitcoin prices 4-3-18 to 4-3-19.csv', newline='') as csvfile:
        data = csv.reader(csvfile)
        training_set = data[:portion_training_set * len(data)]
        test_set = data[len(data) - portion_training_set * len(data):]
        (x_train, y_train) = generate_x_and_y(training_set, length_of_convolution, portion_training_set)
        (x_test, y_test) = generate_x_and_y(test_set, length_of_convolution, portion_training_set)

    return (x_train, y_train), (x_test, y_test)


def generate_x_and_y(observations_set, length_of_convolution, portion_training_set):
    x = np.array([])
    y = np.array([])
    for j in range(len(observations_set) - length_of_convolution - 1):
        i = 0
        for _ in range(length_of_convolution):
            x.append(observations_set[j + i][1])
            i += 1
        y.append(observations_set[j + i + 1][1])
    return x, y

if __name__ == '__main__':
    print(data_generator())

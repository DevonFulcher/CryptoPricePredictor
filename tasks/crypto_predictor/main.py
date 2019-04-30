from utils import data_generator
from tcn import compiled_tcn
import matplotlib.pyplot as plt
import random
import argparse #Use to parse args

parser = argparse.ArgumentParser(description="Load in any csv file")
parser.add_argument("CSVFile",help="a file to parse")
args = parser.parse_args()

def run_task(length_of_convolution = 3,
    kernel_size=3,  # type: int
    dilations=[2 ** i for i in range(9)],  # type: List[int]
    nb_stacks=1,  # type: int
    use_skip_connections=True,  # type: bool
    return_sequences=True, #uncertain about this parameter
    dropout_rate=0.05,  # type: float
    epochs = 100,
    name="run",
    create_plot = False):

    #the portion of the total data set that is dedicated to training
    portion_training_set = .8

    (x_train, y_train), (x_test, y_test) = data_generator(args.CSVFile, length_of_convolution, portion_training_set)

    model = compiled_tcn(num_feat=1,  # type: int
                         num_classes=1,  # type: int
                         nb_filters=20,  # type: int
                         kernel_size=kernel_size,  # type: int
                         dilations=dilations,  # type: List[int]
                         nb_stacks=nb_stacks,  # type: int
                         max_len=None,  # type: int
                         padding='causal',  # type: str
                         use_skip_connections=use_skip_connections,  # type: bool
                         return_sequences=return_sequences, #uncertain about this parameter
                         regression=True,  # type: bool
                         dropout_rate=dropout_rate,  # type: float
                         name=name  # type: str
                         )

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    loss = history.history['loss']

    # Plot training & validation loss values
    if create_plot:
        plt.plot(loss)
        #plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    #output parameters file
    average_loss = str(sum(loss) / len(loss))
    file = open(average_loss + "-" + name + ".csv", "w+")
    file.write("kernel_size, " + str(kernel_size) + "\n" +
               "dilations, " + str(dilations) + "\n" +
               "nb_stacks, " + str(nb_stacks) + "\n" +
               "use_skip_connections, " + str(use_skip_connections) + "\n" +
               "return_sequences, " + str(return_sequences) + "\n" + #uncertain about this parameter
               "dropout_rate, " + str(dropout_rate) + "\n" +
               "epochs, " + str(epochs) + "\n" +
               "name, " + str(name) + "\n" +
               "input file, " + str(args.CSVFile) + "\n" +
               "average loss, " + average_loss + "\n" +
               "loss, " + str(loss) + "\n")


def run_task_indefinite_random():
    #the number of observations before prediction
    range_length_of_convolution = [10, 100]
    range_kernel_size=[2, 10]  # type: int
    range_dilations=[3, 9]  # type: List[int]
    range_nb_stacks=[0, 5] # type: int
    range_dropout_rate=[0.01, .1]  # type: float
    range_epochs = [100, 1000]
    name_num = 0
    namer = "anotherOne"
    while(True):
        name=namer + str(name_num)
        try:
            run_task(length_of_convolution=random.randint(range_length_of_convolution[0], range_length_of_convolution[1]),
                     kernel_size=random.randint(range_kernel_size[0],range_kernel_size[1]),
                     dilations=[2 ** i for i in range(random.randint(range_dilations[0], range_dilations[1]))],
                     nb_stacks=random.randint(range_nb_stacks[0], range_nb_stacks[1]),
                     use_skip_connections=random.choice((True, False)),
                     dropout_rate=random.uniform(range_dropout_rate[0], range_dropout_rate[1]),
                     epochs=random.randint(range_epochs[0], range_epochs[1]),
                     name=name)
        except AssertionError:
            continue
        name_num += 1

'''
def run_task_indefinite_grid:
    #the number of observations before prediction
    range_length_of_convolution = [10, 100]
    range_kernel_size=[4, 10]  # type: int
    range_dilations=[3, 9]  # type: List[int]
    range_nb_stacks=[1, 5] # type: int
    range_dropout_rate=[0.01, .1]  # type: float
    range_epochs = [100, 1000]
    name_num = 0
    namer = "random"
    while(True):
        name=namer + str(name_num)
        try:
            for i in range(range_length_of_convolution[0], range_length_of_convolution[1], 10):
                for j in range(range_kernel_size[0], range_kernel_size[1], 1):
                    for k in range(range_dilations[0], range_dilations[1], 1):
                        for l in range(range_nb_stacks)'''


if __name__ == '__main__':
    run_task_indefinite_random()

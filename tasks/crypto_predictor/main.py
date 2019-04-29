from utils import data_generator
from tcn import compiled_tcn
import argparse #Use to parse args

parser = argparse.ArgumentParser(description="Load in any csv file")
parser.add_argument("CSVFile",help="a file to parse")
parser.add_argument("name", help="name of this run")
args = parser.parse_args()

def run_task():
    #adjustable parameters
    #the number of observations before prediction
    length_of_convolution = 3
    #the portion of the total data set that is dedicated to training
    portion_training_set = .8
    kernel_size=2  # type: int
    dilations=[2 ** i for i in range(9)]  # type: List[int]
    nb_stacks=1  # type: int
    max_len=None  # type: int
    use_skip_connections=True  # type: bool
    return_sequences=True #uncertain about this parameter
    dropout_rate=0.05  # type: float
    name=args.name

    (x_train, y_train), (x_test, y_test) = data_generator(args.CSVFile, length_of_convolution, portion_training_set)

    model = compiled_tcn(num_feat=1,  # type: int
                         num_classes=1,  # type: int
                         nb_filters=20,  # type: int
                         kernel_size=kernel_size,  # type: int
                         dilations=dilations,  # type: List[int]
                         nb_stacks=nb_stacks,  # type: int
                         max_len=max_len,  # type: int
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

    model.fit(x_train, y_train, epochs=100)
    model.evaluate(x=x_test, y=y_test, batch_size=None)

    file = open(name + ".txt", "w+")
    file.write("kernel_size: " + str(kernel_size) + "\n" +
               "dilations: " + str(dilations) + "\n" +
               "nb_stacks: " + str(nb_stacks) + "\n" +
               "max_len: " + str(max_len) + "\n" +
               "use_skip_connections: " + str(use_skip_connections) + "\n" +
               "return_sequences: " + str(return_sequences) + "\n" + #uncertain about this parameter
               "dropout_rate: " + str(dropout_rate) + "\n" +
               "name: " + str(name) + "\n")

    print("\nBe sure to type the loss into ", str(name + ".txt"), " for documentation!")

if __name__ == '__main__':
    run_task()

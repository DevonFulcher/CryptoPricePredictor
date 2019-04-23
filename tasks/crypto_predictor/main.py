from utils import data_generator
from tcn import compiled_tcn
import argparse #Use to parse args

parser = argparse.ArgumentParser(description = "Load in any csv file")
parser.add_argument("CSVFile",help="a file to parse")
args = parser.parse_args()

def run_task():
    (x_train, y_train), (x_test, y_test) = data_generator(args.CSVFile)
    batch_size, timesteps, input_dim = None, -1, 1

    #i = Input(batch_shape=(batch_size, timesteps, input_dim))
    model = compiled_tcn(num_feat=1,  # type: int
                         num_classes=1,  # type: int
                         nb_filters=20,  # type: int
                         kernel_size=2,  # type: int
                         dilations=[2 ** i for i in range(9)],  # type: List[int]
                         nb_stacks=1,  # type: int
                         max_len=None,  # type: int
                         padding='causal',  # type: str
                         use_skip_connections=True,  # type: bool
                         return_sequences=True, #uncertain about this parameter
                         regression=True,  # type: bool
                         dropout_rate=0.05,  # type: float
                         name='tcn'  # type: str
                         )

    # model = compiled_tcn(return_sequences=False,
    #                      num_feat=1,
    #                      num_classes=10,
    #                      nb_filters=20,
    #                      kernel_size=6,
    #                      dilations=[2 ** i for i in range(9)],
    #                      nb_stacks=1,
    #                      max_len=x_train[0:1].shape[1],
    #                      use_skip_connections=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()

    print("y_train: ", y_train)
    print("y_train.shape: ", y_train.shape)
    print("x_train: ", x_train)
    print("x_train.shape: ", x_train.shape)

    model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))
    # model.fit(x_train, y_train.squeeze().argmax(axis=1), epochs=100,
    #           validation_data=(x_test, y_test.squeeze().argmax(axis=1)))


if __name__ == '__main__':
    run_task()

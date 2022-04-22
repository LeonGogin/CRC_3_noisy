# import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_data(arg, test_size = 0.2, base_path = "data/"):
    """
    load data from binary file .npy
    path starting point: data/

    in the case of "train"
        data will be splited in train and validation sub sets

        and return in tuple --> (train_x, valid_x)
        lables are in one-hot encoding --> (train_y, valid_y)

    in the case of "test"
        lables are returned as tuple of one hot and original --> (y, one_hot_y)

    arg: "test" for test data set
        "train" for training data set
    """
    if arg not in ["test", "train"]:
        raise  ValueError("argument should be neither: test or train")

    base_path = "data/"
    x = np.load(base_path + f"X_{arg}.npy" )
    y = np.load(base_path + f"Y_{arg}.npy" )
    one_hot_y = to_categorical(y)
    if arg == "train":
        # Train data set must be spited
        train_x, valid_x, train_y, valid_y = train_test_split(x, one_hot_y,
                                                        test_size = test_size,
                                                        random_state = 13)
        x = (train_x, valid_x)
        y = (train_y, valid_y)
    else:
        y = (y, one_hot_y)

    real = np.load(base_path + f"real_classes_{arg}.npy" )
    return x, y, real


def real_class_dictionary(real):
    """
    compute a dictionary that make correspondance
    between real lables and indexes in dataset
        keys are lables in plane text
        value are array of index corresponding to lable
    """
    keys = np.unique(real)
    d = {}
    for k in keys:
        d[k] = np.where(real == k)[0]
    return d





































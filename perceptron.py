import argparse
import numpy as np
import pandas as pd
import time


class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None  # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        # TODO implement this
        n_rows = xFeat.shape[0]
        d_columns = xFeat.shape[1]
        self.w = np.random.randn(d_columns)

        for epoch in range(self.mEpoch):
            mistakes = 0
            for x in range(n_rows):
                x_vector = xFeat[x]
                activation_value = np.dot(x_vector, self.w)

                # assume class is negative
                y_hat = 0
                # if activation value is positive we change class to (+)
                if activation_value >= 0:
                    y_hat = 1
                # check for mistake
                if y_hat != y[x]:
                    mistakes += 1
                    # mistake on negative
                    if activation_value >= 0:
                        self.w = np.subtract(self.w, x_vector)
                    # mistake on positive
                    else:
                        self.w = np.add(self.w, x_vector)

            if epoch % 100 == 0:
                print('training epoch ', epoch, '...')

            stats[epoch] = {'mistakes': mistakes}

        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        activation_values = np.matmul(xFeat, self.w)
        yHat = []

        for value in activation_values:
            if value >= 0:
                yHat.append(1)
            else:
                yHat.append(0)

        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    mistakes = 0

    for x in range(len(yHat)):
        if yHat[x] != yTrue[x]:
            mistakes += 1

    return mistakes


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def file_to_pandas(filename):
    df = pd.read_csv(filename)
    return df


def get_weights(self):
    copy = self.w

    return copy


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    mistake_count = calc_mistakes(yHat, yTest)
    print(mistake_count)
    print("Accuracy on the test dataset ")
    print(round(mistake_count / yTest.shape[0], 2))

    weights = model.w
    weights_copy = weights.copy()
    most_positive_index = []
    most_negative_index = []
    for x in range(15):
        index = weights_copy.argmax()
        most_positive_index.append(index)
        weights_copy = np.delete(weights_copy, index)

    for y in range(15):
        index = weights_copy.argmin()
        most_negative_index.append(index)
        weights_copy = np.delete(weights_copy, index)




if __name__ == "__main__":
    main()

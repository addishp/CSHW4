import argparse
import pandas as pd
import numpy as np
import requests
import collections
import random
import csv
from sklearn import preprocessing


def model_assessment(filename):
    """
    Given the entire data, decide how
    you want to assess your different models
    to compare perceptron, logistic regression,
    and naive bayes, the different parameters, 
    and the different datasets.
    """
    data = open(filename).read()
    # I separate my data by the indent between the emails ('\n')
    data_text = data.split(sep='\n')
    # the last element is an empty space
    data_text.pop()

    # I separate my data by the indent between the emails ('\n')
    email_splits = data.split(sep='\n')
    # the last element is an empty space so we pop it
    email_splits.pop()

    # Let's shuffle the list and then process it
    # It is easier to handle shuffle first than split data
    random.shuffle(email_splits)

    # At the end of an email, python adds a \n
    # We should remove this
    email_word_lists = []
    y_labels = []
    for email in email_splits:
        word_list = email.split()
        # first value in email_splits is the label
        y_labels.append(int(word_list.pop(0)))
        # now we can pop the words for
        email_word_lists.append(word_list)

    endpoint_train = int(0.8 * len(email_word_lists))
    endpoint_test = len(email_word_lists)
    xTrain_set = email_word_lists[0:endpoint_train]
    yTrain_set = y_labels[0:endpoint_train]
    xTest_set = email_word_lists[endpoint_train:endpoint_test]
    yTest_set = y_labels[endpoint_train:endpoint_test]

    return xTrain_set, yTrain_set, xTest_set, yTest_set


def build_vocab_map(xTrain, yTrain, xTest, yTest):
    # I will build_vocab_map for xTrain and xTest
    # I will only look at xTrain for finding words that appear in at least 30 emails
    vocab_map = collections.Counter()
    for email in xTrain:
        # I will use a dictionary counter for each email
        # this will give me unique keys as well as count for future implementation
        this_email_unique_words = collections.Counter(email).keys()
        vocab_map.update(this_email_unique_words)

    # from the vocab_map I check for words
    # with a value(count) of greater than 30 words
    vocab_map_items = vocab_map.items()
    words_in_30_plus_emails = []
    for key in vocab_map:
        if vocab_map.get(key) >= 30:
            words_in_30_plus_emails.append(key)

    return words_in_30_plus_emails


def each_email_vocab_map(xTrain, xTest):
    xTrain_vocab_maps = []
    for email in xTrain:
        this_email_unique_words = collections.Counter(email)
        xTrain_vocab_maps.append(this_email_unique_words)

    xTest_vocab_maps = []
    for email in xTest:
        this_email_unique_words = collections.Counter(email)
        xTest_vocab_maps.append(this_email_unique_words)

    return xTrain_vocab_maps, xTest_vocab_maps


def construct_binary(xTrain_vocab_maps, xTest_vocab_maps, words_in_30_plus_emails):
    """
    Construct the email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """

    # Let's create the vectors now that we
    # can check for the word in the dict. of each email
    binary_xTrain_vectors = []
    for map in xTrain_vocab_maps:
        vector = []
        for word in words_in_30_plus_emails:
            if word in map.keys():
                vector.append(1)
            else:
                vector.append(0)
        binary_xTrain_vectors.append(vector)

    binary_xTest_vectors = []
    for map in xTest_vocab_maps:
        vector = []
        for word in words_in_30_plus_emails:
            if word in map.keys():
                vector.append(1)
            else:
                vector.append(0)
        binary_xTest_vectors.append(vector)

    # make the table you will export

    return binary_xTrain_vectors, binary_xTest_vectors


def construct_count(xTrain_vocab_maps, xTest_vocab_maps, scaled_word_list):
    """
    Construct the email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """

    # Let's create the vectors now that we
    # can check for the word in the dict. of each email
    count_xTrain_vectors = []
    for map in xTrain_vocab_maps:
        vector = []
        for word in scaled_word_list:
            if word in map.keys():
                vector.append(map.get(word))
            else:
                vector.append(0)
        count_xTrain_vectors.append(vector)

    count_xTest_vectors = []
    for map in xTest_vocab_maps:
        vector = []
        for word in scaled_word_list:
            if word in map.keys():
                vector.append(map.get(word))
            else:
                vector.append(0)
        count_xTest_vectors.append(vector)

    return count_xTrain_vectors, count_xTest_vectors


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    xTrain_set, yTrain_set, xTest_set, yTest_set = model_assessment(args.data)
    scaled_word_list = build_vocab_map(xTrain_set, yTrain_set, xTest_set, yTest_set)
    xTrain_vocab_maps, xTest_vocab_maps = each_email_vocab_map(xTrain_set, xTest_set)
    binary_xTrain_vectors, binary_xTest_vectors = construct_binary(xTrain_vocab_maps, xTest_vocab_maps, scaled_word_list)
    count_xTrain_vectors, count_xTest_vectors = construct_count(xTrain_vocab_maps, xTest_vocab_maps, scaled_word_list)

    # Now I have the vectors I need for each email and need to export it

    pd.DataFrame(binary_xTrain_vectors).to_csv("binary_xTrain.csv", header=scaled_word_list, index=None)
    pd.DataFrame(binary_xTest_vectors).to_csv("binary_xTest.csv", header=scaled_word_list, index=None)
    pd.DataFrame(count_xTrain_vectors).to_csv("count_xTrain.csv", header=scaled_word_list, index=None)
    pd.DataFrame(count_xTest_vectors).to_csv("count_xTest.csv", header=scaled_word_list, index=None)
    pd.DataFrame(yTrain_set).to_csv("yTrain_set.csv", header=['label'], index=None)
    pd.DataFrame(yTest_set).to_csv("yTest_set.csv", header=['label'], index=None)

if __name__ == "__main__":
    main()

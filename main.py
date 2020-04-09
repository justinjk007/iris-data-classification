from neural_net import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def test_trained_network(model, test_X, test_Y):
    _input = torch.tensor(test_X, dtype=torch.float)
    out = model(_input)
    error = 0
    tot = 0
    setosa_guessed_wrong = 0
    versicolor_guessed_wrong = 0
    virginica_guessed_wrong = 0
    setosa_guessed_right = 0
    versicolor_guessed_right = 0
    virginica_guessed_right = 0
    for exp, ans in zip(out, test_Y):
        tot += 1
        max_val, _ = torch.max(exp, 0)
        index = _.item()
        if (index == ans):
            if (index == 0):
                setosa_guessed_right += 1
            elif (index == 1):
                versicolor_guessed_right += 1
            else:
                virginica_guessed_right += 1
        else:
            error += 1
            if (index == 0):
                setosa_guessed_wrong += 1
            elif (index == 1):
                versicolor_guessed_wrong += 1
            else:
                virginica_guessed_wrong += 1

    print("Error is {}".format(error))
    print("Total is {}".format(tot))
    print("Setosa guessed wrong: {}".format(setosa_guessed_wrong))
    print("Versicolor guessed wrong: {}".format(versicolor_guessed_wrong))
    print("Virginica guessed wrong: {}".format(virginica_guessed_wrong))
    print("Setosa guessed right: {}".format(setosa_guessed_right))
    print("Versicolor guessed right: {}".format(versicolor_guessed_right))
    print("Virginica guessed right: {}".format(virginica_guessed_right))


def scale_input(row):
    """Scale input using mean and standard deviation, mean and SD are
     available in the database description"""
    # scaled_input = (input-mean)/standard deviation
    scale_sepal_length = (row['sepal_length'] - 5.84) / 0.83
    scale_sepal_width = (row['sepal_width'] - 3.05) / 0.43
    scale_petal_length = (row['petal_length'] - 3.76) / 1.76
    scale_petal_width = (row['petal_width'] - 1.20) / 0.76
    row['sepal_length'] = scale_sepal_length
    row['sepal_width'] = scale_sepal_width
    row['petal_length'] = scale_petal_length
    row['petal_width'] = scale_petal_width
    return row


def main():
    # load the dataset from file
    dataset = pd.read_csv('iris.data')

    # change labels to numbers
    dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2

    # unscaled_input = dataset[dataset.columns[0:4]].values
    # scaled_input = []
    # for i in unscaled_input:
    #     scaled_input.append(scale_input(i))

    # scale the dataset
    dataset = dataset.apply(scale_input, axis=1)

    # Split the dataset into test and train datasets, 30% of datasets for testing
    train_X, test_X, train_Y, test_Y = train_test_split(
        dataset[dataset.columns[0:4]].values,  # input values aka the x values
        dataset.species.values,  # output values aka the y values
        test_size=0.3,  # move 30% into test data
    )
    # convert output data to make them into tensors later
    train_Y = train_Y.astype(np.float32)
    test_Y = train_Y.astype(np.float32)

    # # scale the data
    # sc = StandardScaler()
    # train_X = sc.fit_transform(train_X)
    # test_X = sc.fit_transform(test_X)

    # load data into tensor
    _input = torch.tensor(train_X, dtype=torch.float)
    _output = torch.tensor(train_Y, dtype=torch.long)

    # This section is for plotting ##############################
    gene_array = []
    loss_array = []
    fig, ax = plt.subplots()
    ax.set(xlabel='generation',
           ylabel='mean sum squared error',
           title='Neural network, error loss after each generation')
    # This section is for plotting ##############################

    # input,output,hidden layer size
    model = NeuralNetwork(i=4, h1=3, h2=3, o=3)

    # loss function and optimizer
    lossFunction = nn.CrossEntropyLoss()
    # learning rate and momentum
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(1500):
        # Forward Pass
        output = model(_input)
        # Loss at each oteration by comparing to target
        loss = lossFunction(output, _output)

        # Backpropogating gradient of loss
        optimizer.zero_grad()
        loss.backward()

        # Updating parameters(weights and bias)
        optimizer.step()
        _loss = loss.item()

        gene_array.append(epoch)
        loss_array.append(_loss)
        print("Epoch {}, Training loss: {}".format(epoch, _loss / len(_input)))

    torch.save(model, "algo1.weights")
    # model = torch.load("68_error.weights")
    test_trained_network(model, test_X, test_Y)
    ax.plot(gene_array, loss_array)
    plt.show()


if __name__ == "__main__":
    main()

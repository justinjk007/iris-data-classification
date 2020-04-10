from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
import numpy as np
import pandas as pd
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, i, h1, h2, o):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(i, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, o)

        # Define relu activation and LogSoftmax output
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out = self.logSoftmax(self.layer3(out))
        return out


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

    # # another way to scale the data
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
    model = NeuralNetwork(i=4, h1=6, h2=4, o=3)

    # loss function and optimizer
    lossFunction = nn.CrossEntropyLoss()
    # lossFunction = nn.BCELoss(size_average=True)  # BinaryCrossEntropy
    # learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10000):
        optimizer.zero_grad()
        # Forward Pass
        output = model(_input)
        # Loss at each oteration by comparing to target
        loss = lossFunction(output, _output)
        # Backpropogating gradient of loss
        loss.backward()
        # Updating parameters(weights and bias)
        optimizer.step()
        _loss = loss.item()
        # Upadate graph points
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

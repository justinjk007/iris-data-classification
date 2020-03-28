import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from neural_net import NeuralNetwork
import data


def spit_out_name_from_output(output):
    # here _ is the index of the max value. For our output the
    # expected output neuron should be activated the most so the
    # maximum digits index numner should be the actual digit
    max_val, _ = torch.max(output, 0)
    index = _.item()
    if (index == 0):  # the index number
        return "setosa"
    elif (index == 1):
        return "versicolor"
    else:
        return "virginica"


def test_trained_network(ANN):
    print("Trained network being tested...\n")
    error_count = 0
    for i in range(len(data.testing_input_mod)):
        _input = torch.tensor(data.testing_input_mod[i], dtype=torch.float)
        exp_output = data.testing_output[i]
        print("Expected output  : ", exp_output)
        _output = ANN.predict(_input)
        print("Predicted output : ", spit_out_name_from_output(_output), "\n")
        if (spit_out_name_from_output(_output) != exp_output):
            error_count += 1
    print("Error count after testing", len(data.testing_input), "inputs: ",
          error_count)


def main():
    # load data
    _input = torch.tensor(data.training_input[:, [0, 1, 2, 3]].tolist(),
                          dtype=torch.float)  # first 4 coloumns input
    _output = torch.tensor(data.training_input[:, 4].tolist(),
                           dtype=torch.float)  # last coloumn is ouput

    # This section is for plotting ##############################
    gene_array = []
    loss_array = []
    fig, ax = plt.subplots()
    ax.set(xlabel='generation',
           ylabel='mean sum squared error',
           title='Neural network, error loss after each generation')
    # This section is for plotting ##############################

    ANN = NeuralNetwork(i=4, h1=3, h2=3, o=3)  # input,output,hidden layer size
    # weight training
    for i in range(1000):
        # mean sum squared error
        mean_error = torch.mean((_output - ANN(_input))**2).detach().item()
        print("Generation: " + str(i) + " error: " + str(mean_error))
        gene_array.append(i)
        loss_array.append(mean_error)
        ANN.train(_input, _output)

    torch.save(ANN, "algo1.weights")
    # ANN = torch.load("14_good.weights")
    test_trained_network(ANN)
    ax.plot(gene_array, loss_array)
    plt.show()


if __name__ == "__main__":
    main()

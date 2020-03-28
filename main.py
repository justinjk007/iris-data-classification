import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from neural_net import NeuralNetwork
import data


def spit_out_digit_from_output(output):
    # here _ is the index of the max value. For our output the
    # expected output neuron should be activated the most so the
    # maximum digits index numner should be the actual digit
    max_val, _ = torch.max(output, 0)
    return _.item()  # return the index number


def test_trained_network(ANN):
    print("Trained network being tested...\n")
    error_count = 0
    for i in range(len(data.testing_input)):
        _input = torch.tensor(data.testing_input[i], dtype=torch.float)
        exp_output = data.testing_output[i]
        print("Expected output  : ", exp_output)
        _output = ANN.predict(_input)
        print("Predicted output : ", spit_out_digit_from_output(_output), "\n")
        if (spit_out_digit_from_output(_output) != exp_output):
            error_count += 1
    print("Error count after testing", len(data.testing_input), "inputs: ",
          error_count)


def main():
    # load data
    _input = torch.tensor(data.training_input, dtype=torch.float)
    _output = torch.tensor(data.training_expected_output, dtype=torch.float)

    # This section is for plotting ##############################
    gene_array = []
    loss_array = []
    fig, ax = plt.subplots()
    ax.set(xlabel='generation',
           ylabel='mean sum squared error',
           title='Neural network, error loss after each generation')
    # This section is for plotting ##############################

    ANN = NeuralNetwork(i=45, o=10, h=5)  # input,output,hidden layer size
    # weight training
    for i in range(15000):
        # mean sum squared error
        mean_error = torch.mean((_output - ANN(_input))**2).detach().item()
        print("Generation: " + str(i) + " error: " + str(mean_error))
        gene_array.append(i)
        loss_array.append(mean_error)
        ANN.train(_input, _output)

    torch.save(ANN, "algo1.weights")
    ANN = torch.load("14_good.weights")
    test_trained_network(ANN)
    ax.plot(gene_array, loss_array)
    plt.show()


if __name__ == "__main__":
    main()

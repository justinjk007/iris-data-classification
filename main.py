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


def scale_input(array):
    """Scale input using mean and standard deviation, mean and SD are
     available in the database description"""
    # scaled_input = (input-mean)/standard deviation
    scale_sepal_length = (array[0] - 5.84) / 0.83
    scale_sepal_width = (array[1] - 3.05) / 0.43
    scale_petal_length = (array[2] - 3.76) / 1.76
    scale_petal_width = (array[3] - 1.20) / 0.76
    return [
        scale_sepal_length,
        scale_sepal_width,
        scale_petal_length,
        scale_petal_width,
    ]


def test_trained_network(ANN):
    print("Trained network being tested...\n")
    error_count = 0

    # scale the data with mean and SD
    unscaled_input = data.testing_input_mod
    scaled_input = []
    for i in unscaled_input:
        scaled_input.append(scale_input(i))

    for i in range(len(data.testing_input_mod)):
        _input = torch.tensor(scaled_input[i], dtype=torch.float)
        exp_output = data.testing_output[i]
        print("Expected output  : ", exp_output)
        _output = ANN.predict(_input)
        print("Predicted output : ", spit_out_name_from_output(_output), "\n")
        if (spit_out_name_from_output(_output) != exp_output):
            error_count += 1
    print("Error count after testing", len(data.testing_input), "inputs: ",
          error_count)


def main():
    # scale the data with mean and SD
    unscaled_input = data.training_input_mod
    scaled_input = []
    for i in unscaled_input:
        scaled_input.append(scale_input(i))

    # load data into tensor
    _input = torch.tensor(scaled_input, dtype=torch.float)
    _output = torch.tensor(data.training_expected_output, dtype=torch.float)

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
    for i in range(15000):
        # mean sum squared error
        mean_error = torch.mean((_output - ANN(_input))**2).detach().item()
        print("Generation: " + str(i) + " error: " + str(mean_error))
        gene_array.append(i)
        loss_array.append(mean_error)
        ANN.train(_input, _output)

    torch.save(ANN, "algo1.weights")
    # ANN = torch.load("1_error.weights")
    test_trained_network(ANN)
    ax.plot(gene_array, loss_array)
    plt.show()


if __name__ == "__main__":
    main()

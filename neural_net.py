import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, i, h1, h2, o):
        super().__init__()  # recommended by pytorch
        self.input_num = i
        self.hidden1_num = h1
        self.hidden2_num = h2
        self.output_num = o
        # returns a tensor with random values, weights applicable for input
        # layer and hidden layer
        self.w1 = torch.randn(self.input_num, self.hidden1_num)
        self.w2 = torch.randn(self.hidden1_num, self.hidden2_num)
        self.w3 = torch.randn(self.hidden2_num, self.output_num)

    def forward(self, _input):
        self.z = torch.matmul(_input, self.w1)  # matrix multiplication
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = torch.matmul(self.z2, self.w2)  # apply weights on hidden 1
        self.z4 = self.sigmoid(self.z3)  # activation function
        self.z5 = torch.matmul(self.z4, self.w3)  # apply weights on hidden 2
        output = self.sigmoid(self.z5)  # activation function
        return output

    def sigmoid(self, _input):
        # _input can be a tensor of any size
        return 1 / (1 + torch.exp(-_input))

    def sigmoidPrime(self, _input):
        # find the derivative the tensor _input
        return _input * (1 - _input)

    def backward(self, _input, exp_output, _output):
        self.output_error = exp_output - _output  # error in output
        self.output_delta = self.output_error * self.sigmoidPrime(_output)

        self.z2_error = torch.matmul(self.output_delta, torch.t(self.w2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.z4_error = torch.matmul(self.output_delta, torch.t(self.w3))
        self.z4_delta = self.z4_error * self.sigmoidPrime(self.z4)

        self.w1 += torch.matmul(torch.t(_input), self.z2_delta)
        self.w2 += torch.matmul(torch.t(self.z2), self.z4_delta)
        self.w3 += torch.matmul(torch.t(self.z4), self.output_delta)

    def train(self, _input, exp_output):
        # forward + backward pass for training
        _output = self.forward(_input)
        self.backward(_input, exp_output, _output)

    def predict(self, _input):
        # Once the neural network is trained, doing a prediction means running
        # the input through a forward pass and collecting the output.
        return self.forward(_input)

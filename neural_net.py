import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, i, o, h):
        super().__init__()  # recommended by pytorch
        self.input_num = i
        self.output_num = o
        self.hidden_num = h
        # returns a tensor with random values, weights applicable for input
        # layer and hidden layer
        self.w1 = torch.randn(self.input_num, self.hidden_num)
        self.w2 = torch.randn(self.hidden_num, self.output_num)

    def forward(self, _input):
        self.z = torch.matmul(_input, self.w1)  # matrix multiplication
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = torch.matmul(self.z2, self.w2)
        output = self.sigmoid(self.z3)  # final activation function
        return output

    def sigmoid(self, _input):
        # _input can be a tensor of any size
        return 1 / (1 + torch.exp(-_input))

    def sigmoidPrime(self, _input):
        # find the derivative the tensor _input
        return _input * (1 - _input)

    def backward(self, _input, exp_output, _output):
        self.output_error = exp_output - _output  # error in output
        # derivative of sig to error
        self.output_delta = self.output_error * self.sigmoidPrime(_output)
        self.z2_error = torch.matmul(self.output_delta, torch.t(self.w2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.w1 += torch.matmul(torch.t(_input), self.z2_delta)
        self.w2 += torch.matmul(torch.t(self.z2), self.output_delta)

    def train(self, _input, exp_output):
        # forward + backward pass for training
        _output = self.forward(_input)
        self.backward(_input, exp_output, _output)

    def predict(self, _input):
        # Once the neural network is trained, doing a prediction means running
        # the input through a forward pass and collecting the output.
        return self.forward(_input)

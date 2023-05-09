import numpy as np
import torch

torch.manual_seed(2023)

def activation_func(x):
    #TODO Implement one of these following activation function: softmax, tanh, ReLU, leaky ReLU
    epsilon = 0.01   # Only use this variable if you choose Leaky ReLU
    return max(epsilon*x, x)

def softmax(x):
    #TODO Implement softmax function here
    norm = np.exp(x - np.max(x))
    return norm / norm.sum()

# Define the size of each layer in the network
num_input = 784  # Number of node in input layer (28x28)
num_hidden_1 = 128  # Number of nodes in hidden layer 1
num_hidden_2 = 256  # Number of nodes in hidden layer 2
num_hidden_3 = 128  # Number of nodes in hidden layer 3
num_classes = 10  # Number of nodes in output layer

# Random input
input_data = torch.randn((1, num_input))
# Weights for inputs to hidden layer 1
W1 = torch.randn(num_input, num_hidden_1)
# Weights for hidden layer 1 to hidden layer 2
W2 = torch.randn(num_hidden_1, num_hidden_2)
# Weights for hidden layer 2 to hidden layer 3
W3 = torch.randn(num_hidden_2, num_hidden_3)
# Weights for hidden layer 3 to output layer
W4 = torch.randn(num_hidden_3, num_classes)

# and bias terms for hidden and output layers
B1 = torch.randn((1, num_hidden_1))
B2 = torch.randn((1, num_hidden_2))
B3 = torch.randn((1, num_hidden_3))
B4 = torch.randn((1, num_classes))

#TODO Calculate forward pass of the network here. Result should have the shape of [1,10]
# Dont forget to check if sum of result = 1.0

hidden_1 = []
hidden_2 = []
hidden_3 = []
result = []

#With hidden_layer1
for i in range(0, num_hidden_1):
    sum = np.dot([item[i] for item in W1], input_data[0]) + B1[0][i]
    output = activation_func(sum)
    hidden_1.append(sum)
hidden_1 = torch.tensor(hidden_1)
hidden_1 = torch.reshape(hidden_1, (1, len(hidden_1)))

#With hidden_layer2
for i in range(0, num_hidden_2):
    sum = np.dot([item[i] for item in W2], hidden_1[0]) + B2[0][i]
    output = activation_func(sum)
    hidden_2.append(sum)
hidden_2 = torch.tensor(hidden_2)
hidden_2 = torch.reshape(hidden_2, (1, len(hidden_2)))

#With hidden_layer3
for i in range(0, num_hidden_3):
    sum = np.dot([item[i] for item in W3], hidden_2[0]) + B3[0][i]
    output = activation_func(sum)
    hidden_3.append(sum)
hidden_3 = torch.tensor(hidden_3)
hidden_3 = torch.reshape(hidden_3, (1, len(hidden_3)))

#With Result
for i in range(0, num_classes):
    sum = np.dot([item[i] for item in W4], hidden_3[0]) + B4[0][i]
    output = activation_func(sum)
    result.append(sum)

result = softmax(result)
result = torch.tensor(result)
result = torch.reshape(result, (1,10))
print(result, result.shape, torch.sum(result))

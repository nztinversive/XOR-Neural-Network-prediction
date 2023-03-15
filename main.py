import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_network(input_nodes, hidden_nodes, output_nodes):
    weights_1 = np.random.rand(input_nodes, hidden_nodes) - 0.5
    weights_2 = np.random.rand(hidden_nodes, output_nodes) - 0.5
    biases_1 = np.zeros((1, hidden_nodes))
    biases_2 = np.zeros((1, output_nodes))

    return weights_1, weights_2, biases_1, biases_2

def forward_pass(X, weights_1, weights_2, biases_1, biases_2):
    hidden_layer_input = np.dot(X, weights_1) + biases_1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_2) + biases_2
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

def backpropagation(X, Y, weights_1, weights_2, biases_1, biases_2, hidden_layer_output, output_layer_output, learning_rate):
    output_error = Y - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_error = np.dot(output_delta, weights_2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    weights_2 += np.dot(hidden_layer_output.T, output_delta) * learning_rate
    biases_2 += np.sum(output_delta, axis=0) * learning_rate
    weights_1 += np.dot(X.T, hidden_delta) * learning_rate
    biases_1 += np.sum(hidden_delta, axis=0) * learning_rate

def train(X, Y, input_nodes, hidden_nodes, output_nodes, learning_rate, iterations):
    weights_1, weights_2, biases_1, biases_2 = initialize_network(input_nodes, hidden_nodes, output_nodes)

    for _ in range(iterations):
        hidden_layer_output, output_layer_output = forward_pass(X, weights_1, weights_2, biases_1, biases_2)
        backpropagation(X, Y, weights_1, weights_2, biases_1, biases_2, hidden_layer_output, output_layer_output, learning_rate)

    return weights_1, weights_2, biases_1, biases_2

def predict(X, weights_1, weights_2, biases_1, biases_2):
    _, output_layer_output = forward_pass(X, weights_1, weights_2, biases_1, biases_2)
    return output_layer_output

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

input_nodes = 2
hidden_nodes = 4
output_nodes = 1
learning_rate = 0.5
iterations = 10000

weights_1, weights_2, biases_1, biases_2 = train(X, Y, input_nodes, hidden_nodes, output_nodes, learning_rate, iterations)

test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_output = predict(test_input, weights_1, weights_2, biases_1, biases_2)

print("Input:")
print(test_input)
print("Predicted Output:")
print(test_output.round())

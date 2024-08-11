import numpy as np
import json

def sigmoid_activation(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

def cost_derivative(output_activation, target):
    return (output_activation - target)

class Network:
    def __init__(self, layer_sizes: list):
        self.layer_count = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(b, 1) for b in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, x):
        activation = x
        for b, w in zip(self.biases, self.weights):
            activation = sigmoid_activation(np.dot(w, activation) + b)

        return activation
    
    def gradient_descent(self, training_data, epochs, batch_size, learning_rate):
        length_of_training_data = len(training_data)

        for epoch in range(epochs):
            data_batches = [training_data[k:k + batch_size] for k in range(0, length_of_training_data, batch_size)]
            
            for batch in data_batches:
                self.update_batch(batch, learning_rate)
                
            print(f"Epoch {epoch} completed")

    def update_batch(self, data_batch, learning_rate):
        updated_gradient_weights = [np.zeros(w.shape) for w in self.weights]
        updated_gradient_biases = [np.zeros(b.shape) for b in self.biases]

        counter = 0

        for data, target in data_batch:
            counter += 1
            gradients = self.backpropagation(data, target)
            updated_gradient_weights = [updated_gradient_weight + gradient_weight for gradient_weight, updated_gradient_weight in zip(updated_gradient_weights, gradients[1])]
            updated_gradient_biases = [updated_gradient_bias + gradient_bias for updated_gradient_bias, gradient_bias in zip(updated_gradient_biases, gradients[0])]
        
        self.weights = [weight - learning_rate / len(data_batch) * weight_update_direction  for weight, weight_update_direction in zip(self.weights, updated_gradient_weights)]
        self.biases = [bias - learning_rate / len(data_batch) * bias_update_direction for bias, bias_update_direction in zip(self.biases, updated_gradient_biases)]
    
    def backpropagation(self, input, target) -> tuple:
        gradients_of_weight = [np.zeros(w.shape) for w in self.weights]
        gradients_of_biases = [np.zeros(b.shape) for b in self.biases]

        current_activation = input
        activations = [current_activation]
        neurone_outputs = []
        for b, w in zip(self.biases, self.weights):
            neurone_output = np.dot(w, current_activation) + b
            neurone_outputs.append(neurone_output)
            current_activation = sigmoid_activation(neurone_output)
            activations.append(current_activation)

        default_gradient = cost_derivative(activations[-1], target) * sigmoid_prime(neurone_outputs[-1])
        gradients_of_weight[-1] = np.dot(default_gradient, activations[-2].transpose())
        gradients_of_biases[-1] = default_gradient

        for layer in range(2, self.layer_count):
            default_gradient = np.dot(self.weights[-layer+1].transpose(), default_gradient) * sigmoid_prime(neurone_outputs[-layer])
            gradients_of_weight[-layer] = np.dot(default_gradient, np.array(activations[-layer-1]).transpose())
            gradients_of_biases[-layer] = default_gradient

        return(gradients_of_biases, gradients_of_weight)

    def load_parameters(self, file_name):
        with open(file_name, "r") as file:
            weight_biases = json.load(file)

        self.weights = [np.array(w) for w in weight_biases["weights"]]
        self.biases = [np.array(b) for b in weight_biases["biases"]]
   
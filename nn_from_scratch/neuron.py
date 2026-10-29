
import numpy as np


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


# Derivatives of activation functions
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu_derivative(x):
    return (np.asarray(x) > 0).astype(float)


class Neuron:
    def __init__(self, input_size, activation="sigmoid", seed=42):
        rng = np.random.default_rng(seed)

        self.weights = rng.normal(0, 0.1, input_size)
        self.bias = 0.0

        self.activation = activation

    def forward(self, x):
        x = np.asarray(x, dtype=float)

        if x.shape != self.weights.shape:
            raise ValueError(
                f"Expected input shape {self.weights.shape}, "
                f"got {x.shape}"
            )

        z = np.dot(x, self.weights) + self.bias

        if self.activation == "sigmoid":
            return sigmoid(z)

        elif self.activation == "tanh":
            return tanh(z)

        elif self.activation == "relu":
            return relu(z)

        else:
            raise ValueError("Unsupported activation function")


if __name__ == "__main__":
    inputs = np.array([0.5, 0.3, 0.2])

    neuron = Neuron(
        input_size=3,
        activation="sigmoid"
    )

    output = neuron.forward(inputs)

    print("Input:", inputs)
    print("Weights:", neuron.weights)
    print("Bias:", neuron.bias)
    print("Neuron Output:", output)

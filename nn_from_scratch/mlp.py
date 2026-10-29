
"""
Phase 1: Neural Network Fundamentals
Module: Multilayer Perceptron (MLP)

Implements:
1. Activation selection
2. Mean squared error
3. Softmax
4. Cross-entropy loss
5. Forward propagation
6. Backpropagation
"""

import numpy as np

from .neuron import (
    sigmoid,
    tanh,
    relu,
    sigmoid_derivative,
    tanh_derivative,
    relu_derivative,
)


# ----------------------------------------
# 1. ACTIVATION FUNCTION SELECTION
# ----------------------------------------

def activate(z, name):
    if name == "sigmoid":
        return sigmoid(z)

    elif name == "tanh":
        return tanh(z)

    elif name == "relu":
        return relu(z)

    elif name == "linear":
        return z

    else:
        raise ValueError(
            f"Unsupported activation: {name}"
        )


def activation_gradient(z, name):
    if name == "sigmoid":
        return sigmoid_derivative(z)

    elif name == "tanh":
        return tanh_derivative(z)

    elif name == "relu":
        return relu_derivative(z)

    elif name == "linear":
        return np.ones_like(z)

    else:
        raise ValueError(
            f"Unsupported activation: {name}"
        )


# ----------------------------------------
# 2. MEAN SQUARED ERROR LOSS
# ----------------------------------------

def mse_loss(predictions, targets):
    diff = predictions - targets

    loss = np.mean(diff ** 2)

    gradient = 2 * diff / diff.size

    return loss, gradient


# ----------------------------------------
# 3. SOFTMAX
# ----------------------------------------

def softmax(logits):
    shifted = logits - np.max(
        logits, axis=1, keepdims=True
    )

    exp_values = np.exp(shifted)

    probabilities = exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

    return probabilities


# ----------------------------------------
# 4. SOFTMAX CROSS-ENTROPY LOSS
# ----------------------------------------

def cross_entropy_loss(logits, targets):
    probabilities = softmax(logits)

    batch_size = logits.shape[0]

    loss = -np.sum(
        targets * np.log(
            np.clip(probabilities, 1e-15, 1.0)
        )
    ) / batch_size

    gradient = (
        probabilities - targets
    ) / batch_size

    return loss, gradient


# ----------------------------------------
# 5. MULTILAYER PERCEPTRON
# ----------------------------------------

class MLP:

    def __init__(
        self,
        layer_sizes,
        hidden_activation="tanh",
        output_activation="linear",
        seed=42,
    ):

        if len(layer_sizes) < 2:
            raise ValueError(
                "At least input and output sizes are required"
            )

        if any(size <= 0 for size in layer_sizes):
            raise ValueError(
                "Layer sizes must be positive"
            )

        rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []
        self.activations = []

        # Initialize each layer
        for i in range(len(layer_sizes) - 1):

            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]

            scale = np.sqrt(
                2.0 / (input_size + output_size)
            )

            weight = rng.normal(
                0.0,
                scale,
                (input_size, output_size)
            )

            bias = np.zeros((1, output_size))

            self.weights.append(weight)
            self.biases.append(bias)

            if i == len(layer_sizes) - 2:
                self.activations.append(
                    output_activation
                )
            else:
                self.activations.append(
                    hidden_activation
                )

        self.grad_weights = []
        self.grad_biases = []

    # ------------------------------------
    # 6. FORWARD PROPAGATION
    # ------------------------------------

    def forward(self, x):
        a = np.asarray(x, dtype=float)

        if a.ndim != 2:
            raise ValueError(
                "Input must have shape (batch, features)"
            )

        if a.shape[1] != self.weights[0].shape[0]:
            raise ValueError(
                "Input feature count does not match network"
            )

        self.cached_activations = [a]
        self.cached_z = []

        for weight, bias, activation_name in zip(
            self.weights,
            self.biases,
            self.activations,
        ):

            z = a @ weight + bias

            a = activate(
                z,
                activation_name
            )

            self.cached_z.append(z)
            self.cached_activations.append(a)

        return a

    # ------------------------------------
    # 7. BACKPROPAGATION
    # ------------------------------------

    def backward(self, grad_output):
        if not hasattr(self, "cached_z"):
            raise RuntimeError(
                "Call forward() before backward()"
            )

        gradient = np.asarray(
            grad_output,
            dtype=float
        )

        if gradient.shape != self.cached_activations[-1].shape:
            raise ValueError(
                "Output gradient has incorrect shape"
            )

        self.grad_weights = [
            None
        ] * len(self.weights)

        self.grad_biases = [
            None
        ] * len(self.biases)

        for i in reversed(range(len(self.weights))):

            z = self.cached_z[i]

            # Chain rule through activation
            grad_z = gradient * activation_gradient(
                z,
                self.activations[i]
            )

            previous_activation = (
                self.cached_activations[i]
            )

            # Gradient with respect to weights
            self.grad_weights[i] = (
                previous_activation.T @ grad_z
            )

            # Gradient with respect to biases
            self.grad_biases[i] = np.sum(
                grad_z,
                axis=0,
                keepdims=True
            )

            # Gradient for previous layer
            gradient = (
                grad_z @ self.weights[i].T
            )

        return gradient


# ----------------------------------------
# 8. DEMONSTRATION
# ----------------------------------------

if __name__ == "__main__":

    model = MLP(
        layer_sizes=[3, 4, 2],
        hidden_activation="tanh",
        output_activation="linear",
        seed=42
    )

    x = np.array([
        [0.5, 0.2, 0.1],
        [0.9, 0.4, 0.7]
    ])

    targets = np.array([
        [1, 0],
        [0, 1]
    ])

    predictions = model.forward(x)

    loss, gradient = cross_entropy_loss(
        predictions,
        targets
    )

    model.backward(gradient)

    print("=" * 45)
    print("MLP FORWARD AND BACKWARD DEMO")
    print("=" * 45)

    print("Input shape:", x.shape)
    print("Output shape:", predictions.shape)
    print("Loss:", loss)

    for i in range(len(model.weights)):
        print(f"\nLayer {i + 1}")
        print("Weights:", model.weights[i].shape)
        print("Biases:", model.biases[i].shape)
        print(
            "Weight gradients:",
            model.grad_weights[i].shape
        )
        print(
            "Bias gradients:",
            model.grad_biases[i].shape
        )

    print("\nMLP demonstration completed!")

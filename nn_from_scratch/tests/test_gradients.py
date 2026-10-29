
"""
Phase 1 - Numerical Gradient Checking

Project:
Sign Language Recognition and Translation Assistant

Tests:
1. Finite-difference gradients for MSE
2. Finite-difference gradients for cross-entropy
3. Every weight and bias in every layer
4. Gradients with different activation functions
5. Gradients with respect to network inputs
"""

import numpy as np
import pytest

from nn_from_scratch.mlp import (
    MLP,
    mse_loss,
    cross_entropy_loss,
)


# ----------------------------------------
# 1. NUMERICAL GRADIENT
# ----------------------------------------

def numerical_gradient(
    parameter,
    loss_function,
    h=1e-5,
):
    """
    Estimate the gradient of each parameter
    using central finite differences.
    """

    gradient = np.zeros_like(
        parameter,
        dtype=np.float64
    )

    for index in np.ndindex(parameter.shape):

        original_value = parameter[index]

        try:
            # Increase parameter by h
            parameter[index] = original_value + h
            loss_plus = loss_function()

            # Decrease parameter by h
            parameter[index] = original_value - h
            loss_minus = loss_function()

        finally:
            # Restore original parameter
            parameter[index] = original_value

        # Central difference formula
        gradient[index] = (
            loss_plus - loss_minus
        ) / (2 * h)

    return gradient


# ----------------------------------------
# 2. RELATIVE ERROR
# ----------------------------------------

def relative_error(analytical, numerical):
    """
    Calculate normalized gradient difference.
    """

    numerator = np.linalg.norm(
        analytical - numerical
    )

    denominator = max(
        1e-12,
        np.linalg.norm(analytical)
        + np.linalg.norm(numerical)
    )

    return numerator / denominator


# ----------------------------------------
# 3. CHECK ALL MODEL PARAMETERS
# ----------------------------------------

def check_model_gradients(
    model,
    x,
    targets,
    loss_function,
    threshold=1e-5,
):
    """
    Compare analytical and numerical gradients
    for every weight and bias.
    """

    # Forward propagation
    predictions = model.forward(x)

    # Calculate loss and output gradient
    loss, grad_output = loss_function(
        predictions,
        targets
    )

    # Analytical backpropagation
    model.backward(grad_output)

    assert np.isfinite(loss)

    # Check weight gradients
    for layer_index in range(len(model.weights)):

        weight = model.weights[layer_index]

        analytical = (
            model.grad_weights[layer_index].copy()
        )

        def calculate_loss():
            predictions = model.forward(x)
            value, _ = loss_function(
                predictions,
                targets
            )
            return value

        numerical = numerical_gradient(
            weight,
            calculate_loss
        )

        error = relative_error(
            analytical,
            numerical
        )

        print(
            f"\nLayer {layer_index + 1} "
            f"weight relative error: {error:.3e}"
        )

        assert error < threshold, (
            f"Weight gradient check failed "
            f"at layer {layer_index + 1}: {error}"
        )

    # Check bias gradients
    for layer_index in range(len(model.biases)):

        bias = model.biases[layer_index]

        analytical = (
            model.grad_biases[layer_index].copy()
        )

        def calculate_loss():
            predictions = model.forward(x)
            value, _ = loss_function(
                predictions,
                targets
            )
            return value

        numerical = numerical_gradient(
            bias,
            calculate_loss
        )

        error = relative_error(
            analytical,
            numerical
        )

        print(
            f"\nLayer {layer_index + 1} "
            f"bias relative error: {error:.3e}"
        )

        assert error < threshold, (
            f"Bias gradient check failed "
            f"at layer {layer_index + 1}: {error}"
        )


# ----------------------------------------
# 4. MSE GRADIENT CHECK
# ----------------------------------------

@pytest.mark.parametrize(
    "hidden_activation",
    ["sigmoid", "tanh", "relu"],
)
def test_mse_gradients(hidden_activation):

    model = MLP(
        layer_sizes=[2, 3, 1],
        hidden_activation=hidden_activation,
        output_activation="linear",
        seed=42
    )

    x = np.array([
        [0.2, 0.4],
        [0.6, 0.8]
    ], dtype=np.float64)

    targets = np.array([
        [0.3],
        [0.7]
    ], dtype=np.float64)

    check_model_gradients(
        model,
        x,
        targets,
        mse_loss
    )


# ----------------------------------------
# 5. CROSS-ENTROPY GRADIENT CHECK
# ----------------------------------------

@pytest.mark.parametrize(
    "hidden_activation",
    ["sigmoid", "tanh", "relu"],
)
def test_cross_entropy_gradients(
    hidden_activation
):

    model = MLP(
        layer_sizes=[2, 4, 3],
        hidden_activation=hidden_activation,
        output_activation="linear",
        seed=24
    )

    x = np.array([
        [0.1, 0.3],
        [0.5, 0.7],
        [0.9, 0.2]
    ], dtype=np.float64)

    targets = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    check_model_gradients(
        model,
        x,
        targets,
        cross_entropy_loss
    )


# ----------------------------------------
# 6. INPUT GRADIENT CHECK
# ----------------------------------------

def test_input_gradients():

    model = MLP(
        layer_sizes=[2, 3, 1],
        hidden_activation="tanh",
        output_activation="linear",
        seed=42
    )

    x = np.array([
        [0.2, 0.4],
        [0.6, 0.8]
    ], dtype=np.float64)

    targets = np.array([
        [0.3],
        [0.7]
    ], dtype=np.float64)

    predictions = model.forward(x)

    _, grad_output = mse_loss(
        predictions,
        targets
    )

    analytical = model.backward(
        grad_output
    )

    def calculate_loss():
        predictions = model.forward(x)
        value, _ = mse_loss(
            predictions,
            targets
        )
        return value

    numerical = numerical_gradient(
        x,
        calculate_loss
    )

    error = relative_error(
        analytical,
        numerical
    )

    print(
        f"\nInput gradient relative error: "
        f"{error:.3e}"
    )

    assert error < 1e-5

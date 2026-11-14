
import numpy as np
import pytest


def test_tensor_shapes():
    scalar = np.array(10)
    vector = np.array([1, 2, 3, 4])
    matrix = np.zeros((2, 3))
    tensor = np.zeros((2, 3, 4))

    assert scalar.shape == ()
    assert vector.shape == (4,)
    assert matrix.shape == (2, 3)
    assert tensor.shape == (2, 3, 4)


def test_matrix_multiplication():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    result = A @ B

    expected = np.array([
        [19, 22],
        [43, 50]
    ])

    np.testing.assert_array_equal(result, expected)


def test_broadcasting():
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    bias = np.array([10, 20, 30])

    result = matrix + bias

    expected = np.array([
        [11, 22, 33],
        [14, 25, 36]
    ])

    np.testing.assert_array_equal(result, expected)


def test_invalid_broadcasting():
    matrix = np.ones((2, 3))
    wrong_bias = np.ones(2)

    with pytest.raises(ValueError):
        _ = matrix + wrong_bias


def test_reductions():
    data = np.array([
        [1, 5, 3],
        [4, 2, 6]
    ])

    assert np.sum(data) == 21
    assert np.mean(data) == 3.5
    assert np.max(data) == 6
    assert np.argmax(data) == 5

    np.testing.assert_array_equal(
        np.sum(data, axis=1),
        [9, 12]
    )


def test_neural_network_layer():
    X = np.array([
        [0.5, 0.2, 0.1],
        [0.9, 0.4, 0.7]
    ])

    W = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ])

    b = np.array([0.1, 0.2])

    output = np.maximum(0, X @ W + b)

    assert output.shape == (2, 2)

    np.testing.assert_allclose(
        output,
        [[0.26, 0.44], [0.66, 0.96]],
        atol=1e-10
    )

from nn_from_scratch.neuron import (
    Neuron,
    sigmoid,
    tanh,
    relu,
    sigmoid_derivative,
    tanh_derivative,
    relu_derivative,
)


def test_sigmoid():
    assert np.isclose(sigmoid(0), 0.5)


def test_tanh():
    assert np.isclose(tanh(0), 0.0)


def test_relu():
    assert relu(-5) == 0
    assert relu(5) == 5


def test_activation_derivatives():
    assert np.isclose(sigmoid_derivative(0), 0.25)
    assert np.isclose(tanh_derivative(0), 1.0)
    assert relu_derivative(-1) == 0
    assert relu_derivative(1) == 1


def test_neuron_output():
    neuron = Neuron(3, activation="sigmoid")
    x = np.array([0.5, 0.3, 0.2])

    result = neuron.forward(x)

    assert np.isfinite(result)
    assert 0 <= result <= 1


def test_neuron_input_shape():
    neuron = Neuron(3)

    with pytest.raises(ValueError):
        neuron.forward(np.array([1.0, 2.0]))

# ----------------------------------------
# MLP UNIT TESTS
# ----------------------------------------

from nn_from_scratch.mlp import (
    MLP,
    softmax,
    mse_loss,
    cross_entropy_loss,
)


def test_softmax_probabilities():
    logits = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 0.0]
    ])

    probabilities = softmax(logits)

    assert probabilities.shape == (2, 3)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(2),
        atol=1e-10
    )


def test_mse_loss():
    predictions = np.array([[2.0, 4.0]])
    targets = np.array([[1.0, 2.0]])

    loss, gradient = mse_loss(
        predictions,
        targets
    )

    assert np.isclose(loss, 2.5)

    np.testing.assert_allclose(
        gradient,
        [[1.0, 2.0]]
    )


def test_mlp_forward_shape():
    model = MLP([3, 4, 2])

    x = np.ones((5, 3))

    output = model.forward(x)

    assert output.shape == (5, 2)


def test_mlp_backward_shapes():
    model = MLP([3, 4, 2])

    x = np.ones((5, 3))
    targets = np.zeros((5, 2))

    predictions = model.forward(x)

    _, gradient = mse_loss(
        predictions,
        targets
    )

    model.backward(gradient)

    assert model.grad_weights[0].shape == (3, 4)
    assert model.grad_weights[1].shape == (4, 2)

    assert model.grad_biases[0].shape == (1, 4)
    assert model.grad_biases[1].shape == (1, 2)


def test_cross_entropy():
    logits = np.array([[2.0, 1.0]])
    targets = np.array([[1.0, 0.0]])

    loss, gradient = cross_entropy_loss(
        logits,
        targets
    )

    assert np.isfinite(loss)
    assert gradient.shape == logits.shape

    assert np.isclose(
        gradient.sum(),
        0.0,
        atol=1e-10
    )


def test_mlp_reproducibility():
    model1 = MLP([3, 4, 2], seed=42)
    model2 = MLP([3, 4, 2], seed=42)

    for w1, w2 in zip(
        model1.weights,
        model2.weights
    ):
        np.testing.assert_array_equal(w1, w2)

from nn_from_scratch.optimizers import SGD, SGDMomentum, Adam


def make_optimizer_test_model():
    model = MLP([2, 1], seed=42)
    model.weights[0][:] = 1.0
    model.biases[0][:] = 0.0
    model.grad_weights = [
        np.array([[0.5], [-0.25]])
    ]
    model.grad_biases = [
        np.array([[0.2]])
    ]
    return model


def test_sgd_update():
    model = make_optimizer_test_model()
    optimizer = SGD(learning_rate=0.1)

    optimizer.step(model)

    np.testing.assert_allclose(
        model.weights[0],
        [[0.95], [1.025]]
    )
    np.testing.assert_allclose(
        model.biases[0],
        [[-0.02]]
    )


def test_momentum_first_step():
    model = make_optimizer_test_model()
    optimizer = SGDMomentum(
        learning_rate=0.1,
        momentum=0.9
    )

    optimizer.step(model)

    np.testing.assert_allclose(
        model.weights[0],
        [[0.95], [1.025]]
    )


def test_momentum_accumulation():
    model = make_optimizer_test_model()
    optimizer = SGDMomentum(
        learning_rate=0.1,
        momentum=0.9
    )

    optimizer.step(model)
    optimizer.step(model)

    np.testing.assert_allclose(
        model.weights[0],
        [[0.855], [1.0725]]
    )


def test_adam_first_step():
    model = make_optimizer_test_model()
    optimizer = Adam(learning_rate=0.01)

    optimizer.step(model)

    expected_weights = np.array([
        [1.0 - 0.01 * 0.5 / (0.5 + 1e-8)],
        [1.0 + 0.01 * 0.25 / (0.25 + 1e-8)]
    ])

    np.testing.assert_allclose(
        model.weights[0],
        expected_weights,
        atol=1e-10
    )

    assert optimizer.t == 1

def test_mlp_learns_xor():
    import numpy as np
    from nn_from_scratch.mlp import MLP, cross_entropy_loss
    from nn_from_scratch.optimizers import Adam

    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])

    y = np.array([0, 1, 1, 0])
    targets = np.eye(2)[y]

    model = MLP(
        layer_sizes=[2, 8, 2],
        hidden_activation="tanh",
        output_activation="linear",
        seed=42
    )

    optimizer = Adam(learning_rate=0.01)

    for epoch in range(3000):
        logits = model.forward(X)
        loss, gradient = cross_entropy_loss(
            logits, targets
        )

        model.backward(gradient)
        optimizer.step(model)

    predictions = np.argmax(model.forward(X), axis=1)

    assert np.array_equal(predictions, y)

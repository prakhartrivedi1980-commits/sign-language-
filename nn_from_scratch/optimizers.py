
import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, model):
        for i in range(len(model.weights)):
            # w = w - learning_rate * gradient
            model.weights[i] -= (
                self.learning_rate * model.grad_weights[i]
            )
            model.biases[i] -= (
                self.learning_rate * model.grad_biases[i]
            )


class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def step(self, model):
        if self.velocity_w is None:
            self.velocity_w = [
                np.zeros_like(w) for w in model.weights
            ]
            self.velocity_b = [
                np.zeros_like(b) for b in model.biases
            ]

        for i in range(len(model.weights)):
            # v = momentum * v + gradient
            self.velocity_w[i] = (
                self.momentum * self.velocity_w[i]
                + model.grad_weights[i]
            )
            self.velocity_b[i] = (
                self.momentum * self.velocity_b[i]
                + model.grad_biases[i]
            )

            # parameter = parameter - learning_rate * v
            model.weights[i] -= (
                self.learning_rate * self.velocity_w[i]
            )
            model.biases[i] -= (
                self.learning_rate * self.velocity_b[i]
            )


class Adam:
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def step(self, model):
        if self.m_w is None:
            self.m_w = [
                np.zeros_like(w) for w in model.weights
            ]
            self.v_w = [
                np.zeros_like(w) for w in model.weights
            ]
            self.m_b = [
                np.zeros_like(b) for b in model.biases
            ]
            self.v_b = [
                np.zeros_like(b) for b in model.biases
            ]

        self.t += 1

        for i in range(len(model.weights)):
            gw = model.grad_weights[i]
            gb = model.grad_biases[i]

            # First moment: exponential average of gradients
            self.m_w[i] = (
                self.beta1 * self.m_w[i]
                + (1 - self.beta1) * gw
            )
            self.m_b[i] = (
                self.beta1 * self.m_b[i]
                + (1 - self.beta1) * gb
            )

            # Second moment: exponential average of squared gradients
            self.v_w[i] = (
                self.beta2 * self.v_w[i]
                + (1 - self.beta2) * gw**2
            )
            self.v_b[i] = (
                self.beta2 * self.v_b[i]
                + (1 - self.beta2) * gb**2
            )

            # Bias correction
            mw_hat = self.m_w[i] / (1 - self.beta1**self.t)
            mb_hat = self.m_b[i] / (1 - self.beta1**self.t)
            vw_hat = self.v_w[i] / (1 - self.beta2**self.t)
            vb_hat = self.v_b[i] / (1 - self.beta2**self.t)

            # Adam parameter update
            model.weights[i] -= (
                self.learning_rate * mw_hat
                / (np.sqrt(vw_hat) + self.epsilon)
            )
            model.biases[i] -= (
                self.learning_rate * mb_hat
                / (np.sqrt(vb_hat) + self.epsilon)
            )

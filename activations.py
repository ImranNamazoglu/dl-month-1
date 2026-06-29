import numpy as np


def linear(x=None, prime=False):
    if prime:
        return linear_prime
    return x


def linear_prime(x):
    return np.ones_like(x, dtype=float)


def sigmoid(x=None, prime=False):
    if prime:
        return sigmoid_prime
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1.0 - y)


def tanh(x=None, prime=False):
    if prime:
        return tanh_prime
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - tanh(x) ** 2


def relu(x=None, prime=False):
    if prime:
        return relu_prime
    return np.maximum(0, x)


def relu_prime(x):
    return (x > 0).astype(float)


def leaky_relu(x=None, negative_slope=0.01, prime=False):
    if prime:
        return lambda values: leaky_relu_prime(values, negative_slope)
    return np.where(x > 0, x, negative_slope * x)


def leaky_relu_prime(x, negative_slope=0.01):
    return np.where(x > 0, 1.0, negative_slope)


def softmax(x=None, axis=-1, prime=False):
    if prime:
        raise NotImplementedError(
            "softmax prime is not element-wise; use its full Jacobian or combine it with the loss derivative."
        )
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


ACTIVATIONS = {
    "linear": linear,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "softmax": softmax,
}


def get_activation(name, prime=False):
    try:
        activation = ACTIVATIONS[name]
    except KeyError:
        valid_names = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"Unknown activation {name!r}. Choose one of: {valid_names}.")
    if prime:
        return activation(prime=True)
    return activation

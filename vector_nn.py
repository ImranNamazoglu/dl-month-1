import numpy as np
from activations import get_activation

rng = np.random.default_rng()

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = rng.normal(0.0, np.sqrt(2.0 / input_size), (input_size, output_size))
        self.biases = np.zeros(output_size)
        self.wgrad = np.zeros_like(self.weights)
        self.bgrad = np.zeros_like(self.biases)
        self.dL_dz = None
        self.dL_dy = None
        self.activation = activation
        self.activation_function = get_activation(activation)

    def __call__(self, inputs):
        z = inputs @ self.weights + self.biases
        return self.activation_function(z), z

    def __repr__(self):
        return f"Layer: weights shape {self.weights.shape}, biases shape {self.biases.shape}, activation {self.activation!r}"

class MLP:
    def __init__(self, widths, activations):
        if len(activations) != len(widths) - 1:
            raise ValueError("There must be one activation per layer.")
        self.layers = [Layer(widths[i], widths[i + 1], activations[i]) for i in range(len(widths) - 1)]

    def __call__(self, inputs):
        # the ys and zs start from the first layer
        zs = []
        ys = []
        activated_output = inputs
        for layer in self.layers:
            activated_output, raw_output = layer(activated_output)
            zs.append(raw_output)
            ys.append(activated_output)
        # returns the output of the mlp, & the zs + ys calculated during forward pass, for each layer
        return activated_output, zs, ys

    
    def __repr__(self):
        return f"MLP: {len(self.layers)} layers\n" + "\n".join(str(layer) for layer in self.layers)

    def _backward(self, inputs, targets):
        # forward pass
        z = []
        y = []
        
        outputs, z, y = self(inputs)
        
        z = list(reversed(z))
        y = list(reversed(y)); y.append(inputs)

        loss = np.mean((outputs - targets) ** 2)
        # backward pass
        reversed_layers = list(reversed(self.layers))
        for i, layer in enumerate(reversed_layers):
            dy_dz = get_activation(layer.activation, prime=True)(z[i])
            dz_dy_prev = reversed_layers[i - 1].weights if i != 0 else None

            layer.dL_dy = reversed_layers[i - 1].dL_dz @ dz_dy_prev.T if i != 0 else 2 * (y[0] - targets) / y[0].size
            layer.dL_dz = layer.dL_dy * dy_dz

            layer.wgrad = y[i + 1].T @ layer.dL_dz
            layer.bgrad = np.sum(layer.dL_dz, axis=0)
        
        return loss

    def fit(self, inputs, targets, learning_rate, epochs):
        losses = []
        for _ in range(epochs):
            current_loss = self._backward(inputs, targets)
            losses.append(current_loss)
            for layer in self.layers:
                layer.weights -= layer.wgrad * learning_rate
                layer.biases -= layer.bgrad * learning_rate
        
        return losses

if __name__ == "__main__":
    from test_dataset import california_housing_market_vector_dataset, california_housing_vector_test_value
    mlp = MLP([9, 16, 16, 1], ["relu", "relu", "linear"])
    inputs, targets = california_housing_market_vector_dataset()
    test_input, test_target = california_housing_vector_test_value()

    learning_rate = 0.05
    epochs = int(10e2)
    step = int(epochs/10)
    losses = mlp.fit(inputs, targets, learning_rate, epochs)[::step]
    for i in losses:
        print(i)
        print("-" * 40)

    prediction, _, _ = mlp(test_input)

    print(prediction, test_target)
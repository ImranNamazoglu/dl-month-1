import numpy as np
from activations import get_activation

rng = np.random.default_rng()

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = rng.normal(0.0, 1.0, (input_size, output_size))
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
        z = []
        y = []
        activated_output = inputs
        for layer in self.layers:
            activated_output, raw_output = layer(activated_output)
            z.append(raw_output)
            y.append(activated_output)
        # returns the output of the mlp, & the zs + ys calculated during forward pass, for each layer
        return activated_output, z, y

    
    def __repr__(self):
        return f"MLP: {len(self.layers)} layers\n" + "\n".join(str(layer) for layer in self.layers)

    def _backward(self, inputs, targets):
        # forward pass
        z = [];     y = []
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

if __name__ == "__main__":
    # mlp = MLP([3, 5, 2], ["relu", "sigmoid"])
    # x = np.array([[0.1, 0.2, 0.3], [0.4, 0.32, 0.13], [0.1, 0.212, 0.3] ,[0.11, 0.2, 0.3], [0.1, 0.2, 0.3]])
    # output = mlp(x)




    # print(mlp)
    # print("-" * 40)
    # print("Output of MLP: ", output)
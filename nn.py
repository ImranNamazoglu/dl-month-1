from engine import Value
import test_dataset
import random

class Perceptron:
    def __init__(self, nin, activation):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(0.5)
        self.activation = activation

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError(f"Expected {len(self.w)} inputs, got {len(x)}.")
        wsum = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "relu":
            return wsum.relu()
        elif self.activation == "sigmoid":
            return wsum.sigmoid()
        elif self.activation == "tanh":
            return wsum.tanh()
        else:
            return wsum

    def parameters(self):
        return self.w + [self.b]

    def length(self):
        return len(self.parameters()) - 1
    
class Layer:                            
    def __init__(self, nin, nout, activation):
        self.activation = activation
        self.neurons = [Perceptron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x):
        output = [n(x) for n in self.neurons]
        return output
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self, widths, activations):
        if len(activations) != len(widths) - 1:
            raise ValueError("There must be one activation per layer.")
        self.layers = [Layer(widths[i], widths[i + 1], activations[i]) for i in range(len(widths) - 1)]
    
    def __call__(self, x):
        current_output = x
        for l in self.layers:
            output = l(current_output)
            current_output = output
        return current_output
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def nullify(self):
        for p in self.parameters():
            p.gradient = 0.0
    
    def fit(self, xs, ys, alpha, epoch):
        ys = Value.list_to_value(ys)
        for _ in range(epoch):
            self.nullify()
            cost = Value(0.0)
            for i, training_data in enumerate(xs):
                output = self(training_data)
                loss = (output[0] - ys[i]) ** 2
                cost += loss
            cost = cost / len(xs)
            cost._backwards()
            for p in self.parameters():
                p.data -= alpha * p.gradient
            print(f"Cost: {cost.data}")
    

if __name__ == "__main__":
    test_x, expected_y = test_dataset.california_housing_test_value()
    xs, ys = test_dataset.california_housing_market_dataset()
    mlp = MLP([9, 16, 8, 1], ["relu", "relu", "linear"])
    mlp.fit(xs, ys, 0.03, 250)
    prediction = mlp(test_x)[0].data
    print(prediction, expected_y)
    print(test_dataset.denormalize_california_housing_price(prediction))

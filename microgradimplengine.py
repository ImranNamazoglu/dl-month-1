import math
import sys
import random
import numpy as np


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.gradient = 0.0
    

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.gradient += 1 * out.gradient
            other.gradient += 1 * out.gradient
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Power must be an int or float.")
        out = Value((self.data ** other), (self, ), "**")
        def _backward():
            self.gradient += (other) * (self.data ** (other - 1)) * out.gradient
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (other * Value(-1.0))
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1
    
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def tanh(self):
        intermediate = (math.exp(self.data * 2) - 1) / (math.exp(self.data * 2) + 1)
        out = Value(intermediate, (self, ), "tanh")
        def _backward():
            self.gradient += (1 - intermediate**2) * out.gradient
        out._backward = _backward
        return out
    
    def sigmoid(self):
        intermediate = 1.0 / (1.0 + math.exp(self.data * -1.0))
        out = Value(intermediate, (self, ), "sigmoid")
        def _backward():
            self.gradient += (intermediate) * (1.0 - intermediate) * out.gradient
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(max(0.0, self.data), (self, ), "relu")
        def _backward():
            if out.data > 0:
                self.gradient += 1.0 * out.gradient
            else:
                self.gradient += 0.0 * out.gradient
        out._backward = _backward
        return out
    
    def _backwards(self):
        visited = set()
        topo = []
        def build_topo(v):
            assert isinstance(v, Value)

            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.gradient = 1.0
        for n in reversed(topo):
            n._backward()

    @staticmethod
    def list_to_value(lst):
        return [Value(x) for x in lst]
    
    def __repr__(self):
        return f"Value: {self.data}"
        

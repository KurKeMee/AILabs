import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        self.hiddenSizes = hiddenSizes
        self.n_layers = len(hiddenSizes)
        
        self.weights = []
        
        Win = np.zeros((1 + inputSize, hiddenSizes[0]))
        Win[0, :] = np.random.randint(0, 3, size=(hiddenSizes[0]))
        Win[1:, :] = np.random.randint(-1, 2, size=(inputSize, hiddenSizes[0]))
        self.weights.append(Win.astype(np.float64))
        
        for i in range(len(hiddenSizes) - 1):
            W = np.zeros((1 + hiddenSizes[i], hiddenSizes[i + 1]))
            W[0, :] = np.random.randint(0, 3, size=(hiddenSizes[i + 1]))
            W[1:, :] = np.random.randint(-1, 2, size=(hiddenSizes[i], hiddenSizes[i + 1]))
            self.weights.append(W.astype(np.float64))
        
        Wout = np.random.randint(0, 2, size=(1 + hiddenSizes[-1], outputSize)).astype(np.float64)
        self.weights.append(Wout)
        
    def predict(self, Xp):
        hidden_outputs = []
        current_input = Xp
        
        for i in range(self.n_layers):
            W = self.weights[i]
            hidden = np.where((np.dot(current_input, W[1:, :]) + W[0, :]) >= 0.0, 1, -1).astype(np.float64)
            hidden_outputs.append(hidden)
            current_input = hidden
        
        Wout = self.weights[-1]
        out = np.where((np.dot(current_input, Wout[1:, :]) + Wout[0, :]) >= 0.0, 1, -1).astype(np.float64)
        
        return out, hidden_outputs

    def train(self, X, y, n_iter=5, eta=0.01):
        for epoch in range(n_iter):
            print(self.weights[-1].reshape(1, -1))
            for xi, target in zip(X, y):
                output, hiddens = self.predict(xi)
                error = target - output
                
                self.weights[-1][1:] += eta * error * hiddens[-1].reshape(-1, 1)
                self.weights[-1][0] += eta * error
                
                if len(self.weights) > 1:
                    hidden_error = error * self.weights[-1][1:].T
                    if len(self.weights) == 2:
                        self.weights[0][1:] += eta * np.outer(xi, hidden_error)
                        self.weights[0][0] += eta * hidden_error.flatten()
        return self
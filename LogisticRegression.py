import numpy as np

class SoftmaxRegression:
    def __init__(self, lr=0.1, epochs=300):
        self.lr = lr
        self.epochs = epochs

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.k = len(np.unique(y))  # number of classes (10)

        self.W = np.zeros((self.n, self.k))
        self.b = np.zeros(self.k)

        y_onehot = self.one_hot(y, self.k)

        for _ in range(self.epochs):
            z = np.dot(X, self.W) + self.b
            y_hat = self.softmax(z)

            dW = (1/self.m) * np.dot(X.T, (y_hat - y_onehot))
            db = (1/self.m) * np.sum(y_hat - y_onehot, axis=0)

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_hat = self.softmax(z)
        return np.argmax(y_hat, axis=1)
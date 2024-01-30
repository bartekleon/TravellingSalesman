
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

class Network:
    def __init__(self, eta, layers, max_i=10000, max_error=0.00001):
      self._eta = eta
      self._layers = layers
      self._w = []
      self._y = []
      self._delta = []
      self.max_i = max_i
      self.max_error = max_error

    def relu(self, x, derivative=False):
      # if derivative:
      #   return x * (1 - x)
      # return 1 / (1 + np.exp(-x))
      if derivative:
        return np.where(x > 0, 1, 0)
      return np.maximum(0, x)

    def _create_weights(self, x_num):
      for i, layer_i in enumerate(self._layers):
        if i == 0:
          self._w.append(np.random.uniform(-1, 1, size=(layer_i, x_num + 1)))
        else:
          self._w.append(np.random.uniform(-1, 1, size=(layer_i, self._w[i - 1].shape[0] + 1)))
        self._y.append(np.zeros(layer_i))
        self._delta.append(np.zeros(layer_i))

    def _forward_progation(self, x):
      for i in range(len(self._layers)):
        if i == 0:
          self._y[i] = np.dot(self._w[i], np.append(np.array(1), x))
        else:
          self._y[i] = np.dot(self._w[i], np.append(np.array(1), self._y[i - 1]))
        self._y[i] = self.relu(self._y[i])

    def _backward_propagation(self, expected_output):
      for i in range(len(self._y) - 1, -1, -1):
        if i == len(self._y) - 1:
          error = expected_output - self._y[i]
          self._delta[i] = error * self.relu(self._y[i], True)
        else:
          self._delta[i] = np.dot(self._w[i+1][:, 1:].T, self._delta[i+1]) * self.relu(self._y[i], True)
      return error

    def _update(self, x):
        for i in range(len(self._layers)):
            for j in range(len(self._delta[i])):
                if i == 0:
                    self._w[i][j] += np.append([1], x) * \
                        self._delta[i][j] * self._eta
                else:
                    self._w[i][j] += (
                        np.append([1], self._y[i - 1]) *
                        self._delta[i][j] * self._eta
                    )

    def fit(self, x, y, verbose=False):
        self._create_weights(np.shape(x)[1])
        data = list(zip(x, y))
        for epoch in range(self.max_i):
            np.random.shuffle(data)
            E = 0
            for x_i, y_i in data:
                self._forward_progation(x_i)
                b = self._backward_propagation(y_i)
                b = b**2
                E += 0.5 * b.sum()
                self._update(x_i)
            if E < self.max_error:
                break
            if verbose:
                print(f"Epoch: {epoch+1} of {self.max_i}, error: {E}")
        return E

    def predict(self, x):
        self._forward_progation(x)
        return self._y[-1]



def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

    
if __name__=='__main__':
    # Load Iris dataset
    iris = load_iris()
    X = iris.data # features, 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
    y = iris.target # label, 'setosa' 'versicolor' 'virginica'

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=42) # split training data and testing data

    train_y_encoded = one_hot_encode(train_y, 3)
    test_y_encoded = one_hot_encode(test_y, 3)

    network = Network(0.2, (12, 24, 3), 200, 0.01)
    network.fit(x_train, train_y_encoded, True)
    result = []
    result_pred = []
    for i in range(len(x_test)):
      y_pred = network.predict(x_test[i])
      y_result = np.zeros(3)
      y_result[np.argmax(y_pred)] = 1
      result_pred.append(y_result)

    print(network._layers)
    print(f"Accuracy: {accuracy_score(test_y_encoded, result_pred)}")
    print(f"roc_auc_score: {roc_auc_score(test_y_encoded, result_pred)}")

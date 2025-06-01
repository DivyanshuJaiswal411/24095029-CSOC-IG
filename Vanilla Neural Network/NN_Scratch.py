import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc

# Load and preprocess data
df = pd.read_csv(r"C:\Users\Divyanshu\Downloads\KaggleV2-May-2016.csv\KaggleV2-May-2016.csv")
X = df.iloc[:, [5, 7, 8, 9, 10, 11, 12]].values.astype(np.float32)
y = (df.iloc[:, 13].values == "Yes").astype(np.float32).reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Layers and activations
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.output = np.clip(self.output, 1e-7, 1 - 1e-7)
        self.dinputs = dvalues * (self.output * (1 - self.output))

# Loss function
class Loss_BinaryCrossEntropy:
    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.mean(self.pos_weight * y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(self.pos_weight * y_true / dvalues - (1 - y_true) / (1 - dvalues)) / samples


# Optimizers
class Optimizer_SGD:
    def __init__(self, learning_rate=0.1, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_momentums = {}
        self.bias_momentums = {}

    def update_params(self, layer, layer_id):
        if self.momentum:
            if layer_id not in self.weight_momentums:
                self.weight_momentums[layer_id] = np.zeros_like(layer.weights)
                self.bias_momentums[layer_id] = np.zeros_like(layer.biases)
            self.weight_momentums[layer_id] = self.momentum * self.weight_momentums[layer_id] - self.learning_rate * layer.dweights
            self.bias_momentums[layer_id] = self.momentum * self.bias_momentums[layer_id] - self.learning_rate * layer.dbiases
            layer.weights += self.weight_momentums[layer_id]
            layer.biases += self.bias_momentums[layer_id]
        else:
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases

class Optimizer_Adam:
    def __init__(self, learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0
        self.m_weights = {}
        self.v_weights = {}
        self.m_biases = {}
        self.v_biases = {}

    def update_params(self, layer, layer_id):
        if layer_id not in self.m_weights:
            self.m_weights[layer_id] = np.zeros_like(layer.weights)
            self.v_weights[layer_id] = np.zeros_like(layer.weights)
            self.m_biases[layer_id] = np.zeros_like(layer.biases)
            self.v_biases[layer_id] = np.zeros_like(layer.biases)

        self.iterations += 1

        self.m_weights[layer_id] = self.beta_1 * self.m_weights[layer_id] + (1 - self.beta_1) * layer.dweights
        self.v_weights[layer_id] = self.beta_2 * self.v_weights[layer_id] + (1 - self.beta_2) * (layer.dweights ** 2)
        self.m_biases[layer_id] = self.beta_1 * self.m_biases[layer_id] + (1 - self.beta_1) * layer.dbiases
        self.v_biases[layer_id] = self.beta_2 * self.v_biases[layer_id] + (1 - self.beta_2) * (layer.dbiases ** 2)

        m_hat_w = self.m_weights[layer_id] / (1 - self.beta_1 ** self.iterations)
        v_hat_w = self.v_weights[layer_id] / (1 - self.beta_2 ** self.iterations)
        m_hat_b = self.m_biases[layer_id] / (1 - self.beta_1 ** self.iterations)
        v_hat_b = self.v_biases[layer_id] / (1 - self.beta_2 ** self.iterations)

        layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

# Neural Network with 2 hidden layers
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, optimizer):
        self.layer1 = Layer_Dense(input_size, hidden_size)
        self.activation1 = Activation_ReLU()
        self.layer2 = Layer_Dense(hidden_size, hidden_size)
        self.activation2 = Activation_ReLU()
        self.layer3 = Layer_Dense(hidden_size, 1)
        self.activation3 = Activation_Sigmoid()
        self.loss_function = Loss_BinaryCrossEntropy()
        self.optimizer = optimizer

    def train(self, X, y, epochs=2000, verbose=True):
        losses = []
        accuracies = []
        for epoch in range(epochs):
            # Forward pass
            self.layer1.forward(X)
            self.activation1.forward(self.layer1.output)

            self.layer2.forward(self.activation1.output)
            self.activation2.forward(self.layer2.output)

            self.layer3.forward(self.activation2.output)
            self.activation3.forward(self.layer3.output)

            # Loss and accuracy
            loss = self.loss_function.forward(self.activation3.output, y)
            predictions = (self.activation3.output > 0.5).astype(int)
            accuracy = np.mean(predictions == y)

            losses.append(loss)
            accuracies.append(accuracy)

            # Backward pass
            self.loss_function.backward(self.activation3.output, y)
            self.activation3.backward(self.loss_function.dinputs)
            self.layer3.backward(self.activation3.dinputs)

            self.activation2.backward(self.layer3.dinputs)
            self.layer2.backward(self.activation2.dinputs)

            self.activation1.backward(self.layer2.dinputs)
            self.layer1.backward(self.activation1.dinputs)

            # Update parameters
            self.optimizer.update_params(self.layer3, 'layer3')
            self.optimizer.update_params(self.layer2, 'layer2')
            self.optimizer.update_params(self.layer1, 'layer1')

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X):
        self.layer1.forward(X)
        self.activation1.forward(self.layer1.output)

        self.layer2.forward(self.activation1.output)
        self.activation2.forward(self.layer2.output)

        self.layer3.forward(self.activation2.output)
        self.activation3.forward(self.layer3.output)

        return (self.activation3.output > 0.20).astype(int)


# Choose optimizer: Adam with smaller learning rate
optimizer_adam = Optimizer_Adam(learning_rate=0.0005)
model = SimpleNeuralNetwork(input_size=7, hidden_size=50, optimizer=optimizer_adam)

# Train
model.train(X_train, y_train, epochs=1000)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No-show", "Show"], yticklabels=["No-show", "Show"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Get raw sigmoid outputs (not thresholded)
model.layer1.forward(X_test)
model.activation1.forward(model.layer1.output)
model.layer2.forward(model.activation1.output)
model.activation2.forward(model.layer2.output)
model.layer3.forward(model.activation2.output)
model.activation3.forward(model.layer3.output)
y_pred_probs = model.activation3.output

# Compute PR-AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
pr_auc = auc(recall, precision)
print(f"PR-AUC (NumPy): {pr_auc:.4f}")

# Plot PR Curve
plt.figure()
plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (NumPy Model)')
plt.legend()
plt.grid(True)
# plt.savefig("pr_curve_numpy.png")  
plt.show()


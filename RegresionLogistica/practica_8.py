#Pablo Hernandez Jesus Maldonado
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de costo 
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    costo = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    return costo

# Gradiente descendente
def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    costos = []

    for _ in range(num_iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * X.T @ (h - y)
        theta -= alpha * gradient
        costo = compute_cost(X, y, theta)
        costos.append(costo)

    return theta, costos

# Preprocesamiento de datos (one-hot)
data = pd.read_csv("Employee.csv")
one_hot = pd.DataFrame()

educacion = set(data["Education"])
ciudad = set(data["City"])
genero = set(data["Gender"])
espera = set(data["EverBenched"])


for category in educacion:
    one_hot[f"Education_{category}"] = (data["Education"] == category).astype(int)

for category in ciudad:
    one_hot[f"City_{category}"] = (data["City"] == category).astype(int)

for category in genero:
    one_hot[f"Gender_{category}"] = (data["Gender"] == category).astype(int)

for category in espera:
    one_hot[f"EverBenched_{category}"] = (data["EverBenched"] == category).astype(int)

columnas = ["Education", "City", "Gender", "EverBenched"]
data = data.drop(columnas, axis=1)
data = pd.concat([data, one_hot], axis=1)

# Separación de datos de entrenamiento y prueba
train_set = data.sample(frac=0.8, random_state=42)
test_set = data.drop(train_set.index)

X_train = train_set.drop("LeaveOrNot", axis=1)
Y_train = train_set["LeaveOrNot"]
X_test = test_set.drop("LeaveOrNot", axis=1)
Y_test = test_set["LeaveOrNot"]

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

theta = np.zeros(X_train.shape[1])
alpha = 0.001
iteraciones = 160



theta, costos = gradient_descent(X_train, Y_train, theta, alpha, iteraciones)
predi_Y = sigmoid(X_test @ theta)

#Metricas de validacion
Y_pred = (predi_Y >= 0.5).astype(int)
threshold = 0.5
predicted_labels = (predi_Y > threshold).astype(int)
true_positive = sum((predicted_labels == 1) & (Y_test == 1))
false_positive = sum((predicted_labels == 1) & (Y_test == 0))
true_negative = sum((predicted_labels == 0) & (Y_test == 0))
false_negative = sum((predicted_labels == 0) & (Y_test == 1))
accuracy = accuracy_score(Y_test, Y_pred)
print("Matriz de Confusión:")
print("Verdaderos Positivos:", true_positive)
print("Falsos Positivos:", false_positive)
print("Verdaderos Negativos:", true_negative)
print("Falsos Negativos:", false_negative)
print("Precisión :", accuracy)

#Grafica
plt.scatter(range(len(predi_Y)), predi_Y, c=Y_test, cmap='coolwarm')
plt.xlabel("Índice de la Muestra")
plt.ylabel("Probabilidad Predicha")
plt.title("Probabilidades Predichas vs. Índice de la Muestra")
plt.show()

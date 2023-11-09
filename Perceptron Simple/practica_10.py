import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def perceptron(X, Y, lr=0.1, iteraciones=1000):
    np.random.seed(0)
    w = np.random.rand(3)
    b = 1

    for iteracion in range(iteraciones):
        for i in range(len(X)):
            y_pred = b * w[0] + X[i, 0] * w[1] + X[i, 1] * w[2]
            y_pred = sigmoide(y_pred)
            error = Y[i] - y_pred
            dw = lr * error * np.array([b, X[i, 0], X[i, 1]])
            w += dw

    resultado = []
    for i in range(len(X)):
        y_pred = b * w[0] + X[i, 0] * w[1] + X[i, 1] * w[2]
        y_pred = sigmoide(y_pred)
        resultado.append((X[i], Y[i], y_pred))

    return w, resultado



X_OR = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Y_OR = np.array([0, 1, 1, 1])
w_OR, resultado_OR = perceptron(X_OR, Y_OR)

print('Compuerta logica OR')
for result in resultado_OR:
    x, y, y_pred = result
    print(f"{x}\t{y}\t -> {y_pred}")
print("Pesos finales de w:", w_OR)



X_AND = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Y_AND = np.array([0, 0, 0, 1])
w_AND, resultado_AND = perceptron(X_AND, Y_AND)

print('Compuerta logica AND')
for result in resultado_AND:
    x, y, y_pred = result
    print(f"{x}\t{y}\t -> {y_pred}")

print("Pesos finales de w:", w_AND)

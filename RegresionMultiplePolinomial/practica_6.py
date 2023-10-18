#Pablo Jesus Hernandez Rodriguez
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Fish.csv')
data = data.sample(frac=1, random_state=42)

total_rows = len(data)
train_rows = int(0.8 * total_rows)

train_data = data.iloc[:train_rows]
test_data = data.iloc[train_rows:]

X1_train = train_data.sort_values('Height')['Height']
X1_test = test_data.sort_values('Height')['Height']

X2_train = train_data.sort_values('Width')['Width']
X2_test = test_data.sort_values('Width')['Width']

Y_train = train_data.sort_values('Weight')['Weight']
Y_test = test_data.sort_values('Weight')['Weight']
X1_train = X1_train.reset_index(drop=True)
X2_train = X2_train.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)
learning_rate = 0.00001
num_iteraciones = 100

def regresion_polinomica_descenso_gradiente(X1, X2, Y, learning_rate, num_iteraciones):
    n = len(X1)
    b0, b1, b2, b3, b4, b5 = 0, 0, 0, 0, 0, 0 
    
    for _ in range(num_iteraciones):
        Y_pred = [b0 + b1 * X1[i] + b2 * X2[i] + b3 * X1[i]**2 + b4 * X2[i]**2 + b5 * X1[i] * X2[i] for i in range(n)]
        
        b0 -= (learning_rate / n) * sum(Y_pred[i] - Y[i] for i in range(n))
        b1 -= (learning_rate / n) * sum((Y_pred[i] - Y[i]) * X1[i] for i in range(n))
        b2 -= (learning_rate / n) * sum((Y_pred[i] - Y[i]) * X2[i] for i in range(n))
        b3 -= (learning_rate / n) * sum((Y_pred[i] - Y[i]) * X1[i]**2 for i in range(n))
        b4 -= (learning_rate / n) * sum((Y_pred[i] - Y[i]) * X2[i]**2 for i in range(n))
        b5 -= (learning_rate / n) * sum((Y_pred[i] - Y[i]) * X1[i] * X2[i] for i in range(n))
    
    return b0, b1, b2, b3, b4, b5


def predecir(b0,b1,b2,b3,b4,b5, X1, X2):
    Y_pred = [b0 + b1*x1 + b2*x2 + b3*x1**2 + b4*x2**2 + b5*x1*x2 for x1, x2 in zip(X1, X2)]
    return Y_pred


def calcular_r2(Y, Y_pred):
    mean_Y = sum(Y) / len(Y)
    ss_total = sum((y - mean_Y)**2 for y in Y)
    ss_residual = sum((y - y_pred)**2 for y, y_pred in zip(Y, Y_pred))
    r2 = 1 - (ss_residual / ss_total)
    return r2


def calcular_mse(Y, Y_pred):
    n = len(Y)
    mse = sum((y - y_pred)**2 for y, y_pred in zip(Y, Y_pred)) / n
    return mse


b0, b1, b2, b3, b4, b5 = regresion_polinomica_descenso_gradiente(X1_train, X2_train, Y_train, learning_rate, num_iteraciones)

X1_test = np.linspace(min(X1_test),max(X1_test),40)
X2_test = np.linspace(min(X2_test),max(X2_test),40)

Y_pred = predecir(b0,b1,b2,b3,b4,b5, X1_test, X2_test)
r2 = calcular_r2(Y_test, Y_pred)
mse = calcular_mse(Y_test, Y_pred)


print(f"Coeficiente b0: {b0}")
print(f"Coeficiente b1: {b1}")
print(f"Coeficiente b2: {b2}")
print(f"Coeficiente b3: {b3}")
print(f"Coeficiente b4: {b4}")
print(f"Coeficiente b5: {b5}")
print(f"Coeficiente de determinación (R^2): {r2}")
print(f"Error cuadrático medio (MSE): {mse}")


plt.figure(figsize=(10, 6))
plt.plot(range(len(Y_pred)),Y_pred, label='Prediccion' ,color="red")
plt.scatter(range(len(Y_test)),Y_test, label='Datos reales' ,color="blue")
plt.xlabel('Iteraciones')
plt.ylabel('Peso')
plt.legend()
plt.show()

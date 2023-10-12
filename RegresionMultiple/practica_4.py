import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

datos_prueba = pd.read_csv('Student_Performance.csv')
datos_prueba = datos_prueba.sample(n=100, random_state=1)

X1 = datos_prueba['Hours Studied'].values
X2 = datos_prueba['Previous Scores'].values
X3 = np.array([1 if item == 'Yes' else 0 for item in datos_prueba['Extracurricular Activities']])

X4 = datos_prueba['Sleep Hours'].values
X5 = datos_prueba['Sample Question Papers Practiced'].values
Y = datos_prueba['Performance Index'].values

sum_x1 = sum(X1) / len(X1)
sum_x2 = sum(X2) / len(X2)
sum_x3 = sum(X3) / len(X3)
sum_x4 = sum(X4) / len(X4)
sum_x5 = sum(X5) / len(X5)
sum_Y = sum(Y) / len(Y)

def minimos_cuadrados(X, Y, sum_X, sum_Y):
    numerator = 0
    denominator = 0
    for i in range(len(X)):
        numerator += (X[i] - sum_X) * (Y[i] - sum_Y)
        denominator += (X[i] - sum_X) ** 2
    return numerator / denominator

q1 = minimos_cuadrados(X1, Y, sum_x1, sum_Y)
q2 = minimos_cuadrados(X2, Y, sum_x2, sum_Y)
q3 = minimos_cuadrados(X3, Y, sum_x3, sum_Y)
q4 = minimos_cuadrados(X4, Y, sum_x4, sum_Y)
q5 = minimos_cuadrados(X5, Y, sum_x5, sum_Y)

q0 = sum_Y - q1 * sum_x1 - q2 * sum_x2 - q3 * sum_x3 - q4 * sum_x4 - q5 * sum_x5

def RegresionLinealMultiple(x1, x2, x3, x4, x5):
    return q0 + q1 * x1 + q2 * x2 + q3 * x3 + q4 * x4 + q5 * x5

prediccion_Y = RegresionLinealMultiple(X1, X2, X3, X4, X5)


mse = mean_squared_error(Y, prediccion_Y)
print(f"MSE -> {mse}")


r2 = r2_score(Y, prediccion_Y)
print(f"R2 -> {r2}")

plt.scatter(Y, prediccion_Y, label="Población", color="blue")
plt.xlabel("Índice de Rendimiento Real")
plt.ylabel("Índice de Rendimiento Predicción")
plt.title("Índice de Rendimiento Real vs. Predicción de la Población")
plt.legend()
plt.show()
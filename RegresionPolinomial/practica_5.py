import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datos_prueba = pd.read_csv('polynomial-regression.csv')
araba_fiyat = datos_prueba['araba_fiyat'].values
araba_max_hiz = datos_prueba['araba_max_hiz'].values
mse = 0
sst = 0
ssr = 0


n = len(araba_fiyat)
sum_x = sum(araba_fiyat)
sum_y = sum(araba_max_hiz)
sum_x2 = sum(x**2 for x in araba_fiyat)
sum_x4 = sum(x**4 for x in araba_fiyat)
sum_xy = sum(x * y for x, y in zip(araba_fiyat, araba_max_hiz))
sum_x2y = sum(x**2 * y for x, y in zip(araba_fiyat, araba_max_hiz))

w2 = (sum_x2y - sum_x * sum_y) / (sum_x4 - sum_x2 * sum_x2)
w1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
w0 = (sum_y - w2 * sum_x2 - w1 * sum_x) / n




def rPolinomial(x):
    return w2 * x**2 + w1 * x + w0
prediccion_y = [rPolinomial(x) for x in range(50,3001)]


for i in range(len(araba_max_hiz)):
    mse += (araba_max_hiz[i] - prediccion_y[i]) ** 2
mse = mse / len(araba_max_hiz)


media_y = sum(araba_max_hiz) / len(araba_max_hiz)
for i in range(len(araba_max_hiz)):
    sst += (araba_max_hiz[i] - media_y) ** 2
    ssr += (araba_max_hiz[i] - prediccion_y[i]) ** 2
r2 = 1 - (ssr / sst)


print('Valor de w2 ->', w2)
print('Valor de w1 ->', w1)
print('Valor de w0 ->', w0)
print("MSE es igual a ->", mse)
print("R2 es igual a ->", r2)


plt.scatter(araba_fiyat, araba_max_hiz, label="Datos reales")
plt.plot(range(50,3001), prediccion_y, color='red', label="Regresión Polinomial (grado 2)")
plt.xlabel("Araba Fiyat")
plt.ylabel("Araba Max Hiz")
plt.legend()
plt.grid(True)
plt.show()

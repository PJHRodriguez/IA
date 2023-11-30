import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
def regresionLinealMultiple(archivo,dataset,colorgrafica,posicion):
    if(archivo!= "Employee.csv"):
        df = pd.read_csv(archivo)
        if archivo == "zoo.csv":
            X = df.iloc[:, 1:-1]
        else:
            X = df.iloc[:, 0:-1]
        
        Y = df.iloc[:, -1]
        Y = Y.to_numpy()
        X = X.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    else:
        df = pd.read_csv(archivo)
        df_encoded = pd.get_dummies(df, columns=['Education', 'City', 'Gender', 'EverBenched'], drop_first=True)
        column_order = df_encoded.columns.tolist()
        column_order = [col for col in column_order if col not in ['LeaveOrNot']] + ['LeaveOrNot']
        df = df_encoded[column_order]
        
        X = df.iloc[:, 0:-1]  
        Y = df.iloc[:, -1]
        Y = Y.to_numpy()
        X = X.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    
    sum_X = np.zeros(len(X_train[0]))
    
    for i in range(len(X_train)):
        for j in range(len(X_train[0])):
            sum_X[j] += X_train[i][j]

    sum_X = sum_X / len(X_train)
    sum_Y = sum(Y_train) / len(Y_train)


    q = np.zeros(len(X[0]))

    def minimos_cuadrados(X, Y, sum_X, sum_Y):
        numerador = 0
        denominador = 0
        for i in range(len(X)):
            numerador += (X[i] - sum_X) * (Y[i] - sum_Y)
            denominador += (X[i] - sum_X) ** 2

        if np.any(denominador != 0):
            return numerador / denominador
        else:
            return 0

    for i in range(len(q)):
        q[i] = minimos_cuadrados(X_train[:, i], Y_train, sum_X[i], sum_Y)

    q0 = sum_Y - sum([sum_X[i] * q[i] for i in range(len(q))])

    def RegresionLinealMultiple(X):
        return q0 + sum([q[i] * X[:,i] for i in range(len(q))])

    prediccion_Y = RegresionLinealMultiple(X_test)
    mse = mean_squared_error(Y_test,prediccion_Y)
    r2 = r2_score(Y_test, prediccion_Y)
    
    print(f"Metricas de evaluacion del dataset {dataset} con Regresion Simple Multiple:")
    print(f"MSE -> {mse}")
    print(f"R2 -> {r2}")
    print("\n")
    simbolo = r'$\hat{Y}$'

    plt.subplot(3,3,posicion)
    plt.scatter(Y_test, prediccion_Y,label= f"Y vs {simbolo}", color=colorgrafica)
    plt.xlabel("Real Y")
    plt.ylabel("Predicción Y")
    plt.title(f"Regreson Simple Multiple de {dataset}" )
    plt.legend()

def regresionLogistica(archivo,dataset,posicion,columna_y):
    
    # Función sigmoide
    def sigmoide(z):
        return 1 / (1 + np.exp(-z))

    # Función de costo 
    def f_costos(X, y, theta):
        m = len(y)
        h = sigmoide(X @ theta)
        costo = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
        return costo

    # Gradiente descendente
    def gradient_descent(X, y, theta, alpha, num_iterations):
        m = len(y)
        costos = []

        for _ in range(num_iterations):
            h = sigmoide(X @ theta)
            gradiente = (1/m) * X.T @ (h - y)
            theta -= alpha * gradiente
            costo = f_costos(X, y, theta)
            costos.append(costo)

        return theta, costos
    
    df = pd.read_csv(archivo)
    columnas_categorias = df.select_dtypes(include=['object']).columns

    if len(columnas_categorias) > 0:
        df = pd.get_dummies(df, columns=columnas_categorias, drop_first=True)

        column_order = df.columns.tolist()
        column_order = [col for col in column_order if col not in [columna_y]] + [columna_y]
        df = df[column_order]

    X = df.iloc[:, 1:-1]
    Y = df.iloc[:, -1]
    Y = Y.to_numpy()
    X = X.to_numpy()  
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  


    # Normalizar las características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    theta = np.zeros(X_train.shape[1])
    alpha = 0.001
    iteraciones = 160



    theta, costos = gradient_descent(X_train, Y_train, theta, alpha, iteraciones)
    predi_Y = sigmoide(X_test @ theta)

    #Metricas de validacion
    Y_pred = (predi_Y >= 0.5).astype(int)
    threshold = 0.5
    p = (predi_Y > threshold).astype(int)

    verdaderoPositivo = sum((p == 1) & (Y_test == 1))
    falsoPositivo = sum((p == 1) & (Y_test == 0))
    verdaderoNegativo = sum((p == 0) & (Y_test == 0))
    falsoNegativo = sum((p == 0) & (Y_test == 1))
    precision = accuracy_score(Y_test, Y_pred)

    print(f"Matriz de Confusión de {dataset}:")
    print("Verdaderos Positivos:", verdaderoPositivo)
    print("Falsos Positivos:", falsoPositivo)
    print("Verdaderos Negativos:", verdaderoNegativo)
    print("Falsos Negativos:", falsoNegativo)
    print("Precisión :", precision)
    print("\n")

    #Grafica
    plt.subplot(3,3,posicion)
    plt.scatter(range(len(predi_Y)), predi_Y, c=Y_test, cmap='coolwarm')
    plt.xlabel("Índice de la Muestra")
    plt.ylabel("Probabilidad Predicha")
    plt.title(f"Regresion logistica de {dataset}")

def perceptronSimple(archivo,dataset,posicion,colorgrafica):
    if archivo != "Employee.csv":
        df = pd.read_csv(archivo)
        if archivo == "zoo.csv":
            X = df.iloc[:, [1, 2]]
        else:
            X = df.iloc[:, [0, 1]]  

        Y = df.iloc[:, -1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    else:
        df = pd.read_csv(archivo)
        X = df.iloc[:, [4, 7]]
        Y = df.iloc[:, -1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    def sigmoide(x):
        return 1 / (1 + np.exp(-x))
    
    def perceptron(X, Y, lr=0.1, iteraciones=100):
        np.random.seed(0)
        w = np.random.rand(X.shape[1] + 1) 
        
        for iteracion in range(iteraciones):
            for i in range(len(X)):

                y_pred = w[0] + X.iloc[i, 0] * w[1] + X.iloc[i, 1] * w[2]
                y_pred = sigmoide(y_pred)
                error = Y.iloc[i] - y_pred
                dw = lr * error * np.array([1, X.iloc[i, 0], X.iloc[i, 1]])
                w += dw

        resultado = []
        prediccion_Y = []
        for i in range(len(X)):

            y_pred = w[0] + X.iloc[i, 0] * w[1] + X.iloc[i, 1] * w[2]
            y_pred = sigmoide(y_pred)
            resultado.append((X.iloc[i].values, Y.iloc[i], y_pred)) 
            prediccion_Y.append(y_pred)

        return w,prediccion_Y,resultado 
    
    w, prediccion,resultado = perceptron(X_test, Y_test)

    

    for result in resultado:
        x, y, y_pred = result
        print(f"{x}\t{y}\t -> {y_pred}")
    
    print(f"Pesos finales de w {w} en el dataset {dataset}")


    simbolo = r'$\hat{Y}$'
    plt.subplot(3,3,posicion)
    plt.scatter(Y_test, prediccion,label= f"Y vs {simbolo}", color=colorgrafica)
    plt.xlabel("Real Y")
    plt.ylabel("Predicción Y")
    plt.title(f"Perceptron Simple de {dataset}" )
    plt.legend()

    

regresionLinealMultiple('zoo.csv',"Zoologico","green",1)
regresionLinealMultiple('Employee.csv',"Empleados","blue",2)
regresionLinealMultiple('heart.csv',"Problemas Cardiacos","orange",3)

regresionLogistica('heart.csv','Problemas Cardiacos',4,"output")
regresionLogistica('zoo.csv','Zoologico',5,"class_type")
regresionLogistica('Employee.csv','Empleados',6,"LeaveOrNot")


perceptronSimple('Employee.csv',"Empleados",7,"blue")
perceptronSimple('zoo.csv',"Zoologico",8,"green")
perceptronSimple('heart.csv',"Problemas Cardiacos",9,"orange")

plt.tight_layout() 
plt.show()
#Pablo Hernandez
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("irisbin.csv", header=None)  

#Train-Test 
train1_set = dataset.sample(frac=0.8, random_state=42)
test1_set = dataset.drop(train1_set.index)


#Fixed
train_size = int(0.8 * len(dataset))
train2_set = dataset[:train_size]
test2_set = dataset[train_size:]

#Block
block_size = len(dataset) // 5
train3_sets = [dataset[i * block_size:(i + 1) * block_size] for i in range(4)]
test3_set = dataset[(4 * block_size):]

#Odd-Row
train4_set = dataset[::2]#Pares
test4_set = dataset[1::2]#Impares  

#Stratified
train5_set = dataset[dataset[0] < 6.0]
test5_set = dataset[dataset[0] >= 6.0]




plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.title('Particion Train-Test Split')
plt.scatter(train1_set[0],train1_set[4],color='blue', label='Datos de entrenamiento', marker='o')
plt.scatter(test1_set[0],test1_set[4], color='red', label='Datos de prueba', marker='o')
plt.xlabel('X1')
plt.ylabel('Y1')
plt.legend()

plt.subplot(2,3,2)
plt.title('Particion Fixed Split')
plt.scatter(train2_set[0],train2_set[4] ,color='blue', label='Datos de entrenamiento', marker='o')
plt.scatter(test2_set[0],test2_set[4], color='red', label='Datos de prueba', marker='o')
plt.xlabel('X1')
plt.ylabel('Y1')
plt.legend()

plt.subplot(2, 3, 3)
plt.title('Partición Block Split')
for train_set in train3_sets:
    plt.scatter(train_set[0], train_set[4], color='blue', label='Datos de entrenamiento', marker='o')
plt.scatter(test3_set[0], test3_set[4], color='red', label='Datos de prueba', marker='o')
plt.xlabel('X1')
plt.ylabel('Y1')
plt.legend()

plt.subplot(2, 3, 4)
plt.title('Partición Odd-Row Split')
plt.scatter(train4_set[0], train4_set[4], color='blue', label='Datos de entrenamiento', marker='o')
plt.scatter(test4_set[0], test4_set[4], color='red', label='Datos de prueba', marker='o')
plt.xlabel('X1')
plt.ylabel('Y1')
plt.legend() 

plt.subplot(2, 3, 5)
plt.title('Particion Stratified split')
plt.scatter(train5_set[0], train5_set[4], color='blue', label='Datos de entrenamiento', marker='o')
plt.scatter(test5_set[0], test5_set[4], color='red', label='Datos de prueba', marker='o')
plt.xlabel('X1')
plt.ylabel('Y1')
plt.legend()

plt.tight_layout()
plt.show()

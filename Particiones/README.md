En el codigo que se realizo tratamos de conocer algunas formas de particiones que podemos realizar con nuestros datos para crear el entrenamiento y las pruebas para evaluar algun modelo y validarlo.<br>
Muestreo Aleatorio (Train-Test Split)<br>
Este método divide los datos en dos conjuntos uno para entrenamiento y otro para pruebas seleccionando aleatoriamente datos para cada conjunto.<br>
Porción Fija (Fixed Split)<br>
En este metodo se asigna un porcentaje fijo de los datos al conjunto de entrenamiento y el restante al conjunto de prueba lo ya sea el primer 80 por ciento de los primeros datos como entrenmiento y el resto para prueba<br>
Partición en Bloques (Block Split)<br>
Se dividen los datos en una cantidad de bloques de igual tamaño utilizando la mayoría de los bloques como datos de entrenamiento y uno como datos de prueba. Este método es útil cuando se necesita mantener una estructura de bloques en los datos.<br>
Partición por Posición (Position-Based Split)<br>
Este método divide los datos tomando datos en posiciones pares o impares, en este caso se utilizo los pares como entrenamiento y los impares como prueba.<br>
Partición Estratificada (Stratified Split)<br>
Este método de partición se centra en mantener la proporción de clases en los conjuntos de entrenamiento y prueba. En lugar de seleccionar aleatoriamente datos, se asegura de que cada subconjunto contenga una cantidad proporcional de muestras de cada clase.<br>
En este caso no habia etiquetas que permitiran utilizar un mejor uso con el dataset pero parti a la mitad los datos que tenian el valor de 1 y -1 para y1 en entrenamiento y prueba<br>

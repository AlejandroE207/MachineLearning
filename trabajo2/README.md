# :chart_with_upwards_trend: REGRESIÓN LINEAL DE DATASET IRIS 
Realizado por:
* Alejandro Espinosa Riveros
* Dominic Nicolas Alonso Barajas
---
Inicialmente se desarrollo un modelo del DataSet IRIS utilizando los cuatro features como entrada en una regresión lineal RIDGE. Al momento de revisar los pesos de *W* de las entradas, nos logramos dar cuenta que dos de ellas que son el **largo** y **ancho** del sepalo, son las menos relevantes para la clasificación a comparación a el **largo** y **ancho** de petalo.
Debido a haber identificado esto, se decidio solo tener en cuenta estas dos variables para la regresión linea, con el fin de tener mayor facilidad al momento de entrenar el modelo y graficar los resultados en un hiperplano.

---
## Inicio de código
Se importa la libreria Pandas, Matplotlib para las gráficas y módulos de scikit-learn para cargar el dataset Iris, dividirlo en entrenamiento y prueba, entrenar el modelo de regresión lineal Ridge y evaluar su precisión.
```Python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, classification_report
```

--- 
## Carga y preparación de datos
Ya que el dataset se incluye en la libreria de *sklearn* se tiene un mayor acceso y solo basta con importarla y llamarla.
```Python
iris = load_iris()
```
El dataset de IRIS cuenta con 150 registros totales, con 4 features (largo y ancho de sepalo, largo y ancho de pétalo), junto con las 3 clases de flores (Setosa, Versicolor, Virginica).
Ya que el dataset esta cargado en el programa, se preparan para entregarlos al modelo, para esto de asigna todas las filas de los registros de flores de la columna 2 y 3 a la varibale *X* y en la variable *y* se asigna la categoria a la que corresponde cada uno de los registros.
```Python
X = iris.data[:, 2:4]          # solo petal length y petal width
y = iris.target               
class_names = ['Setosa (0)', 'Versicolor (1)', 'Virginica (2)']
```
---
## Train/ Test

Se divide el conjunto de datos en entrenamiento y prueba usando *train_test_split*. El 70 % de las muestras se usa para entrenar y el 30 % para evaluar, teniendo la misma proporción de clases gracias al parámetro stratify=y.
```Python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=37, stratify=y
)
```
---
## Escalado
Luego se haber identificado los datos que seran para el entrenamiento y los que seran utilizados para la etapa de pruebas, ahora se prepara los datos para que todas las variables estén en la misma escala, evitando que unas dominen a otras por su magnitudes y esto por medio de las siguientes partes del codigo:
### Primera
Crea un objeto de tipo StandardScalar con el fin de estandarizar las variables con la formula: $$`(z=(x-μ)/(σ)`$$
```Python
scaler = StandardScaler()
```
### Segunda
Para evitar que el modelo vea informacion del conjunto de test, se utiliza la siguiente linea.

```Python
X_train_s = scaler.fit_transform(X_train)
```
En la cual se calcula la media y desviación estándar de cada columna usando solo los datos de entrenamiento y escalarlos.

### Tercera
En esta parte solo se utiliza *transform* para aplicar la misma escala que se aprendio con los datos de *X*, en otras palabras se usa la media y la desviacion estandar para transformar los datos de prueba.
```Python
X_test_s  = scaler.transform(X_test)
```
---
## Crear Modelo
Ridge(alpha=1.27):usamos esta función debido a  que es una regresión lineal con una penalización L2 
Esto cumple con las condiciones puestas, puesto que , busca una función de la forma:
y= w_0 + w_1 x_1 + w_2 x_2
Donde w_0 es el umbral de conocimiento y w_1, w_2 son los pesos.
Ridge añade un término de penalización en la funcion para evitar que los pesos crezcan demasiado y que la relación entre variables siga siendo lineal.
```Python
model = Ridge(alpha=1.27)
model.fit(X_train_s, y_train)
```
---
## Evaluación del modelo
Una vez el modelo definido, se toma el modelo ya entrenado para evaluar los resultados.
Primero predice los valores  del conjunto de prueba y luego los aproxima para tener valores de 1,2,3, estos números determinaran el tipo de flor y así realizar la "clasificación" pedida en el ejercicio.
Después calcula la exactitud, y finalmente muestra la precisión, recall y f1-score para cada tipo de flor.
```Python
y_pred_cont = model.predict(X_test_s)
# Redondeo a entero y recorte al rango [0,2]
y_pred = np.clip(np.rint(y_pred_cont), 0, 2).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print("\n=== RESULTADOS DEL MODELO ===")
print(f"Exactitud (accuracy) en test: {accuracy:.3f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=class_names))
```
---
## Grafica
Como estamos usando 2 variables de entrada, generamos un gráfico 3D mostrando los puntos de los datos de entrenamiento en rojo y el plano de predicción del modelo Ridge en verde. Para eso, toma los rangos de las dos variables, calcula las predicciones del modelo en todos los puntos  y luego dibuja  los puntos y el plano para ver cómo el modelo se ajusta.
```Python
x1_range = np.linspace(X_train_s[:,0].min(), X_train_s[:,0].max(), 30)
x2_range = np.linspace(X_train_s[:,1].min(), X_train_s[:,1].max(), 30)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

Y_pred_plane = (model.intercept_
                + model.coef_[0]*X1_grid
                + model.coef_[1]*X2_grid)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_train_s[:,0], X_train_s[:,1], y_train,
           color='red', label='Datos de entrenamiento')

ax.plot_surface(X1_grid, X2_grid, Y_pred_plane,
                color='green', alpha=0.5, edgecolor='k')

ax.set_xlabel('Petal length (escalado)')
ax.set_ylabel('Petal width (escalado)')
ax.set_zlabel('Etiqueta (0–2)')
ax.set_title('Plano de regresión Ridge (2 variables)')
plt.tight_layout()
plt.show()
```
<img width="792" height="667" alt="image" src="https://github.com/user-attachments/assets/2428e55f-cef5-4d81-8b2f-a0d305c3c5ff" />
---

## Clasificación manual al usuario
Una vez el modelo entrenado y testeado, se da la opción de que el usuario , ingrese la longitud y el ancho del pétalo, él sistema los pasa al modelo  para que lo "clasifique". Luego muestra el numero de la ponderacion y su respectiva "clasificacion".El bucle termina cuando el usario lo desee.
```Python
print("\n--- Clasificación manual ---")
print("Introduce la longitud y el ancho del pétalo en cm.")
print("Escribe 0 en cualquiera de los campos para terminar.\n")

while True:
    try:
        petal_length = float(input("Longitud del pétalo (cm): "))
        if petal_length == 0:
            print("Fin del bucle.")
            break

        petal_width = float(input("Ancho del pétalo (cm): "))
        if petal_width == 0:
            print("Fin del bucle.")
            break

        new_sample = np.array([[petal_length, petal_width]])
        new_sample_scaled = scaler.transform(new_sample)

        pred_cont = model.predict(new_sample_scaled)[0]
        pred_class = int(np.clip(np.rint(pred_cont), 0, 2))

        print(f"Predicción continua: {pred_cont:.2f}")
        print(f"Clase predicha: {pred_class} ({class_names[pred_class]})\n")

    except ValueError:
        print(" Entrada no válida. Intenta de nuevo.\n")

```
--- 
# LINK DE REPOSITORIO: https://github.com/AlejandroE207/MachineLearning/tree/main/trabajo2 

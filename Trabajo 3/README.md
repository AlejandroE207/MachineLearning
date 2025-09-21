# :deciduous_tree: ARBOL DE DECISIONES PARA IDENTIFICAR HAM O SPAM
Realizado por:
* Alejandro Espinosa Riveros
* Dominic Nicolas Alonso Barajas
---

Se desarrolla un codigo que genera un arbol de decisiones para identificar los correos HAM o SPAM en base al dataset ya utilizado en trabajos anteriores. Este codigo realiza un bucle de 50 iteraciones en el cual se realiza esta misma cantidad de veces el proceso de aleatoriedad de los datos para su seleccion, el entrenamiento del model en base a los datos ya seleccionados, la predicción, el calculo de efectividad y calcula el promedio de F1. Esto con el fin de identificar y entender el comportamiento de la precisión durante todas las epocas de entrenamiento.

---
## Inicio de codigo
Lo primero que se realiza en el codigo para la correcta creación del modelo de arbol de decision, es importar las librerias que nos ayudara al desarrollo del codigo, las cuales son como pandas, numpy, matplotlib.pyplop y varios modulos de sklearn.
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import zscore
```

---
## Lectura de dataset
Para tener una mayor facilidad y mejor manejo de los datos del dataset, con ayuda de la libreria pandas se lee el archivo .csv para cargar los datos en el codigo en formato de un dataframe. 

Ya que se tiene el dataframe creado con los datos del dataset que se utilizara para identificar los correos HAM o SPAM, se realiza la separacion de las variables predictoras que en este caso seran aquellas caracteristicas que se deben de tener en cuenta para identificar el tipo de correo (*x*), de la variable objetivo la cual sera la etiqueta del tipo de correo (*y*).
```Python
df = pd.read_csv("dataset_datos_convertidos.csv")
X = df[['PorcentajeTexto', 'CorreoConTLS', 'ArchivosAdjuntosPeligrosos',
        'OfertasIrreales', 'ImagenesCodigoOculto', 'HeaderRemitenteFalso',
        'ContenidoSensible', 'Fecha_Num', 'Dominio_Num',
        'Rango_IP', 'URL_Num']]
y = df['Etiqueta_Num']
```
---
## Inicialización de listas
Se crean dos listas las cuales seran en las que se almacerana los resultados de precision y f1-score de cada una de las 50 iteraciones que realizara el programa, con el objetivo de utilizarlas al momento de graficar el comportamiento de cada una de estas.
```Python
precisiones = []
f1_scores = []
```
---
## Entrenamiento y predicción
Por medio de un bucle for, se realiza las 50 iteraciones el cual es el objetivo de este trabajo, ya que este trata de observar el comportamiento de la precisión en cada una de todas las epocas que el modelo realiza en el proceso de entrenamiento y test.

Dentro del bucle se aleatoriza el dataset y se divide una parte para el entrenamiento (70%) y otra parte para test (30%). Luego de esto, se hace la creación del modelo y se entrena ingresando los valores *x* y *y* como parametros. Ya que el arbol de decisiones esta entrenado, se realiza la etapa de test ingresando ese 30% de datos que fueron separados exclusivamente para esta etapa.

Ya que se tiene valores de respuesta del test, se compara los *y_test* con los *y_pred* para identificar el porcentaje de efectividad y ser almacenado en la lista de *precisiones* y para el F1 score, se promedia el F1 ponderado por la cantidad de muestras de ham y spam, este tambien es almacenado en su lista correspondiente *f1_score*.

```Python
for i in range(50):  # 50 épocas/iteraciones
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    arbol = DecisionTreeClassifier()  
    arbol.fit(X_train, y_train)
    y_pred = arbol.predict(X_test)
    precisiones.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
```
---
## Resultados
Luego de que el bucle termina las 50 iteraciones, se imprime el resultado de la precision y f1 score en las 50 epocas, posteriormente se muestra un reporte de la ultima iteracion.
```Python
print(" Resultados ")
print(f"Precision promedio: {np.mean(precisiones):.3f}")
print(f"F1-score promedio: {np.mean(f1_scores):.3f}")
print("\nUltima ejecución:\n")
print(classification_report(y_test, y_pred,
      target_names=[str(c) for c in sorted(y.unique())]))
```
<img width="553" height="358" alt="image" src="https://github.com/user-attachments/assets/ab6610f7-7b58-4fcd-8e56-6dd570fa7f20" />

---
## Graficos
Finalmente se crean y se imprimen los graficos necesarios para el analisis del comportamiento de la precisión, F1 score, el z-score de precisión y F1 Score, junto con el diagrama del arbol.

### Grafico Precisión vs Epocas
En este diagrama que logra apreciar el comportamiento de la precisión en cada epoca, llegando a la conclusión de que la epoca 25 se logro la mayor precisión del 82%.
```Python
plt.figure(figsize=(8,4))
plt.plot(range(1, 51), precisiones, marker='o', color='blue')
plt.xlabel('Epoca')
plt.ylabel('Precision (Accuracy)')
plt.title('Precision vs Épocas')
plt.grid(True)
plt.show()
```
<img width="993" height="584" alt="image" src="https://github.com/user-attachments/assets/e404a252-49a6-4caf-af20-ac34201add35" />

### Grafico F1-Score vs Epocas
En este diagrama se logra visualizar el comportamiento del f1 score, lograndose notar que tuvo un desempeño muy similar al de la precisión, pero confirmando que el modelo tuvo el mejor rendimiento en la epoca 25.
```Python
plt.figure(figsize=(8,4))
plt.plot(range(1, 51), f1_scores, marker='s', color='green')
plt.xlabel('Epoca')
plt.ylabel('F1-score ponderado')
plt.title('F1-score vs Epocas')
plt.grid(True)
plt.show()
```
<img width="991" height="582" alt="image" src="https://github.com/user-attachments/assets/fa197dce-e506-4a78-b178-ee75fb9c1ca4" />

### Grafico Z-score de precisión y F1-Score
En este grafico se logra detectar que la precisión es relativamente estable, sin sufrir grandes desviaciones respecto a la media, a diferencia del F1-Score, que es mucho mas variable con episodios de rendimiento muy alto y muy bajo, reflejando una mayor sensibilidad a los cambios de los datos.
```Python
plt.figure(figsize=(8,4))
plt.plot(range(1, 51), z_acc, marker='o', color='orange', label='Z-score Precision')
plt.plot(range(1, 51), z_f1, marker='s', color='red', label='Z-score F1')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Epoca')
plt.ylabel('Z-score')
plt.title('Z-score de Precision y F1-score')
plt.legend()
plt.grid(True)
plt.show()
```
<img width="993" height="576" alt="image" src="https://github.com/user-attachments/assets/424c6f3b-1b16-4756-8729-e7bf91b76247" />

### Grafico de estructura del árbol de decisión
En este grafico se logra visualizar toda la estructura del arbol, con las preguntas que realiza, sus ramas y finalmente sus hojas.
```Python
plt.figure(figsize=(20,10))
plot_tree(
    arbol,
    feature_names=X.columns,
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Diagrama del Árbol de Decision")
plt.show()
```
<img width="1919" height="1027" alt="image" src="https://github.com/user-attachments/assets/bcfed2fa-1e9d-41b9-a98d-672428ee108a" />

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

#Se importa la libreria pandas y numpy para el manejo y carga del dataset de los 1000 correos, y sklearn para el modelo de arbol de decision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import zscore

#Convertimos el dataset en un dataframe y definimos la variables predictoras, que seran nuestras caracteristicas, (x) y la variable objetivo que sera la etiqueta(y)
df = pd.read_csv("dataset_datos_convertidos.csv")
X = df[['PorcentajeTexto', 'CorreoConTLS', 'ArchivosAdjuntosPeligrosos',
        'OfertasIrreales', 'ImagenesCodigoOculto', 'HeaderRemitenteFalso',
        'ContenidoSensible', 'Fecha_Num', 'Dominio_Num',
        'Rango_IP', 'URL_Num']]
y = df['Etiqueta_Num']

#Generamos 2 listas vacias las cuales almacenaran los resultados de presicion y f1-score de las 50 iteraciones que realizara el programa
precisiones = []
f1_scores = []

# Por medio de un for , se va a realizar 50 iteraciones para cumplir con el objetivo del trabajo, el cual es observar el comportamiento de la presicion en todas las epocas que procesa en su respectivo entrenamiento y test 
#Dentro de este bucle, se dividie el dataset en conjunto de entrenamiento y prueba, se entrena el modelo de arbol de decision, se hacen predicciones y se guardan las metricas de presicion y f1-score en sus respectivas listas
#Finalmente se calcula el z-score de ambas metricas
for i in range(50):  # 50 épocas/iteraciones
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    arbol = DecisionTreeClassifier()  
    arbol.fit(X_train, y_train)
    y_pred = arbol.predict(X_test)
    precisiones.append(accuracy_score(y_test, y_pred))#Compara y_test con y_pred y obtiene el porcentaje de efectividad
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))#Promedia el F1 ponderando por la cantidad de muestras de ham y spam

z_acc = zscore(precisiones)
z_f1  = zscore(f1_scores)

#Una vez completado el bucle se imprime el resultado de la presicion  y f1 score en las 50 epocas, posteriormente se muestra un reporte de la ultima iteracion
#Despues se generan las graficas de presicion vs epocas, f1-score vs epocas, z-score de presicion y f1-score, y el diagrama del arbol de decision final
print(" Resultados ")
print(f"Precision promedio: {np.mean(precisiones):.3f}")
print(f"F1-score promedio: {np.mean(f1_scores):.3f}")
print("\nUltima ejecución:\n")
print(classification_report(y_test, y_pred,
      target_names=[str(c) for c in sorted(y.unique())]))



plt.figure(figsize=(8,4))
plt.plot(range(1, 51), precisiones, marker='o', color='blue')
plt.xlabel('Epoca')
plt.ylabel('Precision (Accuracy)')
plt.title('Precision vs Épocas')
plt.grid(True)
plt.show()


plt.figure(figsize=(8,4))
plt.plot(range(1, 51), f1_scores, marker='s', color='green')
plt.xlabel('Epoca')
plt.ylabel('F1-score ponderado')
plt.title('F1-score vs Epocas')
plt.grid(True)
plt.show()

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

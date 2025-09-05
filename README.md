# 📘 Codigo para regresión logistica 
Realizado por:
* Alejandro Espinosa Riveros
* Dominic Nicolas Alonso Barajas
---

## 🧾 Resumen / propósito

Se realiza dos codigos, el primero el cual esta enfocado en la transformación de los valores del dataset para que sea optimo y manejable al momento de aplicar regresión logistica, el segundo, es el codigo en el que se realiza la regresión logistica con ayuda de librerias.

---
## 🔄 1. Conversión de valores de dataset
### Inicio de codigo
Se importa la libreria mandas y se carga el dataset original en un Dataframe llamado df.
```Python
import pandas as pd
df = pd.read_csv(r"Trabajo 1\dataset_spam_700_300.csv")
```

### Transformación de campo fecha
Se define la funcion para transformar el primer dato que es el de la fecha, la cual se convierte en 5 rangos horarios, especificamente la hora en un valor discreto (1 a 5). Los rangos son:
* 00:00 – 03:59
* 04:00 – 07:59
* 08:00 – 11:59
* 12:00 – 17:59
* 18:00 – 23:59
  
Realmente lo que hace es convertir cada valor de *Fecha* a *Timestamp* y extrae la *hora* y mapea la hora a 5 rangos.
```Python
def convertir_fecha_5r(fecha):
    hora = pd.to_datetime(fecha, errors="coerce").hour
    if pd.isna(hora):
        return 3
    if 0 <= hora <= 3:
        return 1
    elif 4 <= hora <= 7:
        return 2
    elif 8 <= hora <= 11:
        return 3
    elif 12 <= hora <= 17:
        return 4
    else:
        return 5
```
Ya luego de haber creado la función se hace el llamo de la funcion y se elimina la feature de fecha que tenia el tipo de dato anterior, la cual sera remplazada por una feature que contendra los rangos horarios.
```Python
df["Fecha_Num"] = df["Fecha"].apply(convertir_fecha_5r)
df = df.drop(columns=["Fecha"])
```
### Transformación de campo dominio
Se crea la funcion la cual resive como parametro el valor del dominio, identifica si se encuentra entre alguno de las categorias y si si, se le clasifica con el numero identificador de la categoria:

1. Muy Confiables.
2. Confiables.
3. Neutrales.
4. Sospechosos.

```Python
def clasificar_dominio_5r(dominio):
    muy_confiables = ["gmail.com", "yahoo.com"]
    confiables = ["hotmail.com", "outlook.com", "protonmail.com"]
    neutrales = ["zoho.com", "tutanota.com", "icloud.com", "aol.com",
                  "mail.com", "empresa.org", "exito.com"]
    sospechosos = ["secure-mail.net", "fakebank.co", "securepay.net"]
    muy_sospechosos = ["freemail.xyz", "ofertas123.com",
                      "trabajo-rápido.org", "noticiaslive.co",
                      "exit0.com"]

    if dominio in muy_confiables:
        return 1
    elif dominio in confiables:
        return 2
    elif dominio in neutrales:
        return 3
    elif dominio in sospechosos:
        return 4
    else:
        return 5
```
Finalmente se remplaza el feature *Dominio* por la nueva feature *Dominio_num*.
```Python
df["Dominio_Num"] = df["Dominio"].apply(clasificar_dominio_5r)
df = df.drop(columns=["Dominio"])
```
### Transformación de campo IP
La función definida recibe como parametro la ip del correo e internamente la convierte a un número entero de 32-bits utilizando coeficientes (256)^3..256^0 generando un mapeo unico. Luego calcula los percentiles 20%, 40%, 60%, 80% en esa feature numerica. Finalmente asigna la IP a un rango entre 1 y 5 segun en que cuartil/percentil cae el valor.
```Python
def ip_a_numero(ip):
    try:
        octetos = list(map(int, ip.split(".")))
        return octetos[0]*256**3 + octetos[1]*256**2 + octetos[2]*256 + octetos[3]
    except:
        return 0

df["IP_num"] = df["IP"].apply(ip_a_numero)
p20, p40, p60, p80 = df["IP_num"].quantile([0.2, 0.4, 0.6, 0.8])

def rango_ip_5r(num):
    if num <= p20:
        return 1
    elif num <= p40:
        return 2
    elif num <= p60:
        return 3
    elif num <= p80:
        return 4
    else:
        return 5
```
Luego elimina la feature *IP_num* y la remplaza por la *Rango_IP*.
```Python
df["Rango_IP"] = df["IP_num"].apply(rango_ip_5r)
df = df.drop(columns=["IP_num", "IP"])
```
### Transformación de campo URL
La función recibe como parametro el URL anexado en el correo, busca substring sospechosos y asigna un puntaje entre el 1 y el 5 segun la clasificación interna de la palabra. Si no encuentra nada y la URL es corta (menor de 15 caracteres) asigna puntaje 2 sino, 1.
```Python
def clasificar_url_5r(url):
    url = str(url).lower()
    muy_sospechosos = ["phishing", "malicious", "freegift", "getprize",
                      "lottery", ".xyz", ".biz", ".io", ".co", ".win"]
    sospechosos = ["secure-login", "freemoney", "discountzone",
                  "investcrypto", "promo"]
    neutro = [".com", ".org", ".net", "bank", "offer"]

    if any(s in url for s in muy_sospechosos):
        return 5
    elif any(s in url for s in sospechosos):
        return 4
    elif any(s in url for s in neutro):
        return 3
    elif len(url) < 15:
        return 2
    else:
        return 1
```
Finalmente elimina la feature *UrlIndexados* y la remplaza por la feature *URL_Num*.
```Python
df["URL_Num"] = df["UrlsIndexados"].apply(clasificar_url_5r)
df = df.drop(columns=["UrlsIndexados"])
```
### Transformación de campos Bool
El objetivo de este bloque de codigo es convertir los features booleanos (True/False) en bit (1/0) por medio del metodo *astype(in)*.
```Python
columnas_bool = ["CorreoConTLS", "ArchivosAdjuntosPeligrosos",
                "OfertasIrreales", "ImagenesCodigoOculto",
                "HeaderRemitenteFalso", "ContenidoSensible"]

for col in columnas_bool:
    df[col] = df[col].astype(int)
```
### Transformación de campo etiqueta
Este bloque de codigo mapea la columna *Etiqueta* en 1/0 dependiendo si es registro se identifica como "HAM" o "SPAM".
```Python
df["Etiqueta_Num"] = df["Etiqueta"].map({1:1, 0:0})
df = df.drop(columns=["Etiqueta"])
```
### Almacenamiento de dataset
Al finalizar la transformación de todos los datos, se almacena el nuevo dataset con el nombre *dataset_datos_convertidos.csv* en la misma ruta raíz donde esta ubicado el codigo de transformación.
```Python
df.to_csv(r"Trabajo 1\dataset_datos_convertidos.csv", index=False)
print("✅ Dataset convertido y listo para regresión logística")
print(df.head())
```
---
## 2. Regresión Logistica
### Importaciones
Inicialmente se importan librerias como *pandas* para el manejo de tablas, *train_test_split* para partir los datos en conjuntos de entrenamiento y de prueba, *logisticRegression* el modelo para aplicar regresión logistica al dataset, entre otras.
```Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
```
### Carga de dataset
Se lee el CSV en un DataFrame, en el codigo se hara referencia con el nombre de *df*. Este dataset ya posee los datos categorizados en rangos numericos con el fin de que sea mas facil aplicar la regresión logistica.
```Python
df = pd.read_csv(r"Trabajo 1\dataset_datos_convertidos.csv")
```
### Mezcla del dataset
Por medio de *.sample(frac=1)* se obtiene una copia barajada, ademas se restablece los indices consecutivos, esto con el fin de que no se vaya a entrenar el modelo de regresión logistica con datos ordenados y mucho menos con datos de una sola categoria (HAMP/SPAM).
```Python
df = df.sample(frac=1).reset_index(drop=True)
```
### Separación de datos
Se separa los datos en 2 variables, en *X* la cual contendra todos los features del dataset excepto el de etiqueta, ya que este sera almacenado en la variable *y*. Esto con el objetivo de cumplir la estructura que exige la libreria para realizar regresión logistica y tenga un manejo mas sencillo.
```Python
X = df.drop(columns=["Etiqueta_Num"])
y = df["Etiqueta_Num"]
```
### Dividir datos en entrenamiento y prueba
En el siguiente bloque de codigo lo que se realiza es que en base a los 1000 datos que tiene registrados el dataset, se ingresan al modelo los porcentajes de esos datos que van a corresponder a la etapa de entrenamiento y a la etapa de prueba, en este caso sera 70% para entrenamiento y 30% para pruebas. A demas, se procura mantener la proporción entre clases para que no se presente un desequilibrio de los datos al momento del entrenamiento.
```Python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y
)
```
### Entrenar modelo
Se crea una instancia del modelo, asignando el parametro de la cantidad maxima de iteraciones del optimizador para evitar alertas de convergencia, lo cual es muy util cuando hay muchas catacteristicas. En este caso la cantidad maxima de iteraciones sera de 1000.
Ya que se haya creado la instancia del modelo con su parametro, ahora se entrena por medio del metodo *.fit()* en el cual se ingresa como parametro los valores de las variables *X* y *y*.
```Python
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)
```
### Predicciones
Se recibe la predicción (1/0) según la probabilidad estimada y el umbral por defecto que en este caso es de *0.5* de los valores de entrenamiento los cuales son ingresados como parametro.
```Python
y_pred = modelo.predict(X_test)
```
### Metricas
Se calcula e imprime las metricas en base a las predicciones anteriores.
La metrica *accuracy* lo que realiza es calcular la exactitud de la predicción en base a la proporcion de predicciones correctas respecto al total.
El *F1 score* combina precisión y recobrado tambien llamado recall (qué tanto de lo positivo real logró capturar el modelo).
Luego de calcular las anteriores metricas se imprimen por consola.
```Python
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f" Exactitud (Accuracy): {acc*100:.2f}%")
print(f" F1 Score: {f1:.2f}")
```
### Grafico de Exactitud y F1 Score
Se utiliza *Seaborn* para dibujar un gráfico de barras, el cual en el eje *X* se representa la exactitud y el f1 score, en el eje *y* los valores calculados de las metricas en porcentaje.
```Python
plt.figure(figsize=(6,4))
sns.barplot(x=["Exactitud", "F1 Score"], y=[acc*100, f1*100], palette="coolwarm")
plt.ylim(0, 100)
plt.ylabel("Porcentaje (%)")
plt.title("Desempeño del Modelo")
for i, v in enumerate([acc*100, f1*100]):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
plt.show()
```
<img width="747" height="590" alt="image" src="https://github.com/user-attachments/assets/d3a273b8-6e8f-4148-8989-84f9323859cf" />

### Matriz de confusión
Inicialmente se calcula la matriz de confusión comprando las etiquetas reales con las predichas, esta matriz es una tabla 2x2, luego se calcula los porcentajes de cada etiqueta.
Se utiliza *Seaborn Heatmap* para mostrar la matriz com una imagen conde cada fila muestra como se distribuyeron las predicciones para una clase real. . 

<img width="695" height="668" alt="image" src="https://github.com/user-attachments/assets/990baa32-4068-43b7-aa76-99072a2b4d99" />

### El modelo finalmente tiene una exactitud entre el 80% y el 84%  en la identificación si un correo es HAM o SPAM, ya que se tiene una variabilidad en la seleccion de los datos de entrenamiento.
---


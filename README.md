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

### Transformación de fecha
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
Ya luego de haber creado la función se hace el llamo de la funcion y se elimina la columna de fecha que tenia el tipo de dato anterior, la cual sera remplazada por una columna que contendra los rangos horarios
```Python
df["Fecha_Num"] = df["Fecha"].apply(convertir_fecha_5r)
df = df.drop(columns=["Fecha"])
```

---


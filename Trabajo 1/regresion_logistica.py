import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1️ Cargar dataset
df = pd.read_csv(r"dataset_datos_convertidos.csv")

# 2️ Mezclar dataset
df = df.sample(frac=1).reset_index(drop=True)

# 3️ Separar X e y
X = df.drop(columns=["Etiqueta_Num"])
y = df["Etiqueta_Num"]

# 4️ Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y
)

# 5️ Entrenar modelo
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# 6️ Predicciones
y_pred = modelo.predict(X_test)

# 7️ Métricas
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f" Exactitud (Accuracy): {acc*100:.2f}%")
print(f" F1 Score: {f1:.2f}")

# 8️ Graficar Exactitud y F1 Score
plt.figure(figsize=(6,4))
sns.barplot(x=["Exactitud", "F1 Score"], y=[acc*100, f1*100], palette="coolwarm")
plt.ylim(0, 100)
plt.ylabel("Porcentaje (%)")
plt.title("Desempeño del Modelo")
for i, v in enumerate([acc*100, f1*100]):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
plt.show()

# 9️ Matriz de confusión en porcentaje
cm = confusion_matrix(y_test, y_pred)
cm_porcentaje = cm.astype('float') / cm.sum(axis=1)[:, None] * 100
plt.figure(figsize=(7,6))
sns.heatmap(cm_porcentaje, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=["Predicho HAM", "Predicho SPAM"],
            yticklabels=["Real HAM", "Real SPAM"])
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.title("Matriz de Confusión (%)")
plt.show()

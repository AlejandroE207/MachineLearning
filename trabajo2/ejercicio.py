import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, classification_report

# =======================
# 1. Cargar y preparar datos
# =======================
iris = load_iris()
X = iris.data[:, 2:4]          # solo petal length y petal width
y = iris.target               # etiquetas 0,1,2
class_names = ['Setosa (0)', 'Versicolor (1)', 'Virginica (2)']

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=37, stratify=y
)

# Escalado
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# =======================
# 2. Entrenar modelo
# =======================
model = Ridge(alpha=1.27)
model.fit(X_train_s, y_train)

# =======================
# 3. Evaluación del modelo
# =======================
y_pred_cont = model.predict(X_test_s)
# Redondeo a entero y recorte al rango [0,2]
y_pred = np.clip(np.rint(y_pred_cont), 0, 2).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print("\n=== RESULTADOS DEL MODELO ===")
print(f"Exactitud (accuracy) en test: {accuracy:.3f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=class_names))

# =======================
# 4. Gráfica 3D del plano de regresión
# =======================
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

# =======================
# 5. Clasificación manual en bucle
# =======================
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

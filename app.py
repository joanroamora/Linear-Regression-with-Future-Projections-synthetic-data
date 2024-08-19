import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synth_data import generateData  # Importar la función generateData desde synth_data.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    # Preguntar si se quiere generar un nuevo set de datos
    create_new_data = input("Do you want to generate a new dataset? (yes/no): ").strip().lower()

    if create_new_data == "yes":
        # Preguntar cuántas filas se desean generar
        n_rows = int(input("How many rows do you want to generate? "))
        generateData(n_rows)  # Generar el nuevo dataset
        print(f"New dataset with {n_rows} rows generated.")

    # Mostrar archivos CSV en la carpeta /data y permitir la selección
    data_files = os.listdir('data')
    csv_files = [file for file in data_files if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the 'data' directory.")
        return

    print("Available datasets:")
    for i, file in enumerate(csv_files):
        print(f"{i + 1}. {file}")

    file_index = int(input("Select the dataset to use by number: ")) - 1
    selected_file = csv_files[file_index]
    data_path = f"data/{selected_file}"

    # Cargar el dataset seleccionado
    data = pd.read_csv(data_path)
    print(f"Using dataset: {selected_file}")

    # Separar características y objetivo
    X = data[['Feature']]  # Usar la columna 'Feature' como entrada
    y = data['Target']     # Usar la columna 'Target' como salida

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Predecir 50 datos futuros basados en el modelo
    X_future = np.array([[X['Feature'].max() + i] for i in range(1, 51)])
    X_future = pd.DataFrame(X_future, columns=['Feature'])  # Solución para la advertencia de nombres de características
    y_future_pred = model.predict(X_future)

    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Original Data')  # Datos originales
    plt.plot(X, model.predict(X), color='green', label='Linear Regression Line')  # Línea de regresión
    plt.plot(X_future, y_future_pred, color='red', linestyle='dashed', label='Future Predictions')  # Predicciones futuras
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression and Future Projections')
    plt.legend()
    
    # Guardar la gráfica en un archivo
    plt.savefig('linear_regression_projection.png')
    print("La gráfica se ha guardado como 'linear_regression_projection.png'")

if __name__ == "__main__":
    main()

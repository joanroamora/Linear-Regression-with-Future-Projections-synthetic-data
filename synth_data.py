import numpy as np
import pandas as pd
from datetime import datetime
import os

def generateData(n_rows):
    # Generar datos sintéticos
    np.random.seed(42)
    X = 2 * np.random.rand(n_rows, 1)
    y = 4 + 3 * X + np.random.randn(n_rows, 1)

    # Crear un DataFrame
    data = pd.DataFrame(np.c_[X, y], columns=['Feature', 'Target'])

    # Crear el nombre del archivo con la fecha y hora actual (snaptime)
    snaptime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snaptime_{snaptime}.csv"

    # Asegurarse de que la carpeta /data existe
    os.makedirs('data', exist_ok=True)

    # Guardar el DataFrame en un archivo CSV en la carpeta /data
    data.to_csv(f"data/{filename}", index=False)

    print(f"Dataset exportado a: data/{filename}")



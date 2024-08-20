# :us: Linear Regression Project with Future Predictions

This project implements a linear regression model to analyze a synthetic dataset, predict future values, and plot the results. The project consists of two main scripts: `app.py` and `synth_data.py`.

## Files

### 1. `app.py`

This script is the core of the application and performs the following functions:

- **Synthetic Data Generation:** Allows the user to generate a new dataset with a specified number of rows.
- **Dataset Selection:** Displays available CSV files in the `data` folder and allows the user to select one for analysis.
- **Model Training:** Uses linear regression to train a model with the selected data.
- **Model Evaluation:** Calculates and displays the Mean Squared Error (MSE) to evaluate the model's performance.
- **Future Predictions:** Generates predictions for 50 future data points based on the trained model.
- **Visualization:** Plots the original data, the regression line, and future predictions, saving the plot as `linear_regression_projection.png`.

### 2. `synth_data.py`

This script is responsible for generating and exporting a synthetic dataset. It works as follows:

- **Data Generation:** Creates synthetic data based on a simple linear relationship with noise.
- **CSV Export:** Saves the generated dataset as a CSV file in the `data` folder, with a unique filename that includes the current date and time.

## How to Run

1. **Install Dependencies:**
Run the Script:

To run the main application:

bash
Copiar código
python app.py
Follow the on-screen instructions to generate a new dataset, select an existing one, train the model, and visualize the predictions.

###**Output**
The script will generate a PNG file with the linear regression plot and future projections in the project's main directory, named linear_regression_projection.png.

###**Directory Structure**
plaintext
Copiar código
├── app.py
├── synth_data.py
├── data/
│   └── [generated datasets are stored here]
├── linear_regression_projection.png
└── README.md
**Notes**
The data/ directory is automatically created if it does not exist, and it is where the generated datasets are stored.
The generateData function allows generating a new dataset with the number of rows specified by the user.
The application can handle multiple CSV files in the data directory, allowing flexibility in dataset selection for analysis.



🇨🇴 ##Proyecto de Regresión Lineal con Predicciones Futuras

Este proyecto implementa un modelo de regresión lineal para analizar un conjunto de datos sintéticos, predecir valores futuros y graficar los resultados. El proyecto está compuesto por dos scripts principales: app.py y synth_data.py.

**Archivos**
1. app.py
Este script es el núcleo de la aplicación y realiza las siguientes funciones:

**Generación de Datos Sintéticos:**
Permite al usuario generar un nuevo conjunto de datos con un número especificado de filas.

**Selección de Dataset:** Muestra los archivos CSV disponibles en la carpeta data y permite al usuario seleccionar uno para su análisis.

**Entrenamiento del Modelo:** Utiliza regresión lineal para entrenar un modelo con los datos seleccionados.
Evaluación del Modelo: Calcula y muestra el error cuadrático medio (MSE) para evaluar el rendimiento del modelo.

**Predicciones Futuras:** Genera predicciones para 50 futuros puntos de datos basados en el modelo entrenado.
**Visualización:** Grafica los datos originales, la línea de regresión y las predicciones futuras, guardando la gráfica como linear_regression_projection.png.


###2. synth_data.py
Este script se encarga de generar y exportar un dataset sintético. Funciona de la siguiente manera:

**Generación de Datos:** 

Crea datos sintéticos basados en una simple relación lineal con ruido.

**Exportación a CSV:** Guarda el dataset generado en un archivo CSV dentro de la carpeta data, con un nombre único que incluye la fecha y hora actuales.
Cómo Ejecutar
Instalación de Dependencias:

Asegúrate de tener las siguientes bibliotecas instaladas:

bash
Copiar código
pip install numpy pandas matplotlib scikit-learn
Ejecutar el Script:

###Para ejecutar la aplicación principal:

bash
Copiar código
python app.py
Sigue las instrucciones en pantalla para generar un nuevo dataset, seleccionar uno existente, entrenar el modelo y visualizar las predicciones.

###Resultados
El script generará un archivo PNG con la gráfica de regresión lineal y las proyecciones futuras en el directorio principal del proyecto, con el nombre linear_regression_projection.png.

###Estructura de Directorios
plaintext
Copiar código
├── app.py
├── synth_data.py
├── data/
│   └── [datasets generados se almacenan aquí]
├── linear_regression_projection.png
└── README.md
**Notas**
El directorio data/ se crea automáticamente si no existe, y es donde se almacenan los datasets generados.
La función generateData permite generar un nuevo conjunto de datos con la cantidad de filas que el usuario especifique.
La aplicación puede manejar múltiples archivos CSV en el directorio data, permitiendo flexibilidad en la selección de datasets para el análisis.
   Make sure you have the following libraries installed:

   ```bash
   pip install numpy pandas matplotlib scikit-learn

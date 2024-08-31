# PROGRAMA PRINCIPAL

# importar librerias necesarias
import pandas as pd # para manejar dataframes
import  Notebooks.functions as functions
from sklearn.preprocessing import OrdinalEncoder
import joblib 

# ------ Procesar los archivos de video ----------------------
# directorio donde se encuentran los datos
directory = '../dataset/SkeletonData/PruebaFinal' 

columnas = ['SubjectID', 'DateID', 'RepetitionNumber', 'Position',
            'JointName', 'TrackedStatus', '3D_X', '3D_Y', '3D_Z', '2D_X', '2D_Y']

# Extraer la informacion y almacenarla en un DataFrame
df_data = functions.leer_datos_archivo(directory, columnas)

# Eliminar columnas innceserarias
df_data.drop(['TrackedStatus', 'DateID', '2D_X', '2D_Y'], axis=1,
             inplace=True)

# ------ Calcular los ángulos -----------
df_angles = functions.apply_angles(df_data)

# ---- Cálculos estadisticos sobre los ángulos ----
df_stats = functions.calculos_estadísticos(df_angles)
df_stats = functions.formatear_columnas(df_stats)

columnas = ['standardDeviation', 'Maximum', 'Minimum', 'Mean', 'Range',
            'Variance', 'CoV', 'Skewness', 'Kurtosis']
# Reformatea cada columna de diccionario y concatena los resultados
nuevas_columnas = pd.concat([functions.formatear_columnas(col) for col in columnas], axis=1)

# Concatenar las nuevas columnas con el DataFrame original
df_stats = pd.concat([df_stats, nuevas_columnas], axis=1)

# Elimina las columnas originales que contenían diccionarios
df_stats = df_stats.drop(columnas, axis=1)

# Ordena el DataFrame 
df_stats = df_stats.sort_values(['SubjectID', 'GestureLabel', 'RepetitionNumber'])

# ------Preparar el dataset-------
# pasar variable obj a numeric
df_stats = df_stats.apply(pd.to_numeric, errors='ignore')

# -------- Fase 1: Predecir el gesto -------
modelo_fase1 = joblib.load('my_model_knn.pkl.pkl')
gesture_label = modelo_fase1.predict(df_stats)
print(gesture_label)

# ---- Fase 2: Clasificación de la ejecución del movimiento ------


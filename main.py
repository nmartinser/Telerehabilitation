# PROGRAMA PRINCIPAL

# importar librerias necesarias
import pandas as pd # para manejar dataframes
import  Notebooks.functions as functions
import joblib 
import os

# ------ Procesar los archivos de video ----------------------
# directorio donde se encuentran los datos
directory = './dataset/SkeletonData/PruebaFinal' 

columnas = ['SubjectID', 'DateID', 'GestureLabel', 'RepetitionNumber', 'CorrectLabel', 'Position',
            'JointName', 'TrackedStatus', '3D_X', '3D_Y', '3D_Z', '2D_X', '2D_Y']

# Extraer la informacion y almacenarla en un DataFrame
df_data = functions.leer_datos_archivo(directory, columnas)

# Eliminar columnas innceserarias
df_data.drop(['DateID','GestureLabel', 'CorrectLabel', 'TrackedStatus', '2D_X', '2D_Y'], axis=1,
             inplace=True)


print(df_data.head())

# ------ Calcular los ángulos -----------
print('Calulando ángulos...')
df_angles = functions.apply_angles(df_data)
print(df_angles.head())

# ---- Cálculos estadisticos sobre los ángulos ----
print('Calulando estadísticas...')
df_stats = functions.calculos_estadísticos(df_angles)

columnas = ['standardDeviation', 'Maximum', 'Minimum', 'Mean', 'Range',
            'Variance', 'CoV', 'Skewness', 'Kurtosis']
# Reformatea cada columna de diccionario y concatena los resultados
nuevas_columnas = pd.concat([functions.formatear_columnas(df_stats[col], col) for col in columnas], axis=1)

# Concatenar las nuevas columnas con el DataFrame original
df_stats = pd.concat([df_stats, nuevas_columnas], axis=1)

# Elimina las columnas originales que contenían diccionarios
df_stats = df_stats.drop(columnas, axis=1)

print(df_stats.head())

# ------Preparar el dataset-------
# pasar variable obj a numeric
df_stats = df_stats.apply(pd.to_numeric, errors='ignore')

# -------- Fase 1: Predecir el gesto -------
print('Prediciendo el gesto...')
modelo_fase1 = joblib.load('./Resultados/modelo_fase1.sav')
gesture_labels = modelo_fase1.predict(df_stats)
gesture_label = functions.mas_comun(gesture_labels)

gesture_mapping = {
    0: 'EFL',
    1: 'EFR',
    2: 'SFL',
    3: 'SFR',
    4: 'SAL',
    5: 'SAR',
    6: 'SFE',
    7: 'STL',
    8: 'STR'
}

gesture_name = gesture_mapping.get(gesture_label, 'Error en la predición')
print(f'Está realizando el gesto {gesture_name}')


# ---- Fase 2: Clasificación de la ejecución del movimiento ------
print('Prediciendo si está bien ejecutado...')

# Busca el archivo correspondiente al gesto predicho
modelo_gesto_path = f'./Resultados/modelo_{gesture_name}.sav'

correct_mapping = {
    1: 'corresto',
    2: 'incorrecto'
}

modelo_gesto = joblib.load(modelo_gesto_path)
correctLabel = modelo_gesto.predict(df_stats)
correctLabel = functions.mas_comun(correctLabel)
correct_name = gesture_mapping.get(correctLabel, 'Error en la predición')
print(f'El gesto se ha ejecutado de forma: {correct_name}')





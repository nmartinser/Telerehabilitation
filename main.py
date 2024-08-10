# PROGRAMA PRINCIPAL

# importar librerias necesarias
import pandas as pd # para manejar dataframes
import numpy as np
import Notebooks.funtions as funtions

# ------ Procesar los archivos de video ----------------------

# directorio donde se encuentran los datos
directory = '../dataset/SkeletonData/RawData' 

columnas = ['SubjectID', 'DateID', 'GestureLabel', 'RepetitionNumber', 'CorrectLabel', 'Position',
            'JointName', 'TrackedStatus', '3D_X', '3D_Y', '3D_Z', '2D_X', '2D_Y']

# Extraer la informacion y almacenarla en un DataFrame
df_data = funtions.leer_datos_archivo(directory, columnas)

# Eliminar columnas innceserarias
df_data.drop(['TrackedStatus', 'DateID', '2D_X', '2D_Y'], axis=1,
             inplace=True)

# ------ Calcular los ángulos -----------

df_angles = funtions.apply_angles(df_data)
# PROGRAMA PRINCIPAL

# importar librerias necesarias
import pandas as pd # para manejar dataframes
import Notebooks.funtions as funtions
from sklearn.preprocessing import OrdinalEncoder
import joblib 

# ------ Procesar los archivos de video ----------------------
# directorio donde se encuentran los datos
directory = '../dataset/SkeletonData/RawData' 

columnas = ['SubjectID', 'DateID', 'RepetitionNumber', 'Position',
            'JointName', 'TrackedStatus', '3D_X', '3D_Y', '3D_Z', '2D_X', '2D_Y']

# Extraer la informacion y almacenarla en un DataFrame
df_data = funtions.leer_datos_archivo(directory, columnas)

# Eliminar columnas innceserarias
df_data.drop(['TrackedStatus', 'DateID', '2D_X', '2D_Y'], axis=1,
             inplace=True)

# ------ Calcular los ángulos -----------
df_angles = funtions.apply_angles(df_data)

# ------Preparar el dataset-------
encoder = OrdinalEncoder(categories=[list(set(df_angles["Position"].values))])
encoder.fit(df_angles[["Position"]])
df_angles["Position"] = encoder.transform(df_angles[["Position"]])
# pasar variable obj to numeric
df_angles = df_angles.apply(pd.to_numeric, errors='ignore')

# -------- Fase 1: Predecir el gesto -------
modelo_fase1 = joblib.load('my_model_knn.pkl.pkl')
gesture_label = modelo_fase1.predict(df_angles)

# ---- Fase 2: Clasificación de la ejecución del movimiento ------

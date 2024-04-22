# importar librerias necesarias
import pandas as pd # para manejar dataframes
import matplotlib.pyplot as plt
import numpy as np

print('Cargando csv...')
df_data = pd.read_csv('./csvFiles/raw_pacientes.csv', dtype=object)
print(df_data.head())

def calculate_angle(df: pd.DataFrame, joint_a: str, joint_b: str, joint_c: str):
    a = df.loc[df['JointName'] == joint_a, ['3D_X', '3D_Y']].to_numpy()[0]
    b = df.loc[df['JointName'] == joint_b, ['3D_X', '3D_Y']].to_numpy()[0]
    c = df.loc[df['JointName'] == joint_c, ['3D_X', '3D_Y']].to_numpy()[0]

    # convertir a numerico
    a = pd.to_numeric(a)
    b = pd.to_numeric(b)
    c = pd.to_numeric(c)


    # cacular los radianes y el angulo
    # los [1] representan las y y los [0] las x
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # para tener un maximo de 180 grados
    if angle > 180.0:
        angle = 360 - angle

    return angle

print('Calculando angulos')

groups = df_data.groupby(['SubjectID', 'GestureLabel', 'RepetitionNumber', 'CorrectLabel'])
angle_data = []
joint = []
for key, values in groups:
    angle_data.append({
        'SubjectID': key[0],
        'GestureLabel': key[1],
        'RepetitionNumber': key[2],
        'CorrectLabel': key[3]
    })
    n_rows = 25
    for i in range(0, n_rows, len(values)):
        print(calculate_angle(values.iloc[i:i+n_rows], 'ShoulderLeft', 'ElbowLeft', 'WristLeft'))

pd.DataFrame(angle_data)
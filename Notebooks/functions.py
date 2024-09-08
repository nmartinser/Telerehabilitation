# funciones creadas para el programa principal

# importar librerias necesarias
import pandas as pd # para manejar dataframes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------ Procesar los archivos de video ----------------------

# Función para guardar en el dataframe los datos que aparecen en los nombre de los archivos
def leer_nombre_archivo(archivo:str) -> list[str]:
    """
    Extrae datos específicos del nombre de un archivo de texto.

    Parámetros
    ----------
    archivo : str
        Nombre del archivo, con el formato 'SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.txt'.

    Return
    -------
    campos : list[str]
        Lista de cadenas de texto que contiene los datos extraídos del nombre del archivo:
        [SubjectID, DateID, GestureLabel, RepetitionNumber, CorrectLabel, Position].
    """
    archivo = archivo.split('.')[0] # quita la extension txt
    campos = archivo.split('_') # separa los campos por _
    return campos


# Función para extraer la información de dentro de los archivos
def leer_datos_archivo(lista_archivos:list, columnas:list[str]) -> pd.DataFrame:
    """
    Compila la información de los archivos subidos a la app y los guarda en un DataFrame.

    Parámetros
    ----------
    lista_archivos : list[UpleadedFile]
        Lista de archivos subidos.
    columnas : list[str]
        Lista con los nombres de las columnas para el DataFrame de salida.

    Return
    -------
    pd.DataFrame
        DataFrame con todos los datos recopilados de los archivos.
    """
     # lista para almacenar los datos extraídos
    list_data = []

    # Itera sobre cada archivo
    for uploaded_file in lista_archivos:
         # Extrae los datos del nombre del archivo
        campos = leer_nombre_archivo(uploaded_file.name)

        # Lee el contenido del archivo y decodifica a string (asume UTF-8)
        file_content = uploaded_file.read().decode('utf-8')  # Ensure content is decoded to str

        # Itera sobre las líneas del archivo ya decodificado
        for line in file_content.splitlines():
            # Divide la línea por comas y extrae la información deseada
            # omitiendo timestamp y otros datos innecesarios
            line_data = line.strip().split(',')[3:]
            # Quita los paréntesis
            cleaned_data = [item.replace('(', '').replace(')', '') for item in line_data] 
            # por cada linea de los archivos necesitamos bloques de 7 valores
            for i in range(0, len(cleaned_data), 7): 
                list_data.append(campos + cleaned_data[i:i + 7])
    df = pd.DataFrame(list_data, columns=columnas)
    return df


# Función para calcular los ángulos
def calculate_angle(df: pd.DataFrame, joint_a: str, joint_b: str, joint_c: str) -> float:
    """
    Calcula el ángulo entre dos vectores definidos por tres puntos (keypoints) en un DataFrame.

    Parámetros
    -------
    df : pd.DataFrame
        DataFrame que contiene los datos de los keypoints, con una columna 'JointName' y columnas de coordenadas '3D_X', '3D_Y', '3D_Z'.
    joint_a : str
        Nombre del primer keypoint.
    joint_b : str
        Nombre del segundo keypoint (punto de conexión entre los dos segmentos).
    joint_c: str
        Nombre del tercer keypoint.

    Return
    -------
    float : El ángulo en grados entre los dos vectores formados por los tres keypoints.

    """
    # Extraer posiciones de los keypoints
    positions = df.set_index('JointName')[['3D_X', '3D_Y', '3D_Z']].loc[[joint_a, joint_b, joint_c]]

    # # Convertir las posiciones a tipo numérico
    positions = positions.apply(pd.to_numeric)

    # Vector u (joint_a to joint_b) y Vector v (joint_b to joint_c)
    u = np.array([positions.iloc[1, 0] - positions.iloc[0, 0], positions.iloc[1, 1] - positions.iloc[0, 1], positions.iloc[1, 2] - positions.iloc[0, 2]])
    v = np.array([positions.iloc[2, 0] - positions.iloc[1, 0], positions.iloc[2, 1] - positions.iloc[1, 1], positions.iloc[2, 2] - positions.iloc[1, 2]])
  
    # Producto vectorial y modulo de los vectores
    producto_vectorial = np.dot(u, v)
    modulo_u = np.linalg.norm(u)
    modulo_v = np.linalg.norm(v)

    # Comprobar que ninguno de los módulos sea 0
    if (modulo_u * modulo_v) == 0:
        return 0

    # Caclulo del coseno
    cos_angle = producto_vectorial / (modulo_u * modulo_v)

    # Calcular el angulo
    angle = np.arccos(cos_angle) * 180.0 / np.pi

    # Asugurar que el ángulo esté entre los 180 grados
    if angle > 180.0:
        angle = 360 - angle

    return angle
 
def apply_angles(df: pd.DataFrame) -> pd.DataFrame:
    # Aplica la función para caluclar los ángulos a los datos en crudo

    angles = []

    # Agrupar el DataFrame por cada 25 filas e iterar
    for _, group in df.groupby(np.arange(len(df)) // 25):
        # Calculate angles for the group
        elbow_angle_left = calculate_angle(group, 'ShoulderLeft', 'ElbowLeft', 'WristLeft')
        elbow_angle_right = calculate_angle(group, 'ShoulderRight', 'ElbowRight', 'WristRight')
        left_arm_angle = calculate_angle(group, 'HipLeft', 'ShoulderLeft', 'ElbowLeft')
        right_arm_angle = calculate_angle(group, 'HipRight', 'ShoulderRight', 'ElbowRight')
        arms_together_angle = calculate_angle(group, 'SpineBase', 'SpineShoulder', 'WristLeft')
        wrist_angle_left = calculate_angle(group, 'ElbowLeft', 'WristLeft', 'HandLeft')
        shoulder_angle_left = calculate_angle(group, 'ShoulderLeft', 'SpineShoulder', 'ElbowLeft')
        wrist_angle_right = calculate_angle(group, 'ElbowRight', 'WristRight', 'HandRight')
        shoulder_angle_right = calculate_angle(group, 'ShoulderRight', 'SpineShoulder', 'ElbowRight')
        hip_angle_left = calculate_angle(group, 'HipLeft', 'SpineBase', 'KneeLeft')
        knee_angle_left = calculate_angle(group, 'HipLeft', 'KneeLeft', 'AnkleLeft')
        ankle_angle_left = calculate_angle(group, 'KneeLeft', 'AnkleLeft', 'FootLeft')
        hip_angle_right = calculate_angle(group, 'HipRight', 'SpineBase', 'KneeRight')
        knee_angle_right = calculate_angle(group, 'HipRight', 'KneeRight', 'AnkleRight')
        ankle_angle_right = calculate_angle(group, 'KneeRight', 'AnkleRight', 'FootRight')
        
        # # Extraer columnas adicionales
        additional_data = group.iloc[0][['SubjectID', 'RepetitionNumber', 'Position']]

        # Almacenar la información en un diccionario
        angles.append({
            **additional_data,
            'ElbowAngleLeft': elbow_angle_left,
            'ElbowAngleRight': elbow_angle_right,
            'ShoulderAngleLeft': shoulder_angle_left,
            'ShoulderAngleRight': shoulder_angle_right,
            'WristAngleLeft': wrist_angle_left,
            'WristAngleRight': wrist_angle_right,
            'HipAngleLeft': hip_angle_left,
            'KneeAngleLeft': knee_angle_left,
            'AnkleAngleLeft': ankle_angle_left,
            'HipAngleRight': hip_angle_right,
            'KneeAngleRight': knee_angle_right,
            'AnkleAngleRight': ankle_angle_right,
            'LeftArmAngle': left_arm_angle,
            'RightArmAngle': right_arm_angle,
            'ArmsTogetherAngle': arms_together_angle
        })

    # Crear un DataFrame a partir de la lista de diccionarios
    return pd.DataFrame(angles)

# Cálculos estadísticos sobre los ángulos
def calculos_estadísticos(df:pd.DataFrame) -> pd.DataFrame:
    """
    Realiza cálculos estadísticos sobre los ángulos en un
    DataFrame agrupado por sujeto, gesto y número de repetición.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene información sobre los ángulos,
        así como otras columnas relacionadas con el sujeto,
        gesto, repetición, etc.

    Returns
    -------
    pd.DataFrame
        DataFrame que contiene las estadísticas descriptivas
        calculadas para cada grupo de ángulos, con una fila por
        combinación de sujeto, gesto y repetición.
    """

    # Agrupa el DataFrame 
    groups = df.groupby(["SubjectID", "RepetitionNumber"])

    # Lista para almacenar los datos de salida
    data = []

    # Itera sobre cada grupo
    for (subject_id, repetition_number), group in groups:
        # Selecciona solo las columnas que contienen los ángulos 
        angles = group.iloc[:, 3:]

        # Calcula la media y la desviación estándar para los ángulos
        means = angles.mean()
        std_devs = angles.std()

        # Almacena las estadísticas en un diccionario 
        data.append({
            'SubjectID': subject_id,
            'RepetitionNumber': repetition_number,
            'Position': group['Position'].iloc[0],
            'Duration': len(group),  # Duración en número de frames
            'standardDeviation': std_devs,
            'Maximum': angles.max(),
            'Minimum': angles.min(),
            'Mean': means,
            'Range': angles.max() - angles.min(),
            'Variance': angles.var(),
            'CoV': std_devs / means,  # Coeficiente de variación
            'Skewness': angles.skew(),  # Asimetría
            'Kurtosis': angles.kurtosis()  # Curtosis
        })

    # Convierte la lista de diccionarios en un DataFrame y lo ordena
    df_stats = pd.DataFrame(data)
    df_stats['RepetitionNumber'] = pd.to_numeric(df_stats['RepetitionNumber'], errors='coerce')
    df_stats = df_stats.sort_values(['RepetitionNumber'])

    return df_stats


# Función para formatear columnas que contienen diccionarios
def formatear_columnas(columna:pd.Series, nombre_columna:str) -> pd.DataFrame: 
    '''
    Esta función toma una columna de un DataFrame que contiene diccionarios
    y expande cada diccionario en nuevas columnas, donde cada nueva columna
    corresponde a:  clave del diccionario + _ + nombre de la columna original. 

    Parámetro
    --------
    nombre_columna : str
        Nombre de la columna del DataFrame que contiene los diccionarios.

    Return
    --------
     Un nuevo DataFrame con columnas separadas para cada clave de los diccionarios.
    '''
    keys = set().union(*(d.keys() for d in columna))
    data = {}
    for key in keys:
        data[key + '_' + nombre_columna] = columna.apply(lambda x: x.get(key))
    return pd.DataFrame(data)

# Función para crear la gráfica
def repetition_graph(df:pd.DataFrame, keyPoint:str, movementAxis:str) -> plt.Figure:
    """
    Crea una gráfica en la que se representa la posición de un keypoint en función del tiempo (frame).
    Cada línea en la gráfica corresponde a una repetición distinta del gesto.

     La función realiza los siguientes pasos:
    1. Agrupa el DataFrame por el número de repetición (`RepetitionNumber`). Esto permite analizar
        cada repetición de manera individual.
    2. Itera a través de cada grupo (cada repetición). Dentro de cada grupo:
        - Extrae las posiciones del keypoint especificado (`keyPoint`) y de la base de la columna
        (`SpineBase`) en el eje de movimiento (`movementAxis`).
        - Normaliza las posiciones del keypoint restando las posiciones correspondientes de la base
        de la columna (`SpineBase`). Esto ajusta las posiciones del keypoint para centrarlas en
        relación con la base de la columna.
        - Grafica las posiciones normalizadas del keypoint a lo largo de los frames de la repetición.
    3. Configura las etiquetas del eje y y del eje x, así como los límites del eje y de la gráfica
    para mejorar la visualización.

    Parámetros
    --------
    df : DataFrame
        Datos de los keypoints, que debe incluir las columnas 'RepetitionNumber', 
               'JointName' y el eje de movimiento especificado.
    keyPoint : str
        Nombre del keypoint cuya posición se desea visualizar.
    movementAxis : str
        El eje del movimiento a analizar, puede ser '3D_X', '3D_Y', '3D_Z', '2D_X', o '2D_Y'.
    """
    df = df.apply(pd.to_numeric, errors='ignore')
    normalized_data = []

    groups = df.groupby("RepetitionNumber") # agrupa el dataframe según la repeticion
    for repetition , group in groups:
        # cogemos tambien la posicion de la base de la columna para poder normalizar los datos
        spinBase = group[group['JointName']=='SpineBase'][movementAxis]
        KeyPoint = group[group['JointName']==keyPoint][movementAxis]
        normalize_values = KeyPoint.values - spinBase.values
        
        normalized_df = pd.DataFrame({
            'Frame': range(len(normalize_values)),
            'Normalized_Position': normalize_values,
            'RepetitionNumber': repetition
        })
        # Append to the list of normalized data
        normalized_data.append(normalized_df)

    # Concatenate all normalized data into a single DataFrame for plotting
    normalized_df_all = pd.concat(normalized_data)


    # Create a Seaborn lineplot
    fig = plt.figure(figsize=(6,4))
    sns.lineplot(data=normalized_df_all, x='Frame', y='Normalized_Position',
                 hue='RepetitionNumber', palette='viridis', legend=False)

    # Customize the plot
    plt.ylabel(f"Posición en {movementAxis}")
    plt.xlabel("Frame number")
    plt.title(f"Movimiento de {keyPoint} ")
    
    return fig

# Función para crear la gráfica
def angle_graph(df:pd.DataFrame, angle:str):
  
    fig = plt.figure(figsize=(6,4))
    sns.lineplot(data=df, x=df.index, y=angle, hue='RepetitionNumber',
                    palette='viridis', legend=False)
    
    # Configurar etiquetas y título para cada subgráfico
    plt.ylabel(f"Ángulo ({angle})")
    plt.xlabel("Frame number")
    plt.title(f"Movimiento del ángulo {angle}")
    
    return fig


# Seleccionar el más común de una lista
def mas_comun(lista:list):
    lista = lista.tolist()
    return max(set(lista), key = lista.count)

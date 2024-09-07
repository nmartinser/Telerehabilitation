import streamlit as st
import pandas as pd
import  Notebooks.functions as functions
import joblib

# Función para extraer la información de dentro de los archivos
def leer_datos_archivo(lista_archivos:str, columnas:list[str]) -> pd.DataFrame:
     # lista para almacenar los datos extraídos
    list_data = []

    # Itera sobre cada archivo
    for uploaded_file in lista_archivos:
         # Extrae los datos del nombre del archivo
        campos = functions.leer_nombre_archivo(uploaded_file.name)

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

# Titulo para web
st.title('Tele-rehabilitacion')
st.markdown('Esta app sirve para monitorear nueve ejercicios de rehabililtación')
uploaded_files = st.file_uploader('Sube aquí tu ejercicio', accept_multiple_files=True,
                                  help='Sube un archivo.txt por cada repetición')

columnas = ['SubjectID', 'DateID', 'GestureLabel', 'RepetitionNumber', 'CorrectLabel', 'Position',
            'JointName', 'TrackedStatus', '3D_X', '3D_Y', '3D_Z', '2D_X', '2D_Y']

if uploaded_files:
    # Extraer la informacion y almacenarla en un DataFrame
    df_data = leer_datos_archivo(uploaded_files, columnas)
    # Eliminar columnas innceserarias
    df_data.drop(['DateID','GestureLabel', 'CorrectLabel', 'TrackedStatus', '2D_X', '2D_Y'], axis=1,
                inplace=True)
    fig_data = functions.repetition_graph(df_data, keyPoint='WristRight', movementAxis='3D_Y')

    st.pyplot(fig_data)

    # ------ Calcular los ángulos -----------

    df_angles = functions.apply_angles(df_data)
    fig_angles = functions.angle_graph(df_angles, 'RightArmAngle')
    st.pyplot(fig_angles)

    # ---- Cálculos estadisticos sobre los ángulos ----

    df_stats = functions.calculos_estadísticos(df_angles)

    columnas = ['standardDeviation', 'Maximum', 'Minimum', 'Mean', 'Range',
                'Variance', 'CoV', 'Skewness', 'Kurtosis']
    # Reformatea cada columna de diccionario y concatena los resultados
    nuevas_columnas = pd.concat([functions.formatear_columnas(df_stats[col], col) for col in columnas], axis=1)

    # Concatenar las nuevas columnas con el DataFrame original
    df_stats = pd.concat([df_stats, nuevas_columnas], axis=1)

    # Elimina las columnas originales que contenían diccionarios
    df_stats = df_stats.drop(columnas, axis=1)

    st.write(df_stats.head())

    # -------- Fase 1: Predecir el gesto -------
    st.write('Prediciendo el gesto...')
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

    gesture_name_mapping = {
        0: 'Flexión del codo izquierdo',
        1: 'Flexión del codo derecho',
        2: 'Flexión del hombro izquierdo',
        3: 'Flexión del hombro derecho',
        4: 'Abducción del hombro izquierdo',
        5: 'Abducción del hombro derecho',
        6: 'Elevación frontal del hombro',
        7: 'Toque lateral izquierdo',
        8: 'Toque lateral derecho'
    }

    gesture_short_name = gesture_mapping.get(gesture_label, 'Error en la predición')
    gesture_name = gesture_name_mapping.get(gesture_label, 'Error en la predición')

    st.write(f'Está realizando el gesto: {gesture_name}')

    # ---- Fase 2: Clasificación de la ejecución del movimiento ------
    st.write('Prediciendo si está bien ejecutado...')

    # Busca el archivo correspondiente al gesto predicho
    modelo_gesto_path = f'./Resultados/modelo_{gesture_short_name}.sav'

    correct_mapping = {
        1: 'correcta',
        2: 'incorrecta'
    }

    best_pipeline, expected_columns = joblib.load(modelo_gesto_path)
    df_stats = df_stats.reindex(columns=expected_columns)

    correct_labels = best_pipeline.predict(df_stats)

    for (i,correct_label) in enumerate(correct_labels):
        correct_name = correct_mapping.get(correct_label, 'Error en la predición')
        repetition_number = df_stats['RepetitionNumber'][i]
        st.write(f'La repetición {repetition_number} se ha ejecutado de forma: {correct_name}')

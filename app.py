import streamlit as st
import pandas as pd
import  Notebooks.functions as functions
import joblib


# Titulo para web
st.title('Tele-rehabilitacion')
st.markdown('Esta aplicación permite monitorear y evaluar la ejecución de nueve ejegesrcicios de rehabilitación mediante el análisis de datos de video.')
st.markdown('Para comenzar, carga los archivos de datos correspondientes a un ejercicio desde el menú desplegable.')
st.image("./Imagenes/gestures.png")
st.sidebar.title("Datos")
uploaded_files = st.sidebar.file_uploader('Sube aquí tu ejercicio', accept_multiple_files=True,
                                  help='Sube un archivo.txt por cada repetición')


st.sidebar.markdown("**Configuración de las gráficas**")

columnas = ['SubjectID', 'DateID', 'GestureLabel', 'RepetitionNumber', 'CorrectLabel', 'Position',
            'JointName', 'TrackedStatus', '3D_X', '3D_Y', '3D_Z', '2D_X', '2D_Y']

if uploaded_files:
    with st.spinner('Cargando los datos ...'):
        #----- Cargar los datos y preprocesar --------
        df_data = functions.leer_datos_archivo(uploaded_files, columnas)

        df_data.drop(['DateID','GestureLabel', 'CorrectLabel', 'TrackedStatus', '2D_X', '2D_Y'], axis=1,
                    inplace=True)
    with st.spinner('Calculando ángulos ...'):
        df_angles = functions.apply_angles(df_data)

    with st.spinner('Calculando estadísticas ...'):
        df_stats = functions.calculos_estadísticos(df_angles)

        columnas = ['standardDeviation', 'Maximum', 'Minimum', 'Mean', 'Range',
                    'Variance', 'CoV', 'Skewness', 'Kurtosis']

        nuevas_columnas = pd.concat([functions.formatear_columnas(df_stats[col], col) for col in columnas], axis=1)

        df_stats = pd.concat([df_stats, nuevas_columnas], axis=1)

        df_stats = df_stats.drop(columnas, axis=1)
    
    # -------- Fase 1: Predecir el gesto -------
    st.header('Predicciones')
    with st.spinner('Prediciendo el gesto...'):
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

    
        gesture_short_name = gesture_mapping.get(gesture_label, 0)
        gesture_name = gesture_name_mapping.get(gesture_label, 0)
        st.subheader('Está realizando el gesto: ')
        if gesture_short_name!=0:
            st.info(f'{gesture_name} ({gesture_short_name})')
        else:
            st.error("Error en la predición")
        
    # ---- Fase 2: Clasificación de la ejecución del movimiento ------
    with st.spinner('Prediciendo si las repeticiones están bien ejecutados...'):
        st.subheader('Ejecucción del movimiento por repetición: ')
        # Busca el archivo correspondiente al gesto predicho
        modelo_gesto_path = f'./Resultados/modelo_{gesture_short_name}.sav'

        correct_mapping = {
            1: 'correcta',
            2: 'incorrecta'
        }

        best_pipeline, expected_columns = joblib.load(modelo_gesto_path)
        df_stats = df_stats.reindex(columns=expected_columns)
        correct_labels = best_pipeline.predict(df_stats)

        label_per_column = len(correct_labels) // 2 + (len(correct_labels) % 2 > 0)
        columnas = st.columns(2)

        for i,correct_label in enumerate(correct_labels):
            correct_name = correct_mapping.get(correct_label, 'Error en la predición')
            repetition_number = df_stats['RepetitionNumber'][i]

            col_idx = i // label_per_column
            if correct_label==1:
                columnas[col_idx].success(f'Repetición {repetition_number}: {correct_name}')
            elif correct_label==2:
                columnas[col_idx].warning(f'Repetición {repetition_number}: {correct_name}')
            else:
                columnas[col_idx].error("Error en la predición")


    #-------- Gráficas ------
    joint_name = st.sidebar.selectbox('Elija el punto del cuerpo que quiere visualizar',
                                        df_data['JointName'].unique())

    angle_name = st.sidebar.selectbox('Elija el ángulo que quiere visualizar',
                                        df_angles.columns[3:])
    
    st.header('Gráficas')
    columns = st.columns(2)
    if joint_name:
        with columns[0]:
            st.markdown("**Representación de un punto del cuerpo por repetición**")
            fig_data = functions.repetition_graph(df_data, keyPoint=joint_name, movementAxis='3D_Y')
            st.pyplot(fig_data)

    if angle_name:
        with columns[1]:
            st.markdown("**Representación de un ángulo por repetición**")
            fig_angles = functions.angle_graph(df_angles, angle_name)
            st.pyplot(fig_angles)
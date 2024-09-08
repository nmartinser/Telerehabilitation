<h1 align="center"> Inteligencia Artificial aplicada a la Telerehabilitación </h1>

Este proyecto se centra en el desarrollo de modelos de aprendizaje automático para clasificar distintos gestos realizados por pacientes en un entorno de telerehabilitación, así como para discernir si estos gestos están correctamente ejecutados.

<p align="center">
  <img src="/Imagenes/gestures.png" width="400" title="Ejercicios rehabilitación">
</p>

## 📚 Tutorial

Para acceder a la aplicación web, haz clic en el siguiente enlace: [Tele-Rehabilitacion App](https://tele-rehabilitation-app-0bccfa1c526d.herokuapp.com/)

**Pasos para usar la aplicación:**

1. En la carpeta dataset de este repositorio, encontrarás los datos de los videos de un paciente, organizados por gestos. Cada subcarpeta contiene archivos .txt que corresponden a las repeticiones de un gesto específico.

2. Para usar la aplicación, descarga una de estas subcarpetas y carga los archivos .txt.

3. Una vez dentro de la aplicación, verás un menú desplegable donde puedes cargar los datos de los videos correspondientes a un movimiento.


## 📁 Descripción del repositorio

### 📓 Notebooks
En esta carpeta se encuentran los notebooks de jupyter (.ipynb). Contiene los siguientes archivos:

<details>
<summary>1. Procesar los datos de los videos</summary>
  
* Descripción: Este notebook procesa archivos de datos de video en formato crudo, extrayendo información esencial sobre cada grabación, como la ID del sujeto, el número de repetición, la precisión del gesto, y la posición de los puntos clave del cuerpo. Seguidamente calcula el ángulo entre disintos puntos del cuepro, y por último se realizan cálculos estadísticos (mínimo, máximo, desviación estándar, media, etc.) sobre los ángulos.  
* Salida: Genera tres archivos CSV:\
  _ *raw_pacientes.csv*: Contiene información detallada sobre cada grabación.\
  _ *angles.csv*: Incluye ángulos calculados entre keypoints.\
  _ *medidasPerRepetition.csv*: contiene una fila por repetición y gesto, que incluye estadísticas para cada ángulo calculado.

</details>

<details><summary>2. Análisis de los datos</summary>

* Descripción: análisis exploratorio de los datos procesados. Incluye visualizaciones como gráficos de barras para ver la distribución de sujetos por gesto y estado de ejecución, y gráficos de líneas para analizar los ángulos de los movimientos a lo largo del tiempo.
</details>

<details><summary>3. Fase 1: Clasificación del movimiento</summary>

* Descripción: Implementa, entrena y evalúa modelos de clasificación para identificar el tipo de gesto realizado por el paciente. Este notebook establece las bases para la clasificación de gestos en etapas posteriores.
* Salida: *modelo_fase1.sav*,  archivo que guarda el pipeline completo de clasificación entrenado, incluyendo tanto el preprocesamiento como el modelo final
</details>

<details><summary>4. Fase 2: Clasificación de la ejecución del movimiento</summary>

* Descripción: para cada tipo de gesto identificado en la Fase 1, se desarrollan modelos de clasificación separados, para determinar si un gesto es ejecutado de manera correcta o incorrecta.
* Salida:\
  _ Los resultados detallados del ajuste de modelos tras aplicar varias técnica de balanceo de datos se almacenan en el archivo *Results_imblearn.txt* \
  _ Nueve archivos *.sav*, uno para cada gesto, que almacenan el pipeline completo de clasificación entrenado, incluyendo tanto el preprocesamiento como el modelo final.
  

</details>

<p align="center">
  <img src="/Imagenes/esquema_modelos.png" width="600" title="Esquema fases">
</p>

### dataset

Esta carpeta contiene los datos de un paciente, organizados en subcarpetas separadas para cada gesto registrado durante las sesiones de telerehabilitación. Dentro de cada subcarpeta, se encuentra un archivo .txt por cada repetición, que almacena los datos extraídos de los videos.

### 📋 Resultados
Aquí se almacenan los archivos intermedios y resultados finales generados durante el procesamiento de datos y el entrenamiento de los modelos.

### ⚙️ Archivos de configuración

* *app.py*: programa principal de la aplicación.
 
* *.gitignore* y *.slugignore*: listas de archivos y carpetas que deben ser ignorados por el control de versiones o el despliegue.

* *setup.sh*, *Procfile* y *requirements.txt*: archivos necesarios para la instalación y despliegue de la aplicación.



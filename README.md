<h1 align="center"> Inteligencia Artificial aplicada a la Telerehabilitación </h1>

Este proyecto se centra en el desarrollo de modelos de aprendizaje automático para clasificar distintos gestos realizados por pacientes en un entorno de telerehabilitación, así como para discernir si estos gestos están correctamente ejecutados.

<p align="center">
  <img src="/Imagenes/gestures.png" width="400" title="Ejercicios rehabilitación">
</p>


## 📁 Descripción del repositorio

### 📓 Notebooks
En esta carpeta se encuentran los notebooks de jupyter (.ipynb). Contiene los siguientes archivos:

<details>
<summary>1. Procesar los datos de los videos</summary>
  
* Descripción: Este notebook procesa archivos de datos de video en formato crudo, extrayendo información esencial sobre cada grabación, como la ID del sujeto, el número de repetición, la precisión del gesto, y la posición de los puntos clave del cuerpo. También calcula ángulos entre estos puntos para un análisis posterior.
  
* Salida: Genera dos archivos CSV:\
  _ *raw_pacientes.csv*: Contiene información detallada sobre cada grabación.\
  _ *angles.csv*: Incluye ángulos calculados entre keypoints.
</details>

**2. Análisis de los datos**
* Descripción: análisis exploratorio de los datos procesados. Incluye visualizaciones como gráficos de barras para ver la distribución de sujetos por gesto y estado de ejecución, y gráficos de líneas para analizar los ángulos de los movimientos a lo largo del tiempo.

**3. Cálculos estadísticos sobre los ángulos**
* Descripción: Calcula estadísticas descriptivas (mínimo, máximo, desviación estándar, media, etc.) para los ángulos de los keypoints en cada repetición de los gestos. El resultado es un DataFrame que condensa esta información en una fila por repetición.
* Salida: medidasPerRepetition.csv``, archivo que contiene una fila por repetición y gesto, que incluye estadísticas para cada ángulo calculado.

**4. Fase 1: Clasificación del movimiento**
* Descripción: Implementa, entrena y evalúa modelos de clasificación para identificar el tipo de gesto realizado por el paciente. Este notebook establece las bases para la clasificación de gestos en etapas posteriores.

**5. Fase 2: Clasificación de la ejecución del movimiento**
* Descripción: para cada tipo de gesto identificado en la Fase 1, se desarrollan modelos de clasificación separados, para determinar si un gesto es ejecutado de manera correcta o incorrecta.

<p align="center">
  <img src="/Imagenes/esquema_modelos.png" width="600" title="Esquema fases">
</p>

****
### 📋 Resultados
Aquí es donde el código guarda los archivos intermediarios y los resultados finales generados durante el procesamiento y análisis.





<h1 align="center"> Inteligencia Artificial aplicada a la Telerehabilitación </h1>

## 📁 Descripción del repositorio

### 📓notebooks
En esta carpeta se encuentran los notebooks de jupyter (.ipynb). Contiene los siguientes archivos:
1. Procesar los datos de los videos
* Descripción: Este notebook procesa archivos de datos de video en formato crudo, extrayendo información esencial sobre cada grabación, como la ID del sujeto, el número de repetición, la precisión del gesto, y la posición de los puntos clave del cuerpo. También calcula ángulos entre estos puntos para un análisis posterior.
* Salida: Genera dos archivos CSV:
- raw_pacientes.csv: Contiene información detallada sobre cada grabación, utilizado en el análisis de datos.
- angles.csv: Incluye ángulos calculados entre keypoints, utilizado para análisis de postura y movimiento.

2. Análisis de los datos
* Descripción: Este notebook se enfoca en el análisis exploratorio de los datos procesados. Incluye visualizaciones como gráficos de barras para ver la distribución de sujetos por gesto y estado de ejecución, y gráficos de líneas para analizar los ángulos de los movimientos a lo largo del tiempo.

3. Cálculos estadísticos sobre los ángulos
* Descripción: Calcula estadísticas descriptivas (mínimo, máximo, desviación estándar, media, etc.) para los ángulos de los keypoints en cada repetición de los gestos. El resultado es un DataFrame que condensa esta información en una fila por repetición.
* Salida: Un DataFrame estructurado que facilita el análisis comparativo de las repeticiones y gestos, proporcionando métricas clave para la evaluación de la calidad de la ejecución.

![Esquema fases](/images/esquema_modelos.png)

## Descripción del proyecto
La tele-rehabilitación, facilitada por avances tecnológicos, ofrece una solución prometedora al permitir
que los pacientes realicen ejercicios desde casa, mejorando la accesibilidad y ahorrando tiempo tanto para
los pacientes como para el personal sanitario. Además, puede aumentar la motivación de los pacientes y
garantizar una mayor continuidad en el tratamiento. La evolución hacia un modelo donde la tecnología funcione
como una extensión del médico puede ayudar a reducir la sobrecarga de trabajo, y la Inteligencia Artificial
desempeña un papel crucial al permitir la evaluación de las actividades y el diseño de planes personalizados.

![Gestures](/images/gestures.png)


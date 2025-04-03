# Proyecto de Análisis de Fútbol

Una herramienta de visión por ordenador para el análisis en tiempo real de partidos de fútbol (enfocado en la Bundesliga), seguimiento de jugadores y visualización táctica. Este proyecto utiliza detección de objetos para mapear jugadores y el balón, además de usar un detector de poses para obtener una vista táctica 2D del campo de fútbol. Todo esto está hecho con YOLOv8.

## 🚀 Características

- **Detección y seguimiento de jugadores en tiempo real**: Identifica jugadores, diferencia entre equipos y realiza seguimiento del balón
- **Mapeo basado en homografía**: Proyecta las posiciones detectadas en una vista táctica 2D
- **Detección de puntos del campo**: Reconoce puntos clave en el campo para un mapeo preciso
- **Agrupamiento de equipos**: Diferencia automáticamente entre equipos utilizando aprendizaje automático basado en colores
- **Visualización interactiva**: Muestra simultáneamente el vídeo original y la vista táctica del minimapa
- **Grabación de vídeo**: Opción para guardar todas las visualizaciones generadas

## 📋 Requisitos previos

- Python 3.7+
- GPU compatible con CUDA (recomendado para un rendimiento óptimo)

## 🔧 Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/augustinbesu/Football-Analysis.git
cd Football-Analysis
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Descarga los pesos de los modelos:
   - Coloca el modelo de detección de puntos del campo en `models/field-keypoints-detection/weights/best.pt`
   - Coloca el modelo de detección de jugadores y balón en `models/players-ball-detection/weights/best.pt`

## 💻 Uso

### Herramienta de mapeo de puntos del campo

Antes de analizar vídeos, necesitas crear un mapeo entre los puntos clave del detector y las coordenadas del campo:

```bash
python field_mapping.py
```

Esta herramienta interactiva te permite:
- Hacer clic en los puntos del campo
- Asignar números de puntos clave del detector
- Guardar el mapeo en `point_mapping.json`

### Análisis de vídeo

Analiza grabaciones de fútbol con el siguiente comando:

```bash
python main.py --video ruta/al/video.mp4
```

#### Argumentos de línea de comandos

- `--video`: Ruta al archivo de vídeo
- `--pose_model`: Ruta al modelo de puntos clave del campo (por defecto: models/field-keypoints-detection/weights/best.pt)
- `--player_model`: Ruta al modelo de detección de jugadores (por defecto: models/players-ball-detection/weights/best.pt)
- `--mapping_file`: Ruta al archivo de mapeo de puntos (por defecto: point_mapping.json)
- `--save`: Guarda los vídeos de salida en la carpeta "results"

#### Ejemplos

Análisis básico:
```bash
python main.py --video videos/partido.mp4
```

Guardar todas las salidas en archivos:
```bash
python main.py --video videos/partido.mp4 --save
```

## 🖼️ Archivos de salida

Al utilizar la opción `--save`, se generan los siguientes archivos en la carpeta "results":

- `[nombre_video]_output.mp4`: Vídeo original con cajas de detección de jugadores
- `[nombre_video]_minimapa.mp4`: Vista táctica 2D con posiciones de jugadores
- `[nombre_video]_puntos.mp4`: Vídeo con puntos del campo superpuestos
- `[nombre_video]_mapeados.mp4`: Vista del campo con puntos detectados

## 🧪 Estructura del proyecto

- `main.py`: Script principal de análisis con la clase HomographyTester
- `field_mapping.py`: Herramienta interactiva para crear mapeos de puntos
- `utils.py`: Funciones de utilidad para visualización del campo y manejo de coordenadas
- `point_mapping.json`: Mapeo entre puntos clave del detector y coordenadas del campo
- `models`: Directorio que contiene los modelos de detección entrenados
- `results`: Directorio de salida para los vídeos guardados

## 🛠️ Cómo funciona

1. **Detección**: El sistema utiliza dos modelos YOLOv8: uno para la detección de puntos del campo y otro para la detección de jugadores/balón
2. **Homografía**: Se calcula una matriz de transformación de perspectiva utilizando los puntos del campo detectados
3. **Seguimiento**: Se realiza un seguimiento en tiempo real de jugadores y el balón mediante una combinación de algoritmos de detección y seguimiento
4. **Clasificación de equipos**: El agrupamiento K-means en colores de camisetas diferencia entre equipos
5. **Visualización**: Todas las detecciones se visualizan en el vídeo original y se mapean en una vista táctica del campo

## ⚙️ Detalles técnicos

### Detección de puntos del campo

El sistema identifica puntos específicos en el campo de fútbol como:
- Puntos de esquina (0-3)
- Puntos de la línea de medio campo (4-5)
- Puntos del círculo central (6-9)
- Puntos de esquina del área de penalti (10-13, 21-24)
- Puntos de esquina del área pequeña (16-19, 27-30)

La detección de estos puntos se realiza mediante un modelo YOLOv8 personalizado entrenado específicamente para reconocer elementos característicos del campo de fútbol. El modelo está optimizado para funcionar en diferentes condiciones de iluminación y ángulos de cámara.

### Mapeo de homografía

La homografía es una transformación matemática que permite convertir coordenadas entre diferentes planos o perspectivas. En este proyecto:

1. Se detectan puntos clave del campo en el vídeo (líneas, círculos, etc.)
2. Se establece una correspondencia entre estos puntos y sus ubicaciones en un modelo 2D del campo
3. Se calcula una matriz de homografía utilizando el algoritmo RANSAC para mayor robustez frente a outliers
4. Esta matriz se utiliza para transformar las posiciones de los jugadores y el balón del plano de la imagen al plano 2D del campo

El algoritmo implementa suavizado temporal de la matriz de homografía para evitar saltos bruscos en la visualización táctica.

### Agrupamiento de equipos

Los jugadores se asignan a equipos mediante el siguiente proceso:

1. Se extraen muestras de color del centro de los cuadros delimitadores de los jugadores
2. Se convierten al espacio de color HSV para una mejor diferenciación de colores
3. Se utiliza agrupamiento K-means para identificar los dos colores principales de equipo
4. Se asigna cada jugador al clúster de color de equipo más cercano

El sistema mantiene un modelo de color para cada equipo que se actualiza dinámicamente durante el análisis, lo que permite manejar cambios en la iluminación o en el ángulo de la cámara.

### Detección y seguimiento del balón

El balón se detecta utilizando un modelo YOLOv8 específico. Debido a su pequeño tamaño y rápido movimiento, se implementa:

1. Detección inicial con alta confianza
2. Seguimiento mediante el algoritmo CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
3. Restablecimiento del seguimiento cuando la detección se pierde por un número determinado de frames

### Optimización de rendimiento

El sistema equilibra precisión y rendimiento mediante:
- Configuración de umbrales de confianza para detecciones
- Procesamiento selectivo de regiones de interés
- Implementación de algoritmos de seguimiento eficientes
- Paralelización de tareas donde es posible

## 📊 Métodos implementados

### Clase HomographyTester

La clase principal que coordina todo el proceso de análisis:

- **__init__(pose_model_path, player_model_path, video_path, json_path, show_mapped=False, save_video=False)**: 
  - Inicializa modelos de YOLO para detección de puntos del campo y jugadores
  - Carga el video de entrada y el mapeo de puntos desde un archivo JSON
  - Configura parámetros del campo (dimensiones, factor de escala)
  - Inicializa sistemas de seguimiento para el balón y clasificación de equipos
  - Configura opciones de visualización y grabación basadas en parámetros
  - Si `save_video=True`, crea los VideoWriters necesarios para guardar todas las salidas

- **_scale_field_coordinates()**:
  - Escala las coordenadas del campo cargadas desde el JSON al tamaño actual del campo de visualización
  - Aplica factores de escala manteniendo proporciones y márgenes

- **get_field_coordinates()**:
  - Obtiene las coordenadas de los puntos clave del campo para el minimapa
  - Usa coordenadas predefinidas o las calcula mediante FieldUtils

- **create_field_image()**:
  - Genera una imagen base del campo de fútbol con líneas y marcas

- **get_mapped_points(keypoints)**:
  - Mapea puntos detectados por el modelo a puntos del campo usando un umbral de confianza
  - Filtra detecciones de baja confianza y retorna una lista de puntos mapeados

- **calculate_homography(mapped_points)**:
  - Calcula la matriz de homografía usando RANSAC para mayor robustez
  - Aplica suavizado temporal para evitar fluctuaciones entre frames
  - Maneja casos donde no hay suficientes puntos o la homografía no se puede calcular

- **transform_player_coordinates(player_points, H)**:
  - Transforma coordenadas de jugadores del plano de la imagen al minimapa
  - Valida y corrige puntos transformados para asegurar que estén dentro del campo
  - Aplica recorte ("clipping") para puntos cercanos a los bordes

- **create_field_points_view(mapped_points)**:
  - Crea visualización de puntos del campo sobre el frame original
  - Asigna colores según el tipo de punto (perímetro, círculo central, etc.)
  - Muestra etiquetas con índices y valores de confianza

- **create_mapped_points_view(mapped_points)**:
  - Crea visualización de puntos mapeados directamente en el minimapa
  - Usa el mismo esquema de colores que field_points_view para consistencia

- **process_frame(frame)**:
  - Procesa cada fotograma del vídeo realizando:
    - Detección de puntos clave del campo
    - Cálculo de homografía
    - Detección de jugadores, árbitros y balón
    - Clasificación de equipos mediante clustering
    - Seguimiento del balón cuando no es detectado
    - Transformación de coordenadas al espacio del minimapa
  - Retorna los fotogramas procesados para visualización y grabación

- **run()**:
  - Bucle principal para procesamiento de vídeo
  - Maneja la visualización de diferentes vistas
  - Si `show_mapped=True`, crea una vista superpuesta del campo sobre el vídeo
  - Si `save_video=True`, guarda todos los fotogramas procesados
  - Gestiona controles de usuario (pausa, avance, salida)

- **cluster_teams(frame, player_boxes, min_confidence=0.7)**:
  - Separa jugadores en equipos mediante análisis de color y K-means
  - Extrae regiones centrales de los jugadores para muestreo de color
  - Convierte muestras a espacio HSV para mejor diferenciación
  - Mantiene y actualiza modelos de color para cada equipo
  - Asigna jugadores a equipos basándose en la similitud de color

### Clase FieldUtils

Clase de utilidad para manejar la visualización y coordenadas del campo:

- **__init__(field_width=800, margin=50)**:
  - Inicializa las dimensiones del campo y márgenes
  - Configura proporciones según estándares FIFA

- **create_field_image()**:
  - Genera una representación visual del campo con:
    - Fondo verde con patrón de rayas
    - Líneas blancas para perímetro y divisiones
    - Círculo central y áreas de penalti
    - Puntos de penalti y semicírculos

- **get_field_coordinates()**:
  - Calcula y retorna coordenadas precisas de todos los puntos clave:
    - Esquinas del campo
    - Intersecciones de líneas centrales
    - Puntos del círculo central
    - Áreas grandes y pequeñas
    - Puntos de penalti

- **_draw_field_lines(field)**:
  - Dibuja todas las líneas principales del campo
  - Usa proporciones estándar para dimensiones de áreas y círculos

- **_draw_penalty_arcs(field, margin, mid_y, big_box_width, circle_radius)**:
  - Dibuja los arcos de las áreas de penalti con geometría precisa

- **_draw_arc_outside_box(field, center, radius, box_x, is_left)**:
  - Dibuja arcos con geometría correcta, calculando ángulos de inicio y fin
  - Maneja diferentes casos para arcos izquierdo y derecho

### Clase PointMapperApp (en field_mapping.py)

Interfaz gráfica para el mapeo de puntos:

- **__init__(root)**:
  - Inicializa la interfaz gráfica de Tkinter
  - Configura el canvas para visualización del campo
  - Inicializa estructuras de datos para mapeo de puntos

- **draw_field()**:
  - Dibuja el campo en el canvas con todas las líneas y marcas

- **draw_points()**:
  - Dibuja los puntos disponibles para mapeo
  - Usa colores diferentes para puntos ya mapeados

- **on_click(event)**:
  - Maneja eventos de clic para seleccionar puntos
  - Actualiza el mapeo basado en la entrada del usuario

- **save_mapping()**:
  - Guarda el mapeo actual a la estructura de datos interna
  - Actualiza la visualización para reflejar cambios

- **export_json()**:
  - Exporta el mapeo completo a formato JSON
  - Incluye coordenadas y metadatos del campo

## 🔍 Aplicaciones prácticas

- **Análisis táctico**: Visualizar formaciones y movimientos de equipo
- **Análisis de rendimiento**: Seguimiento de distancias recorridas por jugadores
- **Detección de patrones**: Identificar estrategias comunes y movimientos de juego
- **Análisis de espacios**: Evaluar la ocupación del campo y creación de espacios
- **Estadísticas avanzadas**: Generar métricas sobre posesión zonal y presión

## 📝 Licencia

Licencia MIT

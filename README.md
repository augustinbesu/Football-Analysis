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

### 3. Modelos Preentrenados  

Los siguientes modelos ya han sido entrenados y están disponibles:  

- **Detector de poses**: [`models/field-keypoints-detection/weights/best.pt`](models/field-keypoints-detection/weights/best.pt)  
- **Detector de jugadores y pelota**: [`models/players-ball-detection/weights/best.pt`](models/players-ball-detection/weights/best.pt)  

Si deseas entrenar tus propios modelos, puedes utilizar los conjuntos de datos disponibles en las carpetas `football-dataset` y `keypoints-field`, ambos descargados desde [Roboflow](https://roboflow.com/).


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

## 🛠️ Cómo funciona

### Detección de puntos del campo

El sistema identifica puntos específicos en el campo de fútbol como:
- Puntos de esquina (0-3)
- Puntos de la línea de medio campo (4-5)
- Puntos del círculo central (6-9)
- Puntos de esquina del área de penalti (10-13, 21-24)
- Puntos de esquina del área pequeña (16-19, 27-30)

La detección de estos puntos se realiza mediante un modelo YOLOv8 entrenado específicamente para reconocer elementos característicos del campo de fútbol. El modelo está optimizado para funcionar en diferentes condiciones de iluminación y ángulos de cámara.

Cada punto detectado incluye un valor de confianza (0.0-1.0) que representa la certeza del modelo. Se aplica un umbral de confianza fijo de 0.5 para filtrar detecciones poco fiables.

El sistema también guarda información sobre cuándo se vio cada punto por última vez mediante el diccionario `point_last_seen`. Esto permite identificar si un punto está siendo directamente detectado o si se está utilizando su última posición conocida.

### Mapeo de homografía

La homografía permite convertir coordenadas entre el plano de la imagen del vídeo y una representación 2D del campo. En este proyecto:

1. Se detectan puntos clave del campo en el vídeo
2. Se establece una correspondencia entre estos puntos y sus ubicaciones en el modelo 2D del campo
3. Se calcula una matriz de homografía utilizando el algoritmo RANSAC para mayor robustez frente a outliers
4. Esta matriz se utiliza para transformar las posiciones de los jugadores y el balón del plano de la imagen al plano 2D del campo

El algoritmo implementa suavizado temporal simple (70% de la matriz anterior + 30% de la nueva) para evitar saltos bruscos en la visualización táctica.

### Agrupamiento de equipos

Los jugadores se asignan a equipos mediante el siguiente proceso:

1. Se extraen muestras de color del centro de los cuadros delimitadores de los jugadores
2. Se convierten al espacio de color HSV para una mejor diferenciación de colores
3. Se utiliza agrupamiento K-means para identificar los dos colores principales de equipo
4. Se asigna cada jugador al clúster de color de equipo más cercano

**Optimizaciones en el muestreo de color:**
En lugar de analizar todo el bounding box de cada jugador (lo que sería computacionalmente costoso), el código extrae una región central y reducida:

```python
# Calcular el centro del bounding box
center_x = x1 + width // 2
center_y = y1 + height // 3  # Un poco más arriba del centro para capturar mejor la camiseta

# Definir una región del 50% alrededor del centro
sample_width = width // 2  # 50% del ancho
sample_height = height // 2  # 50% del alto
```

Esta optimización permite un procesamiento mucho más rápido sin sacrificar la precisión en la clasificación de equipos, ya que la parte central del bounding box generalmente contiene los colores más representativos del uniforme.

El sistema acumula muestras de color hasta alcanzar un mínimo de 20 muestras antes de inicializar los colores de referencia de cada equipo. Una vez establecidos estos colores de referencia, se utilizan para clasificar nuevos jugadores detectados según la similitud de color.

### Detección y seguimiento del balón

El balón se detecta utilizando un modelo YOLOv8 específico. Debido a su pequeño tamaño y rápido movimiento, se implementa:

1. Detección inicial con alta confianza
2. Seguimiento mediante el algoritmo CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
3. Restablecimiento del seguimiento cuando la detección se pierde por más de 30 frames

El código utiliza `cv2.TrackerCSRT_create()` para inicializar el seguimiento del balón cuando se detecta con alta confianza, y luego mantiene el seguimiento incluso cuando no hay detecciones nuevas:

```python
if not ball_detected:  # Solo usar tracking si no hay detección
    success, bbox = self.ball_tracker.update(frame)
    if success:
        # Actualizar posición mediante tracking
    else:
        self.tracking_lost_frames += 1
        if self.tracking_lost_frames > 30:  # Reset después de 30 frames
            self.is_tracking_ball = False
```

### Visualización del campo

El sistema genera varias visualizaciones:

1. **Display Frame**: Muestra el vídeo original con cajas delimitadoras para jugadores, árbitros y el balón
2. **Field View (Minimapa)**: Muestra una vista táctica 2D del campo con las posiciones mapeadas
3. **Points View**: Muestra los puntos clave del campo detectados sobre el frame original
4. **Mapped View**: Muestra los puntos clave mapeados en la representación 2D del campo

Los colores se usan de manera consistente en todas las visualizaciones:
- Equipo 1: Azul
- Equipo 2: Rojo
- Árbitros: Rosa
- Balón: Verde con contorno negro (minimapa) o amarillo (frame original)
- Puntos del perímetro: Rojo
- Puntos del círculo central: Azul
- Otros puntos: Verde

## 📊 Componentes principales

### Clase HomographyTester

La clase principal que coordina todo el proceso de análisis:

- **__init__(pose_model_path, player_model_path, video_path, json_path, save_output=False)**: 
  - Inicializa modelos de YOLO para detección de puntos del campo y jugadores
  - Carga el video de entrada y el mapeo de puntos desde el archivo JSON
  - Configura parámetros del campo (dimensiones, factor de escala)
  - Inicializa sistemas de seguimiento para el balón y clasificación de equipos

- **get_mapped_points(keypoints)**:
  - Mapea puntos detectados por el modelo a puntos del campo usando un umbral de confianza
  - Filtra detecciones de baja confianza y retorna una lista de puntos mapeados

- **calculate_homography(mapped_points)**:
  - Calcula la matriz de homografía usando RANSAC para mayor robustez
  - Aplica suavizado temporal para evitar fluctuaciones entre frames

- **transform_player_coordinates(player_points, H)**:
  - Transforma coordenadas de jugadores del plano de la imagen al minimapa
  - Valida y corrige puntos transformados para asegurar que estén dentro del campo

- **create_field_points_view(frame, mapped_points)**:
  - Crea visualización de puntos del campo sobre el frame original
  - Asigna colores según el tipo de punto y muestra valores de confianza

- **process_frame(frame)**:
  - Procesa cada fotograma realizando la detección de puntos, jugadores y balón
  - Maneja la clasificación de equipos y el seguimiento del balón
  - Genera las diferentes visualizaciones

- **run()**:
  - Bucle principal para procesamiento de vídeo
  - Maneja la visualización y grabación de resultados
  - Controla la interfaz de usuario (pausa, avance, salida)

- **cluster_teams(frame, player_boxes, min_confidence=0.7)**:
  - Separa jugadores en equipos mediante análisis de color y K-means
  - Extrae regiones centrales de los jugadores para muestreo optimizado
  - Acumula muestras antes de inicializar los colores de referencia

### Clase FieldUtils

Clase de utilidad para manejar la visualización y coordenadas del campo:

- **create_field_image()**:
  - Genera una representación visual del campo con líneas y marcas

- **get_field_coordinates()**:
  - Calcula y retorna coordenadas precisas de todos los puntos clave del campo

## 🔍 Aplicaciones prácticas

- **Análisis táctico**: Visualizar formaciones y movimientos de equipo
- **Análisis de rendimiento**: Seguimiento de distancias recorridas por jugadores
- **Detección de patrones**: Identificar estrategias comunes y movimientos de juego
- **Análisis de espacios**: Evaluar la ocupación del campo y creación de espacios
- **Estadísticas avanzadas**: Generar métricas sobre posesión zonal y presión

## 📝 Licencia

Licencia MIT

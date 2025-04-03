import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
import argparse
from utils import FieldUtils
from sklearn.cluster import KMeans

class HomographyTester:
    def __init__(self, pose_model_path, player_model_path, video_path, json_path, save_output=False):
        # Cargar modelos
        self.pose_model = YOLO(pose_model_path)
        self.player_model = YOLO(player_model_path)
        
        # Cargar el video
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Opciones de visualización y guardado
        self.save_output = save_output
        self.video_writers = {}
        
        # Valores por defecto si no hay información del campo
        self.original_field_width = 800
        self.original_field_height = 517
        self.original_margin = 50
        self.real_field_width = 105
        self.real_field_height = 68
        
        # Cargar mapping de puntos y coordenadas
        with open(json_path, 'r') as f:
            data = json.load(f)
            
            # Extraer el mapeo de puntos (detector-field)
            if "mapping" in data:
                # Nuevo formato con separación de mapeo y coordenadas
                self.point_mapping = {int(k): v for k, v in data["mapping"].items()}
                
                # Cargar las coordenadas del campo
                if "coordinates" in data:
                    self.field_coordinates = {int(k): tuple(v) for k, v in data["coordinates"].items()}
                    
                    # Cargar dimensiones del campo si están disponibles
                    if "field_dimensions" in data:
                        self.original_field_width = data["field_dimensions"]["canvas_width"]
                        self.original_field_height = data["field_dimensions"]["canvas_height"]
                        self.original_margin = data["field_dimensions"]["margin"]
                        self.real_field_width = data["field_dimensions"]["width"]  # en metros
                        self.real_field_height = data["field_dimensions"]["height"]  # en metros
                else:
                    self.field_coordinates = None
            else:
                # Formato antiguo (solo mapeo)
                self.point_mapping = {int(k): v for k, v in data.items()}
                self.field_coordinates = None
        
        # Configuración del campo
        self.scale_factor = 6  # píxeles por metro
        self.field_width = int(self.real_field_width * self.scale_factor)
        self.field_height = int(self.real_field_height * self.scale_factor)
        self.margin = 20
        
        # Obtener dimensiones del frame
        ret, frame = self.cap.read()
        if ret:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.frame_height, self.frame_width = 720, 1280
        
        # AHORA configurar los video writers (después de tener las dimensiones del campo)
        if self.save_output:
            # Crear directorio para resultados si no existe
            os.makedirs("results", exist_ok=True)
            
            # Obtener fps y dimensiones del video original
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Extraer nombre de archivo del video de entrada
            video_name = os.path.basename(video_path).split('.')[0]
            
            # Crear VideoWriters para cada salida
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writers['video'] = cv2.VideoWriter(
                f"results/{video_name}_output.mp4", fourcc, fps, (self.frame_width, self.frame_height))
            self.video_writers['minimapa'] = cv2.VideoWriter(
                f"results/{video_name}_minimapa.mp4", fourcc, fps, 
                (self.field_width, self.field_height))
            self.video_writers['puntos'] = cv2.VideoWriter(
                f"results/{video_name}_puntos.mp4", fourcc, fps, (self.frame_width, self.frame_height))
            self.video_writers['mapeados'] = cv2.VideoWriter(
                f"results/{video_name}_mapeados.mp4", fourcc, fps, 
                (self.field_width, self.field_height))
        
        # Si tenemos coordenadas, necesitamos escalarlas al tamaño actual
        if self.field_coordinates:
            self._scale_field_coordinates()
        
        # Para homografía y filtrado
        self.H = None
        self.prev_H = None
        self.min_valid_points = 4

        # Inicializar FieldUtils
        self.field_utils = FieldUtils(field_width=self.field_width,
                                    margin=self.margin)

        # Para tracking de la pelota
        self.ball_tracker = cv2.TrackerCSRT_create()  # CSRT es bueno para objetos pequeños y rápidos
        self.is_tracking_ball = False
        self.ball_bbox = None
        self.tracking_lost_frames = 0
        self.max_lost_frames = 30  # Máximo número de frames antes de reiniciar el tracking

        # Para la clasificación de equipos
        self.team_colors = None  # Almacenará los colores de referencia de cada equipo
        self.team_samples = []   # Almacenará muestras de colores para inicialización
        self.min_samples_init = 20  # Mínimo de muestras para inicializar los colores de equipo
        
        # Para puntos predichos
        self.point_last_seen = {}
        self.frame_count = 0

    # El resto del código permanece igual
    def _scale_field_coordinates(self):
        """Escala las coordenadas del JSON al tamaño actual del campo"""
        if not self.field_coordinates:
            return
        
        # Factor de escala para las coordenadas internas (sin márgenes)
        scale_x = (self.field_width - 2*self.margin) / (self.original_field_width - 2*self.original_margin)
        scale_y = (self.field_height - 2*self.margin) / (self.original_field_height - 2*self.original_margin)
        
        # Escalar cada punto
        for idx, (x, y) in self.field_coordinates.items():
            # Normalizar el punto (quitar margen original)
            norm_x = (x - self.original_margin)
            norm_y = (y - self.original_margin)
            
            # Escalar y aplicar nuevo margen
            new_x = (norm_x * scale_x) + self.margin
            new_y = (norm_y * scale_y) + self.margin
            
            # Guardar coordenadas escaladas como enteros
            self.field_coordinates[idx] = (int(new_x), int(new_y))

    def get_field_coordinates(self):
        """Obtiene las coordenadas de los puntos del campo en el minimapa"""
        # Si ya tenemos las coordenadas cargadas y escaladas del JSON, usarlas
        if self.field_coordinates is not None:
            return self.field_coordinates
        
        # Si no, obtenerlas de FieldUtils
        return self.field_utils.get_field_coordinates()

    def create_field_image(self):
        """Crear imagen base del campo"""
        return self.field_utils.create_field_image()

    def get_mapped_points(self, keypoints):
        """Mapear puntos del detector al campo con umbral de confianza fijo"""
        mapped_points = []
        confidence_threshold = 0.5  # Umbral fijo
        
        for field_point, detector_point in self.point_mapping.items():
            try:
                field_idx = int(field_point)
                detector_idx = int(detector_point)
                
                if detector_idx < len(keypoints):
                    point = keypoints[detector_idx]
                    if point[2] > confidence_threshold:
                        mapped_points.append((field_idx, (point[0], point[1]), point[2]))
            except:
                continue
                
        return mapped_points

    def calculate_homography(self, mapped_points):
        """Calcula homografía usando RANSAC para robustez"""
        if len(mapped_points) < self.min_valid_points:
            return self.prev_H if self.prev_H is not None else None
        
        field_coordinates = self.get_field_coordinates()
        src_points = []  # Puntos en el frame
        dst_points = []  # Puntos correspondientes en el minimapa
        
        # Recolectar puntos
        for field_idx, point, _ in mapped_points:
            if field_idx in field_coordinates:
                src_points.append([point[0], point[1]])
                dst_points.append(field_coordinates[field_idx])
        
        if len(src_points) < 4:
            return self.prev_H if self.prev_H is not None else None
        
        try:
            # Calcular homografía con RANSAC
            H, _ = cv2.findHomography(
                np.float32(src_points),
                np.float32(dst_points),
                cv2.RANSAC,
                5.0  # Umbral de error en píxeles
            )
            
            if H is None:
                return self.prev_H if self.prev_H is not None else None
            
            # Aplicar suavizado temporal simple
            if self.prev_H is not None:
                H = 0.7 * self.prev_H + 0.3 * H
            
            self.prev_H = H.copy()
            return H
        
        except Exception as e:
            print(f"Error en homografía: {e}")
            return self.prev_H if self.prev_H is not None else None

    def transform_player_coordinates(self, player_points, H):
        """Transforma coordenadas de jugadores con validación mejorada"""
        if len(player_points) == 0 or H is None:
            return []
        
        try:
            # Convertir a formato homogéneo
            points = np.float32(player_points).reshape(-1, 1, 2)
            
            # Aplicar la transformación
            transformed_points = cv2.perspectiveTransform(points, H)
            transformed_points = transformed_points.reshape(-1, 2)
            
            # Validar y filtrar puntos transformados
            valid_points = []
            for i, point in enumerate(transformed_points):
                # Verificar que el punto está dentro de los límites del campo
                if (self.margin <= point[0] <= self.field_width - self.margin and
                    self.margin <= point[1] <= self.field_height - self.margin):
                    valid_points.append(point)
                else:
                    # Intentar corregir puntos cercanos al borde
                    point[0] = np.clip(point[0], self.margin, self.field_width - self.margin)
                    point[1] = np.clip(point[1], self.margin, self.field_height - self.margin)
                    valid_points.append(point)
            
            return np.array(valid_points)
        except Exception as e:
            print(f"Error en transformación: {e}")
            return []

    def create_field_points_view(self, frame, mapped_points):
        """
        Crear visualización de puntos del campo directamente sobre el frame de video
        """
        # Copiar el frame original para no modificarlo
        points_view = frame.copy()
        
        # Dibujar todos los puntos mapeados encima del frame
        for idx, point, conf in mapped_points:
            is_predicted = idx in self.point_last_seen and (self.frame_count - self.point_last_seen[idx]) > 0
            
            # Color según tipo de punto
            if idx in range(0, 6):
                base_color = (0, 0, 255)  # Rojo (perímetro)
            elif idx in range(6, 10):
                base_color = (255, 0, 0)  # Azul (círculo central)
            else:
                base_color = (0, 255, 0)  # Verde (otros puntos)
            
            radius = int(5 + 5 * conf)
            
            if is_predicted:
                cv2.circle(points_view, (int(point[0]), int(point[1])), radius, base_color, 2)
                cv2.circle(points_view, (int(point[0]), int(point[1])), 2, base_color, -1)
                label = f"{idx} (pred {conf:.2f})"
            else:
                cv2.circle(points_view, (int(point[0]), int(point[1])), radius, base_color, -1)
                label = f"{idx} ({conf:.2f})"
            
            cv2.putText(points_view, label, 
                       (int(point[0]) + 10, int(point[1]) + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return points_view

    def create_mapped_points_view(self, mapped_points):
        """Crear visualización de puntos mapeados directamente en el minimapa"""
        # Crear imagen base del campo
        mapped_view = self.create_field_image()
        
        # Obtener coordenadas del campo
        field_coordinates = self.get_field_coordinates()
        
        # Para cada punto detectado, dibujarlo en su posición correspondiente en el campo
        for field_idx, _, conf in mapped_points:
            if field_idx in field_coordinates:
                # Obtener la posición real en el minimapa para este punto
                map_x, map_y = field_coordinates[field_idx]
                
                # Color según tipo de punto (igual que en points_view)
                if field_idx in range(0, 6):
                    color = (0, 0, 255)  # Rojo (perímetro)
                elif field_idx in range(6, 10):
                    color = (255, 0, 0)  # Azul (círculo central)
                else:
                    color = (0, 255, 0)  # Verde (otros puntos)
                
                # Dibujar punto
                radius = int(5 + 5 * conf)
                cv2.circle(mapped_view, (int(map_x), int(map_y)), radius, color, -1)
                
                # Añadir etiqueta con índice y confianza
                label = f"{field_idx} ({conf:.2f})"
                cv2.putText(mapped_view, label,
                           (int(map_x) + 10, int(map_y) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return mapped_view

    def process_frame(self, frame):
        """Versión simplificada del procesamiento de frame"""
        self.frame_count += 1
        display_frame = frame.copy()
        field_view = self.create_field_image()
        mapped_view = self.create_field_image()
        
        # Detectar keypoints
        pose_results = self.pose_model(frame)[0]
        
        # Verificar detecciones válidas
        has_valid_detections = (pose_results.keypoints is not None and 
                               len(pose_results.keypoints.data) > 0 and 
                               len(pose_results.boxes.conf) > 0)
        
        mapped_points = []
        
        if has_valid_detections:
            # Probar todas las detecciones y quedarse con la que tenga más puntos válidos
            best_mapped_points = []
            for detection_idx in range(len(pose_results.boxes)):
                keypoints = pose_results.keypoints.data[detection_idx].cpu().numpy()
                current_mapped_points = self.get_mapped_points(keypoints)
                
                # Si encontramos una detección con más puntos, la guardamos
                if len(current_mapped_points) > len(best_mapped_points):
                    best_mapped_points = current_mapped_points
            
            mapped_points = best_mapped_points
        
        # Crear vista de puntos sobre el frame original
        points_view = self.create_field_points_view(frame, mapped_points)
        
        # Dibujar puntos en mapped_view
        for idx, point, conf in mapped_points:
            # Actualizar cuándo se vio este punto por última vez
            self.point_last_seen[idx] = self.frame_count
            
            # Dibujar en mapped_view
            field_coordinates = self.get_field_coordinates()
            if idx in field_coordinates:
                map_x, map_y = field_coordinates[idx]
                
                # Color según tipo de punto
                if idx in range(0, 6):
                    color = (0, 0, 255)  # Rojo (perímetro)
                elif idx in range(6, 10):
                    color = (255, 0, 0)  # Azul (círculo central)
                else:
                    color = (0, 255, 0)  # Verde (otros puntos)
                
                radius = int(5 + 5 * conf)
                cv2.circle(mapped_view, (int(map_x), int(map_y)), radius, color, -1)
                label = f"{idx} ({conf:.2f})"
                cv2.putText(mapped_view, label,
                           (int(map_x) + 10, int(map_y) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Calcular homografía y procesar jugadores/pelota
        if len(mapped_points) >= self.min_valid_points:
            self.H = self.calculate_homography(mapped_points)
        
        # Detectar jugadores y pelota
        player_results = self.player_model(frame)[0]
        if player_results.boxes is not None:
            player_positions = []
            ball_positions = []
            referee_positions = []  # Nueva lista para árbitros
            min_confidence = 0.7
            
            ball_detected = False
            ball_confidence = 0.0
            ball_box = None
            
            # Recolectar todos los boxes de jugadores (no pelotas)
            player_boxes = []
            for box, cls, conf in zip(player_results.boxes.xyxy, 
                                    player_results.boxes.cls, 
                                    player_results.boxes.conf):
                
                box = box.cpu().numpy()
                cls_idx = int(cls.item())
                confidence = float(conf.item())
                
                if confidence < min_confidence:
                    continue
                
                class_name = player_results.names[cls_idx]
                
                if class_name.lower() == 'ball':
                    ball_detected = True
                    ball_confidence = confidence
                    ball_box = box
                    ball_positions.append([
                        (box[0] + box[2]) / 2,  # x centro
                        box[3]  # y abajo
                    ])
                    
                    # Dibujar la pelota detectada
                    cv2.rectangle(display_frame, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                (0, 255, 255), 2)  # Amarillo
                    cv2.putText(display_frame, f"Ball {confidence:.2f}",
                               (int(box[0]), int(box[1] - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Inicializar tracker solo para la pelota
                    if confidence > 0.7:
                        bbox = (
                            int(box[0]),  # x
                            int(box[1]),  # y
                            int(box[2] - box[0]),  # width
                            int(box[3] - box[1])   # height
                        )
                        self.ball_tracker = cv2.TrackerCSRT_create()
                        self.ball_tracker.init(frame, bbox)
                        self.is_tracking_ball = True
                        self.ball_bbox = box
                        self.tracking_lost_frames = 0
                elif class_name.lower() == 'referee':  # Nuevo caso para árbitros
                    referee_positions.append((
                        [
                            (box[0] + box[2]) / 2,  # x centro
                            box[3]  # y abajo
                        ], 
                        box
                    ))
                    # Dibujar árbitro en rosa
                    cv2.rectangle(display_frame, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                (255, 192, 203), 2)  # Rosa
                    cv2.putText(display_frame, f"Referee {confidence:.2f}",
                               (int(box[0]), int(box[1] - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 2)
                else:
                    player_boxes.append((box, confidence))
            
            # Actualizar tracking de la pelota si está activo
            if self.is_tracking_ball:
                if not ball_detected:  # Solo usar tracking si no hay detección
                    success, bbox = self.ball_tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        # Dibujar el tracking en el frame
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h),
                                    (0, 255, 0), 2)  # Verde para tracking
                        cv2.putText(display_frame, "Ball (Tracked)",
                                   (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # Actualizar posición de la pelota para el minimapa
                        ball_positions.append([
                            x + w/2,  # x centro
                            y + h  # y abajo
                        ])
                    else:
                        self.tracking_lost_frames += 1
                        if self.tracking_lost_frames > 30:  # Reset después de 30 frames
                            self.is_tracking_ball = False
            
            # Separar jugadores en equipos (excluyendo árbitros)
            team1_boxes, team2_boxes = self.cluster_teams(frame, player_boxes)
            
            # Procesar equipo 1 (azul)
            for box, conf in team1_boxes:
                foot_x = (box[0] + box[2]) / 2
                foot_y = box[3]
                player_positions.append(([foot_x, foot_y], 1))  # 1 para equipo 1
                
                # Dibujar en azul
                cv2.rectangle(display_frame, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            (255, 0, 0), 2)  # Azul
                cv2.putText(display_frame, f"Team 1 {conf:.2f}",
                           (int(box[0]), int(box[1] - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Procesar equipo 2 (rojo)
            for box, conf in team2_boxes:
                foot_x = (box[0] + box[2]) / 2
                foot_y = box[3]
                player_positions.append(([foot_x, foot_y], 2))  # 2 para equipo 2
                
                # Dibujar en rojo
                cv2.rectangle(display_frame, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            (0, 0, 255), 2)  # Rojo
                cv2.putText(display_frame, f"Team 2 {conf:.2f}",
                           (int(box[0]), int(box[1] - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Transformar posiciones usando la homografía
            if self.H is not None:
                # Transformar jugadores de equipos
                if player_positions:
                    positions = [pos for pos, _ in player_positions]
                    teams = [team for _, team in player_positions]
                    transformed_players = self.transform_player_coordinates(positions, self.H)
                    
                    for pos, team in zip(transformed_players, teams):
                        color = (255, 0, 0) if team == 1 else (0, 0, 255)
                        cv2.circle(field_view, (int(pos[0]), int(pos[1])), 5, color, -1)
                
                # Transformar árbitros
                if referee_positions:
                    ref_positions = [pos for pos, _ in referee_positions]
                    transformed_refs = self.transform_player_coordinates(ref_positions, self.H)
                    for pos in transformed_refs:
                        cv2.circle(field_view, (int(pos[0]), int(pos[1])), 5, (255, 192, 203), -1)  # Rosa
                
                # Transformar pelota
                if ball_positions:
                    transformed_ball = self.transform_player_coordinates(ball_positions, self.H)
                    for pos in transformed_ball:
                        cv2.circle(field_view, (int(pos[0]), int(pos[1])), 4, (0, 255, 0), -1)  # Verde sólido
                        cv2.circle(field_view, (int(pos[0]), int(pos[1])), 4, (0, 0, 0), 1)     # Contorno negro
        
        return display_frame, field_view, points_view, mapped_view

    def run(self):
        """Loop principal"""
        paused = False
        
        while self.cap.isOpened():
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
            
            # Procesar frame
            display_frame, field_view, points_view, mapped_view = self.process_frame(frame)
            
            # Guardar videos si está activada la opción
            if self.save_output:
                self.video_writers['video'].write(display_frame)
                self.video_writers['minimapa'].write(field_view)
                self.video_writers['puntos'].write(points_view)
                self.video_writers['mapeados'].write(mapped_view)
            
            # Mostrar resultados
            cv2.imshow('Video', display_frame)
            cv2.imshow('Minimapa', field_view)
            cv2.imshow('Puntos del Campo', points_view)
            cv2.imshow('Puntos Mapeados', mapped_view)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('n') and paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
            
        # Liberar recursos
        self.cap.release()
        
        # Cerrar los writers de video si están activos
        if self.save_output:
            for writer in self.video_writers.values():
                writer.release()
            
        cv2.destroyAllWindows()

    def cluster_teams(self, frame, player_boxes, min_confidence=0.7):
        """
        Separa los jugadores en dos equipos usando K-means en el espacio de color.
        Usa el 50% central del bounding box para mejor identificación.
        """
        samples = []
        valid_boxes = []
        
        for box, conf in player_boxes:
            if conf < min_confidence:
                continue
            
            # Extraer región central del jugador (50% del bounding box)
            x1, y1, x2, y2 = box.astype(int)
            width = x2 - x1
            height = y2 - y1
            
            # Calcular el centro del bounding box
            center_x = x1 + width // 2
            center_y = y1 + height // 3  # Un poco más arriba del centro para capturar mejor la camiseta
            
            # Definir una región del 50% alrededor del centro
            sample_width = width // 2  # 50% del ancho
            sample_height = height // 2  # 50% del alto
            
            # Calcular las coordenadas de la región de interés centrada
            sample_x1 = max(0, center_x - sample_width // 2)
            sample_y1 = max(0, center_y - sample_height // 2)
            sample_x2 = min(frame.shape[1], sample_x1 + sample_width)
            sample_y2 = min(frame.shape[0], sample_y1 + sample_height)
            
            # Extraer la región central
            try:
                roi = frame[sample_y1:sample_y2, sample_x1:sample_x2]
                if roi.size == 0:
                    continue
                
                # Convertir a HSV para mejor separación de colores
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Calcular el color promedio de la región (solo H y S)
                average_color = np.mean(roi_hsv[:, :, :2], axis=(0, 1))
                samples.append(average_color)
                valid_boxes.append((box, conf))
                
                # Debug: Visualizar la región de interés
                cv2.rectangle(frame, (sample_x1, sample_y1), (sample_x2, sample_y2), (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Error al procesar ROI: {e}")
                continue
        
        if len(samples) < 2:
            return [], []
        
        # Convertir a array numpy
        samples = np.array(samples)
        
        # Si aún no tenemos colores de referencia establecidos
        if self.team_colors is None:
            # Acumular muestras hasta tener suficientes
            self.team_samples.extend(samples.tolist())  # Convertir a lista para poder extender
            
            if len(self.team_samples) >= self.min_samples_init:
                # Inicializar los colores de referencia usando todas las muestras acumuladas
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                kmeans.fit(np.array(self.team_samples))
                self.team_colors = kmeans.cluster_centers_
                print("Colores de equipo inicializados")
        
        # Asignar equipos basados en la distancia a los colores de referencia
        if self.team_colors is not None:
            # Calcular distancias a cada color de referencia
            labels = []
            for sample in samples:
                dist1 = np.linalg.norm(sample - self.team_colors[0])
                dist2 = np.linalg.norm(sample - self.team_colors[1])
                label = 0 if dist1 < dist2 else 1
                labels.append(label)
        else:
            # Si aún no tenemos suficientes muestras, usar clustering temporal
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(samples)
        
        # Separar los boxes por equipo
        team1_boxes = []
        team2_boxes = []
        
        for (box, conf), label in zip(valid_boxes, labels):
            if label == 0:
                team1_boxes.append((box, conf))
            else:
                team2_boxes.append((box, conf))
        
        return team1_boxes, team2_boxes

def main():
    parser = argparse.ArgumentParser(description='Visualizador de Jugadores en Minimapa')
    parser.add_argument('--pose_model', type=str, 
                        default="models/field-keypoints-detection/weights/best.pt",
                        help='Path al modelo de pose')
    parser.add_argument('--player_model', type=str,
                        default="models/players-ball-detection/weights/best.pt",
                        help='Path al modelo de detección de jugadores')
    parser.add_argument('--video', type=str,
                        default="test_videos/B1606b0e6_1 (41).mp4", 
                        help='Path al video')
    parser.add_argument('--mapping_file', type=str,
                        default="point_mapping.json",
                        help='Path al archivo de mapeo de puntos')
    parser.add_argument('--save', action='store_true',
                        help='Guardar videos de salida en la carpeta "results"')
    
    args = parser.parse_args()
    
    tester = HomographyTester(args.pose_model, args.player_model, args.video, args.mapping_file, 
                             save_output=args.save)
    tester.run()

if __name__ == "__main__":
    main()
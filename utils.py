import cv2
import numpy as np

class FieldUtils:
    def __init__(self, field_width=800, margin=50):
        # Dimensiones del campo (proporciones FIFA)
        self.real_field_width = 105  # metros
        self.real_field_height = 68  # metros
        
        # Dimensiones en píxeles para la visualización
        self.field_width = field_width
        self.field_height = int(field_width * (self.real_field_height / self.real_field_width))
        self.margin = margin

    def create_field_image(self):
        """Crear imagen base del campo"""
        # Crear imagen base con fondo verde
        field = np.zeros((self.field_height, self.field_width, 3), dtype=np.uint8)
        field[:, :] = (53, 140, 50)  # Color verde del campo
        
        # Crear patrón de rayas
        stripe_width = self.field_width // 12
        for i in range(0, 13, 2):
            x1 = self.margin + i * stripe_width
            x2 = self.margin + (i+1) * stripe_width
            cv2.rectangle(field, 
                         (x1, self.margin), 
                         (x2, self.field_height - self.margin),
                         (45, 125, 45), -1)  # Verde ligeramente más oscuro
        
        # Dibujar líneas blancas del campo
        self._draw_field_lines(field)
        
        return field

    def _draw_field_lines(self, field):
        """Dibujar líneas del campo"""
        width = self.field_width
        height = self.field_height
        margin = self.margin
        mid_x = width // 2
        mid_y = height // 2
        
        # Dimensiones proporcionales
        big_box_width = int(16.5 * (width - 2*margin) / self.real_field_width)
        big_box_height = int(40.3 * (height - 2*margin) / self.real_field_height)
        small_box_width = int(5.5 * (width - 2*margin) / self.real_field_width)
        small_box_height = int(18.3 * (height - 2*margin) / self.real_field_height)
        circle_radius = int(9.15 * (width - 2*margin) / self.real_field_width)
        
        # Líneas principales
        cv2.rectangle(field, (margin, margin), (width-margin, height-margin), (255, 255, 255), 2)
        cv2.line(field, (mid_x, margin), (mid_x, height-margin), (255, 255, 255), 2)
        cv2.circle(field, (mid_x, mid_y), circle_radius, (255, 255, 255), 2)
        cv2.circle(field, (mid_x, mid_y), 3, (255, 255, 255), -1)
        
        # Áreas grandes
        cv2.rectangle(field, 
                     (margin, mid_y-big_box_height//2),
                     (margin+big_box_width, mid_y+big_box_height//2),
                     (255, 255, 255), 2)
        cv2.rectangle(field,
                     (width-margin-big_box_width, mid_y-big_box_height//2),
                     (width-margin, mid_y+big_box_height//2),
                     (255, 255, 255), 2)
        
        # Áreas pequeñas
        cv2.rectangle(field,
                     (margin, mid_y-small_box_height//2),
                     (margin+small_box_width, mid_y+small_box_height//2),
                     (255, 255, 255), 2)
        cv2.rectangle(field,
                     (width-margin-small_box_width, mid_y-small_box_height//2),
                     (width-margin, mid_y+small_box_height//2),
                     (255, 255, 255), 2)
        
        # Puntos de penalti
        penalty_spot_dist = int(11 * (width - 2*margin) / self.real_field_width)
        cv2.circle(field, (margin+penalty_spot_dist, mid_y), 3, (255, 255, 255), -1)
        cv2.circle(field, (width-margin-penalty_spot_dist, mid_y), 3, (255, 255, 255), -1)
        
        # Semicírculos del área
        self._draw_penalty_arcs(field, margin, mid_y, big_box_width, circle_radius)

    def _draw_penalty_arcs(self, field, margin, mid_y, big_box_width, circle_radius):
        """Dibujar semicírculos del área penal"""
        penalty_spot_dist = int(11 * (self.field_width - 2*margin) / self.real_field_width)
        
        # Arco izquierdo
        center_left = (margin + penalty_spot_dist, mid_y)
        self._draw_arc_outside_box(field, center_left, circle_radius, 
                                 margin + big_box_width, True)
        
        # Arco derecho
        center_right = (self.field_width - margin - penalty_spot_dist, mid_y)
        self._draw_arc_outside_box(field, center_right, circle_radius, 
                                 self.field_width - margin - big_box_width, False)

    def _draw_arc_outside_box(self, field, center, radius, box_x, is_left):
        """Dibujar arco que termina en la línea del área"""
        cx, cy = center
        if is_left:
            dx = box_x - cx
        else:
            dx = cx - box_x
            
        if abs(dx) >= radius:
            return
            
        angle = np.degrees(np.arccos(dx / radius))
        
        if is_left:
            start_angle = -angle
            end_angle = angle
        else:
            start_angle = 180 - angle
            end_angle = 180 + angle
            
        cv2.ellipse(field, center, (radius, radius), 
                    0, start_angle, end_angle, (255, 255, 255), 2)

    def get_field_coordinates(self):
        """Obtener coordenadas de los puntos clave del campo"""
        coordinates = {}
        width = self.field_width
        height = self.field_height
        margin = self.margin
        mid_x = width // 2
        mid_y = height // 2
        
        # Dimensiones proporcionales
        big_box_width = int(16.5 * (width - 2*margin) / self.real_field_width)
        big_box_height = int(40.3 * (height - 2*margin) / self.real_field_height)
        small_box_width = int(5.5 * (width - 2*margin) / self.real_field_width)
        small_box_height = int(18.3 * (height - 2*margin) / self.real_field_height)
        circle_radius = int(9.15 * (width - 2*margin) / self.real_field_width)
        
        # Definir todos los puntos clave (igual que antes)
        # Perímetro y línea central (0-5)
        coordinates[0] = (margin, margin)
        coordinates[1] = (mid_x, margin)
        coordinates[2] = (width - margin, margin)
        coordinates[3] = (margin, height - margin)
        coordinates[4] = (mid_x, height - margin)
        coordinates[5] = (width - margin, height - margin)
        
        # Círculo central (6-9)
        coordinates[6] = (mid_x, mid_y - circle_radius)
        coordinates[7] = (mid_x - circle_radius, mid_y)
        coordinates[8] = (mid_x + circle_radius, mid_y)
        coordinates[9] = (mid_x, mid_y + circle_radius)
        
        # Calcular intersecciones de arcos con áreas
        penalty_spot_dist = int(11 * (width - 2*margin) / self.real_field_width)
        
        # Área izquierda (10-20)
        coordinates[10] = (margin, mid_y - big_box_height // 2)
        coordinates[11] = (margin + big_box_width, mid_y - big_box_height // 2)
        coordinates[12] = (margin, mid_y + big_box_height // 2)
        coordinates[13] = (margin + big_box_width, mid_y + big_box_height // 2)
        
        # Calcular intersecciones del arco izquierdo
        dx_left = (margin + big_box_width) - (margin + penalty_spot_dist)
        if dx_left < circle_radius:
            angle = np.arccos(dx_left / circle_radius)
            y_offset = circle_radius * np.sin(angle)
            coordinates[14] = (margin + big_box_width, mid_y - y_offset)
            coordinates[15] = (margin + big_box_width, mid_y + y_offset)
        else:
            coordinates[14] = (margin + big_box_width, mid_y - circle_radius)
            coordinates[15] = (margin + big_box_width, mid_y + circle_radius)
        
        coordinates[16] = (margin, mid_y - small_box_height // 2)
        coordinates[17] = (margin + small_box_width, mid_y - small_box_height // 2)
        coordinates[18] = (margin, mid_y + small_box_height // 2)
        coordinates[19] = (margin + small_box_width, mid_y + small_box_height // 2)
        coordinates[20] = (margin + penalty_spot_dist, mid_y)
        
        # Área derecha (21-31)
        coordinates[21] = (width - margin, mid_y - big_box_height // 2)
        coordinates[22] = (width - margin - big_box_width, mid_y - big_box_height // 2)
        coordinates[23] = (width - margin, mid_y + big_box_height // 2)
        coordinates[24] = (width - margin - big_box_width, mid_y + big_box_height // 2)
        
        # Calcular intersecciones del arco derecho
        dx_right = (width - margin - penalty_spot_dist) - (width - margin - big_box_width)
        if dx_right < circle_radius:
            angle = np.arccos(dx_right / circle_radius)
            y_offset = circle_radius * np.sin(angle)
            coordinates[25] = (width - margin - big_box_width, mid_y - y_offset)
            coordinates[26] = (width - margin - big_box_width, mid_y + y_offset)
        else:
            coordinates[25] = (width - margin - big_box_width, mid_y - circle_radius)
            coordinates[26] = (width - margin - big_box_width, mid_y + circle_radius)
        
        coordinates[27] = (width - margin, mid_y - small_box_height // 2)
        coordinates[28] = (width - margin - small_box_width, mid_y - small_box_height // 2)
        coordinates[29] = (width - margin, mid_y + small_box_height // 2)
        coordinates[30] = (width - margin - small_box_width, mid_y + small_box_height // 2)
        coordinates[31] = (width - margin - penalty_spot_dist, mid_y)
        
        return coordinates 
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import json
from PIL import Image, ImageTk
from utils import FieldUtils

class PointMapperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mapeo de Puntos del Campo")
        
        # Inicializar FieldUtils
        self.field_utils = FieldUtils()
        
        # Obtener dimensiones del campo
        self.field_width = self.field_utils.field_width
        self.field_height = self.field_utils.field_height
        self.margin = self.field_utils.margin
        self.real_field_width = self.field_utils.real_field_width
        self.real_field_height = self.field_utils.real_field_height
        
        # Crear canvas para el campo
        self.canvas = tk.Canvas(root, width=self.field_width, height=self.field_height, bg='#358c32')  # Verde más realista
        self.canvas.pack(pady=10)
        
        # Frame para controles
        control_frame = ttk.Frame(root)
        control_frame.pack(pady=10)
        
        # Entrada para el mapeo
        ttk.Label(control_frame, text="Punto del detector (0-31):").pack(side=tk.LEFT, padx=5)
        self.detector_point = tk.StringVar()
        self.point_entry = ttk.Entry(control_frame, textvariable=self.detector_point, width=5)
        self.point_entry.pack(side=tk.LEFT, padx=5)
        
        # Botón para guardar mapeo
        ttk.Button(control_frame, text="Guardar Mapeo", command=self.save_mapping).pack(side=tk.LEFT, padx=5)
        
        # Botón para exportar a JSON
        ttk.Button(control_frame, text="Exportar JSON", command=self.export_json).pack(side=tk.LEFT, padx=5)
        
        # Diccionario para almacenar el mapeo
        self.point_mapping = {}
        
        # Conjunto para mantener control de los puntos ya mapeados
        self.mapped_field_points = set()
        
        # Dibujar campo y puntos
        self.draw_field()
        self.draw_points()
        
        # Bind click events
        self.canvas.bind('<Button-1>', self.on_click)
        
        # Variable para almacenar el punto seleccionado actualmente
        self.selected_point = None
        
        # Label para mostrar información
        self.info_label = ttk.Label(root, text="Haz clic en un punto y escribe el número del detector correspondiente")
        self.info_label.pack(pady=5)
        
        # Información de las dimensiones
        dimension_label = ttk.Label(root, 
                                    text=f"Campo a escala: {self.real_field_width}m × {self.real_field_height}m")
        dimension_label.pack(pady=2)

    def draw_field(self):
        """Dibujar el campo en el canvas"""
        # Obtener la imagen del campo
        field_img = self.field_utils.create_field_image()
        
        # Convertir a formato PIL para Tkinter
        field_img = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)
        field_img = Image.fromarray(field_img)
        self.field_photo = ImageTk.PhotoImage(image=field_img)
        
        # Dibujar en el canvas
        self.canvas.create_image(0, 0, image=self.field_photo, anchor=tk.NW)

    def draw_points(self):
        """Dibujar puntos en el campo"""
        # Obtener coordenadas de los puntos
        self.field_points = []
        field_coordinates = self.field_utils.get_field_coordinates()
        
        # Dibujar cada punto con su número
        for i, (x, y) in field_coordinates.items():
            point_color = 'green' if i in self.mapped_field_points else 'red'
            point_id = self.canvas.create_oval(x-5, y-5, x+5, y+5, 
                                             fill=point_color, outline='white')
            text_id = self.canvas.create_text(x+10, y-10, 
                                            text=str(i), fill='white')
            self.field_points.append((x, y, point_id, text_id))

    def on_click(self, event):
        # Encontrar el punto más cercano al clic
        min_dist = float('inf')
        closest_point = None
        closest_index = None
        
        for i, (x, y, point_id, text_id) in enumerate(self.field_points):
            dist = ((event.x - x)**2 + (event.y - y)**2)**0.5
            if dist < min_dist and dist < 20:  # 20 pixels de tolerancia
                min_dist = dist
                closest_point = (x, y, point_id, text_id)
                closest_index = i
        
        if closest_point:
            # Resaltar el punto seleccionado
            if self.selected_point:
                self.canvas.itemconfig(self.selected_point[2], fill='red')
            self.canvas.itemconfig(closest_point[2], fill='yellow')
            self.selected_point = closest_point
            
            # Actualizar label con información
            self.info_label.config(text=f"Punto del campo {closest_index} seleccionado")

    def save_mapping(self):
        if self.selected_point and self.detector_point.get().isdigit():
            detector_idx = int(self.detector_point.get())
            if 0 <= detector_idx <= 31:
                # Encontrar el índice del punto del campo
                field_idx = self.field_points.index(self.selected_point)
                self.point_mapping[field_idx] = detector_idx
                
                # Marcar este punto como mapeado y cambiar su color a verde
                self.mapped_field_points.add(field_idx)
                self.canvas.itemconfig(self.selected_point[2], fill='green')
                
                # Actualizar label
                self.info_label.config(
                    text=f"Mapeado: Punto del campo {field_idx} → Punto del detector {detector_idx}")
                
                # Limpiar entrada
                self.detector_point.set("")
                
                # Desseleccionar punto
                self.selected_point = None
            else:
                self.info_label.config(text="Error: El número debe estar entre 0 y 31")
        else:
            self.info_label.config(text="Error: Selecciona un punto y escribe un número válido")

    def export_json(self):
        if self.point_mapping:
            # Crear un diccionario ampliado que contiene tanto el mapeo como las coordenadas
            export_data = {
                "mapping": self.point_mapping,
                "coordinates": {},
                "field_dimensions": {
                    "width": self.real_field_width,
                    "height": self.real_field_height,
                    "canvas_width": self.field_width,
                    "canvas_height": self.field_height,
                    "margin": self.margin
                }
            }
            
            # Guardar las coordenadas de cada punto del campo
            for i, (x, y, _, _) in enumerate(self.field_points):
                export_data["coordinates"][i] = [x, y]
            
            # Guardar en el archivo JSON
            with open('point_mapping.json', 'w') as f:
                json.dump(export_data, f, indent=4)
            
            self.info_label.config(text=f"Mapeo y coordenadas guardados en point_mapping.json ({len(self.point_mapping)} puntos)")
        else:
            self.info_label.config(text="Error: No hay puntos mapeados para guardar")

def main():
    root = tk.Tk()
    app = PointMapperApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
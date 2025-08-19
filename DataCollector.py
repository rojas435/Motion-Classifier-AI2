import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class PoseAnalysisSystem:
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Actividades a detectar
        self.activities = [
            "approaching",
            "departing",
            "rotating",
            "seated",
            "upright"
        ]
        
        # Estructura de datos
        self.landmarks_data = []
        self.current_activity = None
        self.recording = False
        
        # Modelo y escalador
        self.model = None
        self.scaler = StandardScaler()
        
        # Rutas de archivos
        self.model_params_file = 'model_params.json'
        self.model_file = 'pose_model.pkl'
        
        # Buffer para análisis en tiempo real
        self.frame_buffer = []
        self.buffer_size = 30
            
            
    def create_gui(self):
        """Crea una interfaz gráfica rediseñada para la captura de datos y visualización."""
        self.root = tk.Tk()
        self.root.title("Pose Analysis System")
        self.root.geometry("1920x1080")  # Tamaño de la ventana
        self.root.configure(bg="#1e272e")

        # Marco para el video
        video_frame = tk.Frame(self.root, bg="#2f3640", bd=5, relief=tk.RIDGE)
        video_frame.place(relx=0.5, rely=0.35, anchor=tk.CENTER, width=1000, height=500)  # Ajustar tamaño del recuadro

        self.video_label = tk.Label(video_frame, bg="#2f3640")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Marco para controles
        controls_frame = tk.Frame(self.root, bg="#1e272e")
        controls_frame.place(relx=0.5, rely=0.8, anchor=tk.CENTER)  # Mover controles debajo del video

        # Selector de actividad
        activity_label = tk.Label(
            controls_frame, text="Select Activity:", font=("Helvetica", 14), fg="white", bg="#1e272e"
        )
        activity_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.activity_var = tk.StringVar()
        activity_selector = ttk.Combobox(
            controls_frame, textvariable=self.activity_var, values=self.activities, state="readonly"
        )
        activity_selector.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # Botón de grabación
        self.record_btn = tk.Button(
            controls_frame,
            text="Start Recording",
            font=("Helvetica", 14),
            bg="#e84118",
            fg="white",
            activebackground="#c23616",
            activeforeground="white",
            command=self.toggle_recording,
        )
        self.record_btn.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        # Botón para cargar video
        self.load_video_btn = tk.Button(
            controls_frame,
            text="Load Video",
            font=("Helvetica", 14),
            bg="#00a8ff",
            fg="white",
            activebackground="#0097e6",
            activeforeground="white",
            command=self.load_video,
        )
        self.load_video_btn.grid(row=0, column=3, padx=10, pady=10, sticky="w")

        # Add a button for analyzing data in the controls_frame
        analyze_btn = tk.Button(
            controls_frame,
            text="Analyze Data",
            font=("Helvetica", 14),
            bg="#4cd137",
            fg="white",
            activebackground="#44bd32",
            activeforeground="white",
            command=self.analyze_data,
        )
        analyze_btn.grid(row=0, column=4, padx=10, pady=10, sticky="w")

        # Botón para cerrar la aplicación de forma adecuada
        close_btn = tk.Button(
            controls_frame,
            text="Cerrar",
            font=("Helvetica", 14),
            bg="#353b48",
            fg="white",
            activebackground="#718093",
            activeforeground="white",
            command=self.close_app,
        )
        close_btn.grid(row=0, column=5, padx=10, pady=10, sticky="w")

        # Variables de control
        self.is_predicting = False
        self.cap = cv2.VideoCapture(0)

        # Iniciar actualización de video
        self.update_video()

    def load_video(self):
        """Permite al usuario cargar un archivo de video para análisis y mostrar el proceso en tiempo real."""
        from tkinter.filedialog import askopenfilename

        # Preguntar por la actividad antes de cargar el video
        activity = self.activity_var.get()
        if not activity:
            messagebox.showerror("Error", "Por favor selecciona una actividad antes de cargar el video.")
            return

        # Abrir el explorador de archivos para seleccionar el video
        video_path = askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if video_path:
            self.cap.release()  # Liberar la cámara si está en uso
            self.cap = cv2.VideoCapture(video_path)

            # Configurar la actividad seleccionada y comenzar a grabar automáticamente
            self.current_activity = activity
            self.recording = True

            def process_video():
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    # Procesar el frame y recolectar datos
                    frame, landmarks = self.process_frame(frame)
                    if landmarks is not None:
                        self.collect_data(landmarks)

                    # Mostrar el frame procesado en la interfaz
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)

                    # --- Ajuste para mantener relación de aspecto y evitar recortes al montaje del video ---
                    desired_width = 1000
                    desired_height = 500
                    orig_width, orig_height = img.size
                    aspect_ratio = orig_width / orig_height

                    target_ratio = desired_width / desired_height
                    if aspect_ratio > target_ratio:
                        # Imagen más ancha que el widget: ajustar por ancho
                        new_width = desired_width
                        new_height = int(desired_width / aspect_ratio)
                    else:
                        # Imagen más alta que el widget: ajustar por alto
                        new_height = desired_height
                        new_width = int(desired_height * aspect_ratio)

                    img = img.resize((new_width, new_height), Image.LANCZOS)

                    # Crear un fondo negro y centrar la imagen
                    background = Image.new('RGB', (desired_width, desired_height), (30, 39, 46))
                    offset = ((desired_width - new_width) // 2, (desired_height - new_height) // 2)
                    background.paste(img, offset)
                    imgtk = ImageTk.PhotoImage(image=background)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                    self.root.update_idletasks()
                    self.root.update()

                # Detener la grabación y guardar los datos
                self.recording = False
                self.save_data()
                messagebox.showinfo("Información", f"Datos guardados en outputRecords.csv para la actividad: {activity}")

                # Volver al estado inicial mostrando la cámara en vivo
                self.cap.release()
                self.cap = cv2.VideoCapture(0)

            # Procesar el video en tiempo real
            process_video()

    def update_video(self):
        """Actualiza el frame de video en la GUI."""
        ret, frame = self.cap.read()
        if ret:
            frame, landmarks = self.process_frame(frame)
            
            if self.recording and landmarks is not None:
                self.collect_data(landmarks)
            
            if self.is_predicting and landmarks is not None:
                self.predict_activity(landmarks)
            
            # Convertir frame para GUI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # --- Ajuste para mantener relación de aspecto y evitar recortes ---
            desired_width = 1000
            desired_height = 500
            orig_width, orig_height = img.size
            aspect_ratio = orig_width / orig_height

            target_ratio = desired_width / desired_height
            if aspect_ratio > target_ratio:
                # Imagen más ancha que el widget: ajustar por ancho
                new_width = desired_width
                new_height = int(desired_width / aspect_ratio)
            else:
                # Imagen más alta que el widget: ajustar por alto
                new_height = desired_height
                new_width = int(desired_height * aspect_ratio)

            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Crear un fondo negro y centrar la imagen (letterbox)
            background = Image.new('RGB', (desired_width, desired_height), (30, 39, 46))
            offset = ((desired_width - new_width) // 2, (desired_height - new_height) // 2)
            background.paste(img, offset)
            imgtk = ImageTk.PhotoImage(image=background)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.root.after(10, self.update_video)
        
    def process_frame(self, frame):
        """Procesa un frame y extrae landmarks."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            # Cambiar la visualización de las articulaciones
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Verde para las líneas
                self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=6),  # Amarillo para los puntos
            )

            # Calcular y mostrar ángulos importantes
            landmarks_dict = self._extract_landmarks(results.pose_landmarks)
            angles = self._calculate_key_angles(landmarks_dict)

            # Ocultar la visualización de ángulos en el frame
            # Dibuja un contorno alrededor de la cara usando los landmarks clave (nariz, ojos y orejas)
            self._draw_face_contour(frame, results.pose_landmarks)

        return frame, results.pose_landmarks if results.pose_landmarks else None
        
    def _extract_landmarks(self, landmarks):
        """Extrae coordenadas de landmarks importantes."""
        keypoints = {}
        
        important_landmarks = {
            'LEFT_HIP': self.mp_pose.PoseLandmark.LEFT_HIP,
            'RIGHT_HIP': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'LEFT_KNEE': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'RIGHT_KNEE': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'LEFT_ANKLE': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'RIGHT_ANKLE': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'LEFT_SHOULDER': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'RIGHT_SHOULDER': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'LEFT_WRIST': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'RIGHT_WRIST': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'NOSE': self.mp_pose.PoseLandmark.NOSE
        }
        
        for name, landmark in important_landmarks.items():
            keypoints[f'{name}_x'] = landmarks.landmark[landmark].x
            keypoints[f'{name}_y'] = landmarks.landmark[landmark].y
            keypoints[f'{name}_z'] = landmarks.landmark[landmark].z
            
        return keypoints
        
    def _calculate_key_angles(self, landmarks):
        """Calcula ángulos importantes del cuerpo."""
        angles = {}
        
        # Ángulo de rodilla izquierda
        left_knee_angle = self._calculate_angle(
            [landmarks['LEFT_HIP_x'], landmarks['LEFT_HIP_y']],
            [landmarks['LEFT_KNEE_x'], landmarks['LEFT_KNEE_y']],
            [landmarks['LEFT_ANKLE_x'], landmarks['LEFT_ANKLE_y']]
        )
        angles['Left Knee'] = left_knee_angle
        
        # Ángulo de rodilla derecha
        right_knee_angle = self._calculate_angle(
            [landmarks['RIGHT_HIP_x'], landmarks['RIGHT_HIP_y']],
            [landmarks['RIGHT_KNEE_x'], landmarks['RIGHT_KNEE_y']],
            [landmarks['RIGHT_ANKLE_x'], landmarks['RIGHT_ANKLE_y']]
        )
        angles['Right Knee'] = right_knee_angle
        
        # Inclinación del tronco
        trunk_angle = self._calculate_angle(
            [(landmarks['LEFT_SHOULDER_x'] + landmarks['RIGHT_SHOULDER_x'])/2,
            (landmarks['LEFT_SHOULDER_y'] + landmarks['RIGHT_SHOULDER_y'])/2],
            [(landmarks['LEFT_HIP_x'] + landmarks['RIGHT_HIP_x'])/2,
            (landmarks['LEFT_HIP_y'] + landmarks['RIGHT_HIP_y'])/2],
            [(landmarks['LEFT_HIP_x'] + landmarks['RIGHT_HIP_x'])/2,
            (landmarks['LEFT_HIP_y'] + landmarks['RIGHT_HIP_y'])/2 + 0.1]
        )
        angles['Trunk Inclination'] = trunk_angle
        
        return angles
        
    def _calculate_angle(self, a, b, c):
        """Calcula el ángulo entre tres puntos."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
        
    def collect_data(self, landmarks):
        """Recolecta datos de landmarks con etiquetas."""
        if self.current_activity:
            landmarks_dict = self._extract_landmarks(landmarks)
            angles = self._calculate_key_angles(landmarks_dict)
            
            data_point = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                'activity': self.current_activity,
                **landmarks_dict,
                **angles
            }
            
            self.landmarks_data.append(data_point)
            
    def toggle_recording(self):
        """Alterna el estado de grabación."""
        self.recording = not self.recording
        self.current_activity = self.activity_var.get() if self.recording else None
        
        if self.recording:
            self.record_btn.configure(text="Detener Grabación")
        else:
            self.record_btn.configure(text="Iniciar Grabación")
            self.save_data()
            
    def save_data(self, filename='outputRecords.csv'):
        """Guarda los datos recolectados, agregando al archivo existente si ya hay datos previos."""
        if self.landmarks_data:
            new_df = pd.DataFrame(self.landmarks_data)
            if Path(filename).exists():
                try:
                    old_df = pd.read_csv(filename)
                    combined_df = pd.concat([old_df, new_df], ignore_index=True)
                except Exception:
                    combined_df = new_df
            else:
                combined_df = new_df
            combined_df.to_csv(filename, index=False)
            print(f"Datos guardados en {filename}")
            self.landmarks_data = []  # Limpiar datos en memoria después de guardar
            
    def analyze_data(self):
        """Realiza análisis exploratorio de datos."""
        if not self.landmarks_data:
            print("No hay datos para analizar")
            return

        # Renaming variables in the dataset
        df = pd.DataFrame(self.landmarks_data)
        df.rename(columns={
            'timestamp': 'recorded_time',
            'activity': 'movement_type',
            'NOSE_x': 'nose_x_position',
            'NOSE_y': 'nose_y_position',
            'NOSE_z': 'nose_z_position',
            'Left Knee': 'left_knee_bend',
            'Right Knee': 'right_knee_bend',
            'Trunk Inclination': 'trunk_tilt'
        }, inplace=True)

        # Create main directory for analysis
        main_dir = Path("data_insights")
        main_dir.mkdir(exist_ok=True)

        # Generate graphs for each action
        for action in df['movement_type'].unique():
            action_dir = main_dir / action
            action_dir.mkdir(exist_ok=True)

            action_data = df[df['movement_type'] == action]

            # 1. Histogram of trunk tilt
            plt.figure(figsize=(10, 6))
            action_data['trunk_tilt'].hist(bins=20, color='blue', alpha=0.7)
            plt.title(f'Histogram of Trunk Tilt - {action}')
            plt.xlabel('Trunk Tilt')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(action_dir / 'trunk_tilt_histogram.png')
            plt.close()

            # 2. Scatter plot of nose positions
            plt.figure(figsize=(10, 6))
            plt.scatter(action_data['nose_x_position'], action_data['nose_y_position'], alpha=0.5, c='green')
            plt.title(f'Scatter Plot of Nose Positions - {action}')
            plt.xlabel('Nose X Position')
            plt.ylabel('Nose Y Position')
            plt.tight_layout()
            plt.savefig(action_dir / 'nose_positions_scatter.png')
            plt.close()

            # 3. Boxplot of knee bends
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=action_data, y='left_knee_bend', color='orange')
            plt.title(f'Boxplot of Left Knee Bend - {action}')
            plt.tight_layout()
            plt.savefig(action_dir / 'left_knee_bend_boxplot.png')
            plt.close()

            plt.figure(figsize=(12, 6))
            sns.boxplot(data=action_data, y='right_knee_bend', color='purple')
            plt.title(f'Boxplot of Right Knee Bend - {action}')
            plt.tight_layout()
            plt.savefig(action_dir / 'right_knee_bend_boxplot.png')
            plt.close()

            # 4. Heatmap of correlations
            plt.figure(figsize=(10, 8))
            correlation_matrix = action_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title(f'Correlation Heatmap - {action}')
            plt.tight_layout()
            plt.savefig(action_dir / 'correlation_heatmap.png')
            plt.close()
        
    def prepare_features(self, landmarks_dict, angles):
        """Prepara características para el modelo."""
        features = []
        for key in landmarks_dict.keys():
            features.append(landmarks_dict[key])
        for key in angles.keys():
            features.append(angles[key])
        return np.array(features).reshape(1, -1)
            
    
    def run(self):
        """Ejecuta la aplicación."""
        self.create_gui()
        self.root.mainloop()
        
    def cleanup(self):
        """Limpia recursos."""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
    def _draw_face_contour(self, frame, landmarks):
        """Dibuja un contorno alrededor de la cara usando los landmarks clave."""
        # Coordenadas de los landmarks clave para el contorno facial
        nose = [landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x, landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y]
        left_eye = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x, landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].y]
        right_eye = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].y]
        left_ear = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].x, landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].y]
        right_ear = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].y]
        
        # Convertir coordenadas a enteros
        nose = (int(nose[0] * frame.shape[1]), int(nose[1] * frame.shape[0]))
        left_eye = (int(left_eye[0] * frame.shape[1]), int(left_eye[1] * frame.shape[0]))
        right_eye = (int(right_eye[0] * frame.shape[1]), int(right_eye[1] * frame.shape[0]))
        left_ear = (int(left_ear[0] * frame.shape[1]), int(left_ear[1] * frame.shape[0]))
        right_ear = (int(right_ear[0] * frame.shape[1]), int(right_ear[1] * frame.shape[0]))
        
        # Dibujar el contorno facial
        cv2.line(frame, left_ear, left_eye, (255, 0, 0), 2)
        cv2.line(frame, left_eye, nose, (255, 0, 0), 2)
        cv2.line(frame, nose, right_eye, (255, 0, 0), 2)
        cv2.line(frame, right_eye, right_ear, (255, 0, 0), 2)
        cv2.line(frame, left_ear, nose, (255, 0, 0), 2)
        cv2.line(frame, right_ear, nose, (255, 0, 0), 2)
        
    def close_app(self):
        """Cierra la aplicación de forma segura."""
        self.cleanup()
        self.root.destroy()
        
def main():
    system = PoseAnalysisSystem()
    try:
        system.run()
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()
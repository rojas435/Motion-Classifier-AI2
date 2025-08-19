import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from collections import Counter
import logging 

class PosePredictorApp:
    """
    Aplicación para la predicción de actividad física en tiempo real
    utilizando un modelo entrenado y detección de pose con MediaPipe.
    """

    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Clasificador de Actividad en Tiempo Real")

        # --- Configuración de MediaPipe ---
        self.mp_pose_solutions = mp.solutions.pose
        self.mp_drawing_utilities = mp.solutions.drawing_utils
        self.pose_estimator = self.mp_pose_solutions.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # --- Carga del Modelo y Componentes ---
        self.trained_model = None
        self.feature_scaler = None
        self.label_class_encoder = None
        self.model_feature_names = None
        self.model_class_names = None
        self._load_trained_artifacts()

        # --- Estado de la Aplicación ---
        self.is_prediction_active = False
        self.video_capture_source = 0
        self.video_stream = None

        # --- Buffer para Suavizado de Predicciones ---
        self.prediction_log = []
        self.prediction_log_max_size = 7 
        self.display_confidence_threshold = 60.0 

        # --- Inicialización de la Interfaz Gráfica ---
        self._setup_gui_elements()

    def _load_trained_artifacts(self, model_path='trainedModel.pkl'):
        """Carga el modelo serializado, el escalador y el codificador de etiquetas."""
        try:
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
                self.trained_model = model_package.get('model')
                self.feature_scaler = model_package.get('scaler')
                self.label_class_encoder = model_package.get('label_encoder')
                self.model_feature_names = model_package.get('feature_names') # Importante para orden de features
                self.model_class_names = model_package.get('class_names') # Nombres de clases originales

            if not all([self.trained_model, self.feature_scaler, self.label_class_encoder, self.model_feature_names, self.model_class_names]):
                raise ValueError("Algunos artefactos del modelo no se encontraron en el archivo .pkl.")
            
            logging.info(f"Modelo y artefactos cargados exitosamente desde {model_path}.")
            logging.info(f"Características esperadas por el modelo: {self.model_feature_names}")
            logging.info(f"Clases conocidas por el modelo: {self.model_class_names}")

        except FileNotFoundError:
            messagebox.showerror("Error Crítico", f"Archivo del modelo no encontrado: {model_path}")
            self.root.quit()
        except (pickle.UnpicklingError, ValueError, AttributeError, KeyError) as e:
            messagebox.showerror("Error Crítico", f"Error al cargar o validar el modelo: {e}")
            self.root.quit()


    def _setup_gui_elements(self):
        """Configura los widgets de la interfaz gráfica con el diseño de DataCollector.py."""
        self.root.configure(bg="#1e272e")

        main_container = ttk.Frame(self.root, padding="10 10 10 10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Sección de Video ---
        video_display_frame = tk.Frame(main_container, bg="#2f3640", bd=5, relief=tk.RIDGE)
        video_display_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        video_display_frame.columnconfigure(0, weight=1)
        video_display_frame.rowconfigure(0, weight=1)

        self.video_output_label = tk.Label(video_display_frame, bg="#2f3640")
        self.video_output_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Sección de Predicción ---
        prediction_info_frame = tk.Frame(main_container, bg="#1e272e")
        prediction_info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        tk.Label(prediction_info_frame, text="Actividad Estimada:", font=("Helvetica", 14), fg="white", bg="#1e272e").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.predicted_activity_label = tk.Label(prediction_info_frame, text="---", font=("Helvetica", 16, "bold"), fg="blue", bg="#1e272e")
        self.predicted_activity_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        tk.Label(prediction_info_frame, text="Confianza:", font=("Helvetica", 14), fg="white", bg="#1e272e").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.prediction_confidence_label = tk.Label(prediction_info_frame, text="---", font=("Helvetica", 12), fg="white", bg="#1e272e")
        self.prediction_confidence_label.grid(row=1, column=1, sticky=tk.W, padx=5)

        # --- Sección de Controles ---
        control_panel_frame = tk.Frame(main_container, bg="#1e272e")
        control_panel_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10, padx=10)

        self.toggle_predict_button = tk.Button(
            control_panel_frame,
            text="Iniciar Predicción",
            font=("Helvetica", 14),
            bg="#4cd137",
            fg="white",
            activebackground="#44bd32",
            activeforeground="white",
            command=self._toggle_prediction_state,
            width=15
        )
        self.toggle_predict_button.grid(row=0, column=0, padx=10, pady=10)

        self.status_indicator_label = tk.Label(
            control_panel_frame,
            text="Estado: Detenido",
            font=("Helvetica", 10, "italic"),
            fg="red",
            bg="#1e272e"
        )
        self.status_indicator_label.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)

        # --- Botón para cerrar la aplicación ---
        close_app_button = tk.Button(
            control_panel_frame,
            text="Cerrar",
            font=("Helvetica", 14),
            bg="#e84118",
            fg="white",
            activebackground="#c23616",
            activeforeground="white",
            command=self._on_closing_application,
            width=15
        )
        close_app_button.grid(row=0, column=2, padx=10, pady=10)

        # --- Selección de Fuente de Video ---
        source_selection_frame = tk.Frame(main_container, bg="#1e272e")
        source_selection_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10, padx=10)

        tk.Label(source_selection_frame, text="Fuente:", font=("Helvetica", 14), fg="white", bg="#1e272e").grid(row=0, column=0, padx=10, pady=10)
        self.video_source_var = tk.StringVar(value="Cámara 0")
        source_options = ["Cámara 0", "Cámara 1", "Archivo de Video"]
        source_menu = ttk.OptionMenu(source_selection_frame, self.video_source_var, self.video_source_var.get(), *source_options, command=self._change_video_source)
        source_menu.grid(row=0, column=1, padx=10, pady=10)


    def _change_video_source(self, selected_source_str):
        """Maneja el cambio de la fuente de video."""
        if self.video_stream and self.video_stream.isOpened():
            self.video_stream.release()

        if selected_source_str == "Cámara 0":
            self.video_capture_source = 0
        elif selected_source_str == "Cámara 1":
            self.video_capture_source = 1
        elif selected_source_str == "Archivo de Video":
            filepath = filedialog.askopenfilename(title="Seleccionar archivo de video", filetypes=(("Archivos MP4", "*.mp4"), ("Archivos AVI", "*.avi"), ("Todos los archivos", "*.*")))
            if filepath:
                self.video_capture_source = filepath
            else: # El usuario canceló, volver a la cámara por defecto
                self.video_source_var.set("Cámara 0") 
                self.video_capture_source = 0
        
        self._initialize_video_stream()


    def _initialize_video_stream(self):
        """Inicializa o reinicializa el stream de video."""
        if self.video_stream and self.video_stream.isOpened():
            self.video_stream.release()
        
        self.video_stream = cv2.VideoCapture(self.video_capture_source)
        if not self.video_stream.isOpened():
            messagebox.showerror("Error de Video", f"No se pudo abrir la fuente de video: {self.video_capture_source}")
            self.video_capture_source = 0 # Reintentar con cámara 0 si falla
            self.video_source_var.set("Cámara 0")
            self.video_stream = cv2.VideoCapture(self.video_capture_source)
            if not self.video_stream.isOpened():
                 messagebox.showerror("Error Crítico", "No se pudo acceder a la cámara por defecto. La aplicación se cerrará.")
                 self.root.quit()


    def _extract_pose_landmarks(self, pose_results):
        """Extrae un diccionario de coordenadas de landmarks desde los resultados de MediaPipe."""
        landmarks_coords = {}
        landmark_feature_keys = [name for name in self.model_feature_names if '_x' in name or '_y' in name or '_z' in name]
        # Extraemos el nombre base del landmark, e.g., 'NOSE_x' -> 'NOSE'
        required_landmark_names = sorted(list(set([key.rsplit('_', 1)[0] for key in landmark_feature_keys])))

        for landmark_name_str in required_landmark_names:
            try:
                # Convertir el string del nombre del landmark al objeto PoseLandmark de MediaPipe
                mp_landmark_enum = getattr(self.mp_pose_solutions.PoseLandmark, landmark_name_str)
                landmark_data = pose_results.landmark[mp_landmark_enum]
                landmarks_coords[f'{landmark_name_str}_x'] = landmark_data.x
                landmarks_coords[f'{landmark_name_str}_y'] = landmark_data.y
                landmarks_coords[f'{landmark_name_str}_z'] = landmark_data.z
            except AttributeError:
                logging.warning(f"Landmark {landmark_name_str} no encontrado en mp.solutions.pose.PoseLandmark. Verifique los nombres.")
                # Llenar con NaN o 0 si un landmark no se encuentra, para mantener la dimensionalidad
                landmarks_coords[f'{landmark_name_str}_x'] = 0.0 
                landmarks_coords[f'{landmark_name_str}_y'] = 0.0
                landmarks_coords[f'{landmark_name_str}_z'] = 0.0
                
        return landmarks_coords

    def _compute_body_angles(self, current_landmarks):
        """Calcula ángulos corporales clave a partir de las coordenadas de landmarks."""
        angles_values = {}
        
        # Estos cálculos deben ser consistentes con los del entrenamiento
        # Ejemplo: Rodilla Izquierda
        try:
            p_hip_l = [current_landmarks['LEFT_HIP_x'], current_landmarks['LEFT_HIP_y']]
            p_knee_l = [current_landmarks['LEFT_KNEE_x'], current_landmarks['LEFT_KNEE_y']]
            p_ankle_l = [current_landmarks['LEFT_ANKLE_x'], current_landmarks['LEFT_ANKLE_y']]
            angles_values['Left Knee'] = self._calculate_angle_3p(p_hip_l, p_knee_l, p_ankle_l)
        except KeyError:
             angles_values['Left Knee'] = 0.0 # Valor por defecto

        # Ejemplo: Rodilla Derecha
        try:
            p_hip_r = [current_landmarks['RIGHT_HIP_x'], current_landmarks['RIGHT_HIP_y']]
            p_knee_r = [current_landmarks['RIGHT_KNEE_x'], current_landmarks['RIGHT_KNEE_y']]
            p_ankle_r = [current_landmarks['RIGHT_ANKLE_x'], current_landmarks['RIGHT_ANKLE_y']]
            angles_values['Right Knee'] = self._calculate_angle_3p(p_hip_r, p_knee_r, p_ankle_r)
        except KeyError:
            angles_values['Right Knee'] = 0.0

        # Ejemplo: Inclinación del Tronco
        try:
            p_shoulder_mid_x = (current_landmarks['LEFT_SHOULDER_x'] + current_landmarks['RIGHT_SHOULDER_x']) / 2
            p_shoulder_mid_y = (current_landmarks['LEFT_SHOULDER_y'] + current_landmarks['RIGHT_SHOULDER_y']) / 2
            p_hip_mid_x = (current_landmarks['LEFT_HIP_x'] + current_landmarks['RIGHT_HIP_x']) / 2
            p_hip_mid_y = (current_landmarks['LEFT_HIP_y'] + current_landmarks['RIGHT_HIP_y']) / 2
            
            p_shoulder_mid = [p_shoulder_mid_x, p_shoulder_mid_y]
            p_hip_mid = [p_hip_mid_x, p_hip_mid_y]
            p_hip_mid_vertical_ref = [p_hip_mid_x, p_hip_mid_y + 0.1] # Referencia vertical
            angles_values['Trunk Inclination'] = self._calculate_angle_3p(p_shoulder_mid, p_hip_mid, p_hip_mid_vertical_ref)
        except KeyError:
            angles_values['Trunk Inclination'] = 0.0
            
        return angles_values

    def _calculate_angle_3p(self, p1, p2, p3):
        """Calcula el ángulo (en p2) entre los vectores p2->p1 y p2->p3."""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_deg = np.degrees(angle_rad)
        return angle_deg if angle_deg <= 180.0 else 360.0 - angle_deg
        
    def _construct_feature_vector(self, landmark_data, angle_data):
        """Construye el vector de características en el orden esperado por el modelo."""
        feature_vector = []
        # Utiliza self.model_feature_names para asegurar el orden correcto
        for feature_name in self.model_feature_names:
            if feature_name in landmark_data:
                feature_vector.append(landmark_data[feature_name])
            elif feature_name in angle_data:
                feature_vector.append(angle_data[feature_name])
            else:
                # Esto no debería ocurrir si model_feature_names es correcto y los datos se calculan
                logging.warning(f"Característica esperada '{feature_name}' no encontrada en datos de landmarks o ángulos. Usando 0.0.")
                feature_vector.append(0.0)
        return np.array(feature_vector).reshape(1, -1)


    def _draw_face_contour(self, frame, landmarks):
        """Dibuja un contorno alrededor de la cara usando los landmarks clave."""
        # Coordenadas de los landmarks clave para el contorno facial
        nose = [landmarks.landmark[self.mp_pose_solutions.PoseLandmark.NOSE].x, landmarks.landmark[self.mp_pose_solutions.PoseLandmark.NOSE].y]
        left_eye = [landmarks.landmark[self.mp_pose_solutions.PoseLandmark.LEFT_EYE].x, landmarks.landmark[self.mp_pose_solutions.PoseLandmark.LEFT_EYE].y]
        right_eye = [landmarks.landmark[self.mp_pose_solutions.PoseLandmark.RIGHT_EYE].x, landmarks.landmark[self.mp_pose_solutions.PoseLandmark.RIGHT_EYE].y]
        left_ear = [landmarks.landmark[self.mp_pose_solutions.PoseLandmark.LEFT_EAR].x, landmarks.landmark[self.mp_pose_solutions.PoseLandmark.LEFT_EAR].y]
        right_ear = [landmarks.landmark[self.mp_pose_solutions.PoseLandmark.RIGHT_EAR].x, landmarks.landmark[self.mp_pose_solutions.PoseLandmark.RIGHT_EAR].y]

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

    def _process_video_frame(self, frame_bgr):
        """Procesa un único frame: detecta pose, extrae características y dibuja contornos faciales."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # Optimización
        pose_detection_results = self.pose_estimator.process(frame_rgb)
        frame_rgb.flags.writeable = True # Re-habilitar escritura
        frame_bgr_annotated = frame_bgr.copy() # Trabajar sobre una copia para dibujar

        if pose_detection_results.pose_landmarks:
            self.mp_drawing_utilities.draw_landmarks(
                frame_bgr_annotated,
                pose_detection_results.pose_landmarks,
                self.mp_pose_solutions.POSE_CONNECTIONS,
                self.mp_drawing_utilities.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                self.mp_drawing_utilities.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=6)
            )
            self._draw_face_contour(frame_bgr_annotated, pose_detection_results.pose_landmarks)

            if self.is_prediction_active and self.trained_model:
                extracted_landmarks = self._extract_pose_landmarks(pose_detection_results.pose_landmarks)
                calculated_angles = self._compute_body_angles(extracted_landmarks)
                
                current_features = self._construct_feature_vector(extracted_landmarks, calculated_angles)
                scaled_features = self.feature_scaler.transform(current_features)
                
                raw_prediction = self.trained_model.predict(scaled_features)[0]
                predicted_class_label = self.label_class_encoder.inverse_transform([raw_prediction])[0]
                
                class_probabilities = self.trained_model.predict_proba(scaled_features)[0]
                prediction_confidence = np.max(class_probabilities) * 100

                # Lógica de buffer y actualización de GUI
                self.prediction_log.append(predicted_class_label)
                if len(self.prediction_log) > self.prediction_log_max_size:
                    self.prediction_log.pop(0)
                
                if self.prediction_log:
                    most_frequent_prediction = Counter(self.prediction_log).most_common(1)[0][0]
                    
                    if prediction_confidence >= self.display_confidence_threshold:
                        self.predicted_activity_label.config(text=f"{most_frequent_prediction.capitalize()}")
                        self.prediction_confidence_label.config(text=f"{prediction_confidence:.1f}%")
                    else:
                        self.predicted_activity_label.config(text="Analizando...")
                        self.prediction_confidence_label.config(text=f"{prediction_confidence:.1f}% (Baja)")
                
        return frame_bgr_annotated


    def _gui_video_update_loop(self):
        """Bucle principal para capturar y mostrar frames de video."""
        if self.video_stream and self.video_stream.isOpened():
            success, frame = self.video_stream.read()
            if success:
                processed_frame = self._process_video_frame(frame)
                
                # Ajustar tamaño para mostrar en GUI manteniendo relación de aspecto
                h, w, _ = processed_frame.shape
                display_w = 640 # Ancho deseado para mostrar
                display_h = int(h * (display_w / w))
                
                img_resized = cv2.resize(processed_frame, (display_w, display_h))
                img_rgb_for_gui = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                pil_image = Image.fromarray(img_rgb_for_gui)
                imgtk_image = ImageTk.PhotoImage(image=pil_image)

                self.video_output_label.imgtk = imgtk_image
                self.video_output_label.configure(image=imgtk_image)
            else: # Si el video es un archivo y termina
                if isinstance(self.video_capture_source, str): # Es un path de archivo
                    self._reset_prediction_labels()
                    messagebox.showinfo("Información", "El video ha terminado.")
                    self.video_stream.release()

        self.root.after(15, self._gui_video_update_loop)


    def _toggle_prediction_state(self):
        """Activa o desactiva el proceso de predicción."""
        self.is_prediction_active = not self.is_prediction_active
        if self.is_prediction_active:
            self.toggle_predict_button.config(text="Detener Predicción")
            self.status_indicator_label.config(text="Estado: Prediciendo", foreground="green")
        else:
            self.toggle_predict_button.config(text="Iniciar Predicción")
            self.status_indicator_label.config(text="Estado: Detenido", foreground="red")
            self._reset_prediction_labels()


    def _reset_prediction_labels(self):
        """Resetea las etiquetas de predicción en la GUI."""
        self.predicted_activity_label.config(text="---")
        self.prediction_confidence_label.config(text="---")
        self.prediction_log.clear()


    def launch(self):
        """Inicia la aplicación: configura el stream de video y el bucle de la GUI."""
        self._initialize_video_stream()
        self._gui_video_update_loop()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing_application)
        self.root.mainloop()


    def _on_closing_application(self):
        """Acciones a realizar al cerrar la ventana de la aplicación."""
        if messagebox.askokcancel("Salir", "¿Seguro que quieres salir?"):
            if self.video_stream and self.video_stream.isOpened():
                self.video_stream.release()
            cv2.destroyAllWindows()
            self.root.destroy()

def main_app_entry_point():
    """Punto de entrada para la aplicación."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    
    root = tk.Tk()
    app_instance = PosePredictorApp(root)
    app_instance.launch()

if __name__ == "__main__":
    main_app_entry_point()
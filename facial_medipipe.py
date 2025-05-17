import cv2
import mediapipe as mp
import time
import math
import numpy as np

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Índices para landmarks importantes (ajustados para MediaPipe Face Mesh)
# Ojos
LEFT_EYE = [362, 385, 387, 263, 373, 380]  # Puntos del ojo izquierdo
RIGHT_EYE = [33, 160, 158, 133, 153, 144]  # Puntos del ojo derecho

# Boca para detectar bostezos - puntos más precisos
UPPER_LIP = 13   # Punto central del labio superior
LOWER_LIP = 14   # Punto central del labio inferior
MOUTH_LEFT = 78  # Comisura izquierda
MOUTH_RIGHT = 308  # Comisura derecha

# Contorno interior de la boca para detección de área
INNER_MOUTH_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

# Función para calcular distancia euclidiana
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Función para calcular EAR (Eye Aspect Ratio)
def eye_aspect_ratio(landmarks, eye_indices):
    # Puntos verticales
    A = euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    
    # Puntos horizontales
    C = euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    
    # Calcular EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Función mejorada para calcular MAR (Mouth Aspect Ratio)
def mouth_aspect_ratio(landmarks):
    # Distancia vertical entre labios
    vertical_dist = euclidean_distance(landmarks[UPPER_LIP], landmarks[LOWER_LIP])
    
    # Distancia horizontal entre comisuras
    horizontal_dist = euclidean_distance(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
    
    # Calcular MAR (mejorado)
    if horizontal_dist == 0:
        return 0
    
    mar = vertical_dist / horizontal_dist
    return mar

# Función alternativa para detectar bostezos basada en área
def detect_yawn_by_area(landmarks, frame_height, frame_width):
    try:
        # Obtener puntos del contorno de la boca
        mouth_contour = np.array([landmarks[idx] for idx in INNER_MOUTH_INDICES], dtype=np.int32)
        
        # Calcular área del contorno
        area = cv2.contourArea(mouth_contour)
        
        # Normalizar el área respecto al tamaño de la cara
        left_eye_center = landmarks[LEFT_EYE[0]]
        right_eye_center = landmarks[RIGHT_EYE[0]]
        eye_distance = euclidean_distance(left_eye_center, right_eye_center)
        
        # Evitar división por cero
        if eye_distance == 0:
            return False, 0
            
        normalized_area = area / (eye_distance ** 2)
        
        # Visualizar el contorno para depuración (opcional)
        debug_frame = None  # Esto se puede activar para depuración
        
        return normalized_area > MOUTH_AREA_THRESH, normalized_area
    except:
        return False, 0

# Fase de calibración para el umbral de bostezos
def calibrate_yawn_threshold(cap, face_mesh, num_frames=30):
    mar_values = []
    area_values = []
    
    print("Mantenga la boca cerrada naturalmente durante la calibración...")
    print("Comenzando calibración en 3 segundos...")
    time.sleep(3)  # Dar tiempo al usuario para prepararse
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = []
            for id, lm in enumerate(results.multi_face_landmarks[0].landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))
                
            # Calcular MAR
            mar = mouth_aspect_ratio(landmarks)
            mar_values.append(mar)
            
            # Calcular área para método alternativo
            try:
                _, area = detect_yawn_by_area(landmarks, h, w)
                area_values.append(area)
            except:
                pass
            
            cv2.putText(frame, f"Calibrando... {i+1}/{num_frames}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Calibración', frame)
        cv2.waitKey(100)
    
    # Calcular umbrales usando media + factor * desviación estándar
    if mar_values:
        mean_mar = np.mean(mar_values)
        std_mar = np.std(mar_values)
        mar_thresh = mean_mar + 2.5 * std_mar
        print(f"Umbral MAR calibrado: {mar_thresh:.3f}")
    else:
        mar_thresh = 0.6
        print(f"Calibración MAR fallida, usando valor por defecto: {mar_thresh}")
    
    if area_values:
        mean_area = np.mean(area_values)
        std_area = np.std(area_values)
        area_thresh = mean_area + 3.0 * std_area
        print(f"Umbral de área calibrado: {area_thresh:.3f}")
    else:
        area_thresh = 0.05
        print(f"Calibración de área fallida, usando valor por defecto: {area_thresh}")
    
    return mar_thresh, area_thresh

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Parámetros de detección
EYE_AR_THRESH = 0.18  # Ajusta según sea necesario
EYE_AR_CONSEC_FRAMES = 10
MOUTH_AR_CONSEC_FRAMES = 6

# Parámetros para duración de bostezo
YAWN_TIME_THRESHOLD = 2.0  # Duración mínima para considerar un bostezo real (en segundos)
MOUTH_OPEN_COOLDOWN = 1.0  # Tiempo mínimo entre bostezos para evitar detección múltiple

# Ejecutar calibración
print("¿Desea calibrar los umbrales? (s/n)")
calibrate = input().lower() == 's'

if calibrate:
    MOUTH_AR_THRESH, MOUTH_AREA_THRESH = calibrate_yawn_threshold(cap, face_mesh)
else:
    # Valores por defecto si no se calibra
    MOUTH_AR_THRESH = 0.6
    MOUTH_AREA_THRESH = 0.05
    print(f"Usando umbrales por defecto: MAR={MOUTH_AR_THRESH}, Área={MOUTH_AREA_THRESH}")

# Contadores para detección
eye_counter = 0
mouth_counter = 0
head_tilt_counter = 0

# Variables para alertas
drowsy_alert_time = 0
yawn_alert_time = 0
head_pos_alert_time = 0

# Variables para detectar duración de bostezo
mouth_open_start_time = 0
mouth_open_duration = 0
is_mouth_open = False
is_currently_yawning = False
last_yawn_time = 0
yawn_in_progress = False

# Variables para FPS
prev_time = 0
frame_count = 0
fps = 0

# Variable para depuración
debug_mode = False

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame.")
            break
        
        # Calcular FPS
        current_time = time.time()
        frame_count += 1
        if (current_time - prev_time) >= 1:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time
        
        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Procesar con MediaPipe
        results = face_mesh.process(rgb_frame)
        
        # Mostrar FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Mostrar umbrales
        cv2.putText(frame, f"MAR Thresh: {MOUTH_AR_THRESH:.3f}", (20, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Area Thresh: {MOUTH_AREA_THRESH:.3f}", (20, 430), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Procesar resultados de MediaPipe
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Dibujar malla facial (solo si no está en modo de depuración)
                if not debug_mode:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
                
                # Extraer coordenadas de landmarks
                landmarks = []
                for id, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append((x, y))
                
                # Calcular EAR para detectar ojos cerrados
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0
                
                # Calcular MAR para detectar bostezos (método 1)
                mar = mouth_aspect_ratio(landmarks)
                
                # Detectar bostezos por área (método 2)
                yawn_by_area, area_value = detect_yawn_by_area(landmarks, h, w)
                
                # Mostrar valores EAR, MAR y Área
                cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {area_value:.4f}", (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                
                # Detectar ojos cerrados
                if ear < EYE_AR_THRESH:
                    eye_counter += 1
                    if eye_counter >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "ALERTA: Ojos cerrados!", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Alerta por fatiga
                        current_time = time.time()
                        if current_time - drowsy_alert_time > 3:
                            print("¡ALERTA! Conductor con fatiga detectada - ojos cerrados.")
                            drowsy_alert_time = current_time
                else:
                    eye_counter = max(0, eye_counter - 1)
                
                # Detección de boca abierta (basado en ambos métodos)
                mouth_open_detected = (mar > MOUTH_AR_THRESH) or yawn_by_area
                
                # Lógica mejorada para detección de bostezo basada en duración
                if mouth_open_detected:
                    # Si la boca acaba de abrirse, registrar el tiempo de inicio
                    if not is_mouth_open:
                        mouth_open_start_time = current_time
                        is_mouth_open = True
                    
                    # Calcular duración actual de boca abierta
                    mouth_open_duration = current_time - mouth_open_start_time
                    
                    # Mostrar duración de apertura de boca
                    cv2.putText(frame, f"Mouth open: {mouth_open_duration:.1f}s", (20, 350),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
                    
                    # Si la boca ha estado abierta por suficiente tiempo y no hay un bostezo en progreso
                    if mouth_open_duration >= YAWN_TIME_THRESHOLD and not yawn_in_progress:
                        if current_time - last_yawn_time > MOUTH_OPEN_COOLDOWN:
                            yawn_in_progress = True
                            cv2.putText(frame, "ALERTA: Bostezo detectado!", (10, 110),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # Alerta por bostezo
                            if current_time - yawn_alert_time > 3:
                                print("¡ALERTA! Conductor con fatiga detectada - bostezo.")
                                yawn_alert_time = current_time
                                last_yawn_time = current_time
                else:
                    # Reiniciar variables si la boca se cierra
                    is_mouth_open = False
                    mouth_open_duration = 0
                    
                    # Reiniciar estado de bostezo cuando la boca se cierra completamente
                    if yawn_in_progress:
                        yawn_in_progress = False
                
                # Detección de posición de la cabeza (básica)
                # Usando landmarks de las orejas
                left_ear_point = landmarks[234]
                right_ear_point = landmarks[454]
                
                # Calcular inclinación horizontal
                y_diff = abs(left_ear_point[1] - right_ear_point[1])
                head_tilt = y_diff / h  # Normalizado por la altura del frame
                
                # Mostrar valor de inclinación
                cv2.putText(frame, f"Tilt: {head_tilt:.3f}", (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                
                if head_tilt > 0.03:  # Umbral para inclinación de cabeza
                    head_tilt_counter += 1
                    if head_tilt_counter >= 10:  # Si la cabeza está inclinada por 10 frames
                        cv2.putText(frame, "ALERTA: Cabeza inclinada!", (10, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Alerta por posición de cabeza
                        current_time = time.time()
                        if current_time - head_pos_alert_time > 3:
                            print("¡ALERTA! Posición anormal de la cabeza detectada.")
                            head_pos_alert_time = current_time
                else:
                    head_tilt_counter = max(0, head_tilt_counter - 1)
                
                # Visualizar contorno de boca para depuración
                if debug_mode:
                    mouth_contour = np.array([landmarks[idx] for idx in INNER_MOUTH_INDICES], dtype=np.int32)
                    cv2.drawContours(frame, [mouth_contour], -1, (0, 255, 255), 2)
                    
                    # Dibujar puntos específicos de la boca
                    cv2.circle(frame, landmarks[UPPER_LIP], 3, (255, 0, 0), -1)
                    cv2.circle(frame, landmarks[LOWER_LIP], 3, (0, 0, 255), -1)
                    cv2.circle(frame, landmarks[MOUTH_LEFT], 3, (0, 255, 0), -1)
                    cv2.circle(frame, landmarks[MOUTH_RIGHT], 3, (0, 255, 0), -1)
        
        # Mostrar frame
        cv2.imshow('Detector de Fatiga', frame)
        
        # Comandos de teclado
        key = cv2.waitKey(1) & 0xFF
        
        # Salir con 'q'
        if key == ord('q'):
            break
        # Cambiar modo de depuración con 'd'
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Modo de depuración: {'Activado' if debug_mode else 'Desactivado'}")
        # Recalibrar con 'c'
        elif key == ord('c'):
            MOUTH_AR_THRESH, MOUTH_AREA_THRESH = calibrate_yawn_threshold(cap, face_mesh)
        # Ajustar umbrales manualmente
        elif key == ord('a'):  # Disminuir MAR
            MOUTH_AR_THRESH = max(0.1, MOUTH_AR_THRESH - 0.05)
            print(f"MAR Threshold: {MOUTH_AR_THRESH:.2f}")
        elif key == ord('s'):  # Aumentar MAR
            MOUTH_AR_THRESH += 0.05
            print(f"MAR Threshold: {MOUTH_AR_THRESH:.2f}")
        elif key == ord('z'):  # Disminuir Área
            MOUTH_AREA_THRESH = max(0.01, MOUTH_AREA_THRESH - 0.01)
            print(f"Area Threshold: {MOUTH_AREA_THRESH:.3f}")
        elif key == ord('x'):  # Aumentar Área
            MOUTH_AREA_THRESH += 0.01
            print(f"Area Threshold: {MOUTH_AREA_THRESH:.3f}")
        # Ajustar tiempo mínimo para bostezo
        elif key == ord('t'):  # Disminuir tiempo
            YAWN_TIME_THRESHOLD = max(0.5, YAWN_TIME_THRESHOLD - 0.5)
            print(f"Yawn Time Threshold: {YAWN_TIME_THRESHOLD:.1f} segundos")
        elif key == ord('y'):  # Aumentar tiempo
            YAWN_TIME_THRESHOLD += 0.5
            print(f"Yawn Time Threshold: {YAWN_TIME_THRESHOLD:.1f} segundos")

except Exception as e:
    print(f"Error: {e}")
finally:
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
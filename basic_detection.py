import cv2
import dlib
import numpy as np
import argparse
import time

class FaceLandmarkDetector:
    def __init__(self, predictor_path):
        # Inicializar detector de rostros y predictor de landmarks
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Definir grupos de puntos faciales
        self.FACIAL_LANDMARKS_INDEXES = {
            "jaw": list(range(0, 17)),
            "right_eyebrow": list(range(17, 22)),
            "left_eyebrow": list(range(22, 27)),
            "nose": list(range(27, 36)),
            "right_eye": list(range(36, 42)),
            "left_eye": list(range(42, 48)),
            "mouth": list(range(48, 68))
        }
    
    def shape_to_np(self, shape, dtype="int"):
        """Convertir shape de dlib a numpy array"""
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def detect_landmarks(self, frame):
        """Detectar landmarks en el frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        landmarks_list = []
        for face in faces:
            shape = self.predictor(gray, face)
            shape = self.shape_to_np(shape)
            landmarks_list.append(shape)
        
        return landmarks_list, faces
    
    def draw_landmarks(self, frame, landmarks_list, faces):
        """Dibujar landmarks en el frame"""
        for landmarks, face in zip(landmarks_list, faces):
            # Dibujar rectángulo del rostro
            cv2.rectangle(frame, (face.left(), face.top()), 
                         (face.right(), face.bottom()), (0, 255, 0), 2)
            
            # Dibujar puntos de landmarks
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            # Dibujar conexiones opcionales
            self.draw_face_connections(frame, landmarks)
        
        return frame
    
    def draw_face_connections(self, frame, landmarks):
        """Dibujar conexiones entre puntos faciales"""
        # Contorno de la cara
        jaw_points = landmarks[self.FACIAL_LANDMARKS_INDEXES["jaw"]]
        cv2.polylines(frame, [jaw_points], False, (255, 255, 0), 1)
        
        # Cejas
        right_eyebrow = landmarks[self.FACIAL_LANDMARKS_INDEXES["right_eyebrow"]]
        left_eyebrow = landmarks[self.FACIAL_LANDMARKS_INDEXES["left_eyebrow"]]
        cv2.polylines(frame, [right_eyebrow], False, (255, 255, 0), 1)
        cv2.polylines(frame, [left_eyebrow], False, (255, 255, 0), 1)
        
        # Ojos
        right_eye = landmarks[self.FACIAL_LANDMARKS_INDEXES["right_eye"]]
        left_eye = landmarks[self.FACIAL_LANDMARKS_INDEXES["left_eye"]]
        cv2.polylines(frame, [right_eye], True, (255, 255, 0), 1)
        cv2.polylines(frame, [left_eye], True, (255, 255, 0), 1)
        
        # Nariz
        nose = landmarks[self.FACIAL_LANDMARKS_INDEXES["nose"]]
        cv2.polylines(frame, [nose], False, (255, 255, 0), 1)
        
        # Boca
        mouth = landmarks[self.FACIAL_LANDMARKS_INDEXES["mouth"]]
        cv2.polylines(frame, [mouth], True, (255, 255, 0), 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--shape-predictor", required=True,
                       help="path to facial landmark predictor")
    parser.add_argument("-c", "--camera", type=int, default=0,
                       help="camera index")
    args = parser.parse_args()
    
    # Inicializar detector
    detector = FaceLandmarkDetector(args.shape_predictor)
    
    # Inicializar cámara
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Variables para FPS
    fps_counter = 0
    start_time = time.time()
    
    print("Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar landmarks
        landmarks_list, faces = detector.detect_landmarks(frame)
        
        # Dibujar landmarks
        frame = detector.draw_landmarks(frame, landmarks_list, faces)
        
        # Calcular y mostrar FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = time.time()
            print(f"FPS: {fps:.2f}")
        
        # Mostrar información en pantalla
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow("Face Landmarks", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
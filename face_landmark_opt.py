import cv2
import dlib
import numpy as np
import argparse
import time
from threading import Thread
import queue

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.q = queue.Queue()
        self.running = True
        
    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self
        
    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                break
                
            if not self.q.empty():
                self.q.get()
            self.q.put(frame)
            
    def read(self):
        return self.q.get()
    
    def stop(self):
        self.running = False
        self.thread.join()

class OptimizedFaceLandmarkDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Configuración para optimización
        self.skip_frames = 2  # Procesar cada 3 frames
        self.frame_count = 0
        self.last_landmarks = []
        self.last_faces = []
        
    def detect_landmarks_optimized(self, frame):
        """Detección optimizada con skip de frames"""
        self.frame_count += 1
        
        # Solo procesar cada N frames
        if self.frame_count % self.skip_frames == 0:
            # Redimensionar para procesamiento más rápido
            small_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.detector(gray)
            landmarks_list = []
            
            for face in faces:
                # Escalar coordenadas de vuelta al tamaño original
                face_scaled = dlib.rectangle(
                    face.left() * 2, face.top() * 2,
                    face.right() * 2, face.bottom() * 2
                )
                
                # Detectar landmarks en frame original
                gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                shape = self.predictor(gray_original, face_scaled)
                coords = np.zeros((68, 2), dtype="int")
                for i in range(0, 68):
                    coords[i] = (shape.part(i).x, shape.part(i).y)
                landmarks_list.append(coords)
            
            # Actualizar cache
            self.last_landmarks = landmarks_list
            self.last_faces = [dlib.rectangle(f.left() * 2, f.top() * 2, 
                                            f.right() * 2, f.bottom() * 2) 
                              for f in faces]
        
        return self.last_landmarks, self.last_faces

def main_optimized():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--shape-predictor", required=True,
                       help="path to facial landmark predictor")
    parser.add_argument("-c", "--camera", type=int, default=0,
                       help="camera index")
    args = parser.parse_args()
    
    # Inicializar detector optimizado
    detector = OptimizedFaceLandmarkDetector(args.shape_predictor)
    
    # Inicializar cámara con threading
    camera = ThreadedCamera(args.camera).start()
    
    print("Presiona 'q' para salir")
    
    try:
        while True:
            frame = camera.read()
            
            # Detectar landmarks
            landmarks_list, faces = detector.detect_landmarks_optimized(frame)
            
            # Dibujar landmarks
            for landmarks, face in zip(landmarks_list, faces):
                cv2.rectangle(frame, (face.left(), face.top()), 
                             (face.right(), face.bottom()), (0, 255, 0), 2)
                
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Optimized Face Landmarks", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_optimized()
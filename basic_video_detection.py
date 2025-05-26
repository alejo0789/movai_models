import cv2
import dlib
import numpy as np
import time
import os
import argparse # Added argparse for command-line arguments in a standalone script

class FaceLandmarkDetector:
    def __init__(self, predictor_path):
        """
        Initializes the FaceLandmarkDetector with a dlib face detector and a shape predictor.
        Args:
            predictor_path (str): Path to the dlib shape predictor model file (e.g., shape_predictor_68_face_landmarks.dat).
        """
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Shape predictor file not found at: {predictor_path}")
            
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Define groups of facial landmark indexes for easier access and drawing
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
        """
        Converts dlib's shape object (68 facial landmarks) to a NumPy array.
        Args:
            shape (dlib.full_object_detection): The 68-point facial landmark prediction.
            dtype (str): The data type for the NumPy array (default: "int").
        Returns:
            numpy.ndarray: A 68x2 NumPy array of (x, y) coordinates for each landmark.
        """
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def detect_landmarks(self, frame):
        """
        Detects faces and their corresponding 68 facial landmarks in a given frame.
        Args:
            frame (numpy.ndarray): The input image frame (BGR format).
        Returns:
            tuple: A tuple containing:
                - landmarks_list (list): A list of NumPy arrays, where each array contains the 68 landmarks for a detected face.
                - faces (dlib.rectangles): A list of dlib.rectangle objects, each representing a detected face.
        """
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale image
        faces = self.detector(gray)
        
        landmarks_list = []
        for face in faces:
            # Predict the 68 facial landmarks for each detected face
            shape = self.predictor(gray, face)
            # Convert the dlib shape object to a NumPy array
            shape = self.shape_to_np(shape)
            landmarks_list.append(shape)
        
        return landmarks_list, faces
    
    def draw_landmarks(self, frame, landmarks_list, faces):
        """
        Draws the detected facial landmarks and face bounding boxes on the frame.
        Args:
            frame (numpy.ndarray): The input image frame (BGR format).
            landmarks_list (list): A list of NumPy arrays, where each array contains the 68 landmarks for a detected face.
            faces (dlib.rectangles): A list of dlib.rectangle objects, each representing a detected face.
        Returns:
            numpy.ndarray: The frame with drawn landmarks and bounding boxes.
        """
        # Create a copy to draw on, so the original frame is not modified if needed elsewhere
        drawn_frame = frame.copy() 
        for landmarks, face in zip(landmarks_list, faces):
            # Draw the bounding box around the detected face
            cv2.rectangle(drawn_frame, (face.left(), face.top()), 
                          (face.right(), face.bottom()), (0, 255, 0), 2) # Green box
            
            # Draw each individual landmark point
            for (x, y) in landmarks:
                cv2.circle(drawn_frame, (x, y), 2, (0, 0, 255), -1) # Red circles
            
            # Draw connections between facial landmark points for better visualization
            self.draw_face_connections(drawn_frame, landmarks)
        
        return drawn_frame
    
    def draw_face_connections(self, frame, landmarks):
        """
        Draws lines connecting specific facial landmark points to form outlines of facial features.
        Args:
            frame (numpy.ndarray): The input image frame (BGR format).
            landmarks (numpy.ndarray): A NumPy array containing the 68 landmarks for a single face.
        """
        # Jawline contour
        jaw_points = landmarks[self.FACIAL_LANDMARKS_INDEXES["jaw"]]
        cv2.polylines(frame, [jaw_points], False, (255, 255, 0), 1) # Yellow lines
        
        # Eyebrows
        right_eyebrow = landmarks[self.FACIAL_LANDMARKS_INDEXES["right_eyebrow"]]
        left_eyebrow = landmarks[self.FACIAL_LANDMARKS_INDEXES["left_eyebrow"]]
        cv2.polylines(frame, [right_eyebrow], False, (255, 255, 0), 1)
        cv2.polylines(frame, [left_eyebrow], False, (255, 255, 0), 1)
        
        # Eyes (closed polygons)
        right_eye = landmarks[self.FACIAL_LANDMARKS_INDEXES["right_eye"]]
        left_eye = landmarks[self.FACIAL_LANDMARKS_INDEXES["left_eye"]] 
        cv2.polylines(frame, [right_eye], True, (255, 255, 0), 1)
        cv2.polylines(frame, [left_eye], True, (255, 255, 0), 1)
        
        # Nose
        nose = landmarks[self.FACIAL_LANDMARKS_INDEXES["nose"]]
        cv2.polylines(frame, [nose], False, (255, 255, 0), 1)
        
        # Mouth (closed polygon)
        mouth = landmarks[self.FACIAL_LANDMARKS_INDEXES["mouth"]]
        cv2.polylines(frame, [mouth], True, (255, 255, 0), 1)

def main():
    parser = argparse.ArgumentParser(description="Real-time Face Landmark Detection with Video Output")
    parser.add_argument("-p", "--shape-predictor", required=True,
                        help="path to facial landmark predictor (e.g., shape_predictor_68_face_landmarks.dat)")
    parser.add_argument("-c", "--camera", type=int, default=0,
                        help="camera index (default: 0)")
    parser.add_argument("-o", "--output-video", type=str, default="output_video.avi",
                        help="path to save the output video file (e.g., output_video.avi)")
    args = parser.parse_args()
    
    detector = FaceLandmarkDetector(args.shape_predictor)
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {args.camera}.")
        return

    # Get camera properties for video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Fallback if FPS is not properly reported by camera
        fps = 30.0 

    # Define the codec and create VideoWriter object
    # For .avi, typically use XVID (Windows) or MJPG (cross-platform)
    # For .mp4, use H264 or MP4V (requires specific OpenCV build/ffmpeg)
    # Adjust codec based on your system and desired output format
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Use MJPG for broader compatibility with .avi
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for file {args.output_video}.")
        cap.release()
        return

    print(f"Processing video stream and saving to {args.output_video}...")
    print("Press Ctrl+C to stop the script.")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break
            
            # Detect landmarks
            landmarks_list, faces = detector.detect_landmarks(frame)
            
            # Draw landmarks on the frame
            processed_frame = detector.draw_landmarks(frame, landmarks_list, faces)
            
            # Add text overlay for faces detected
            cv2.putText(processed_frame, f"Faces: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write the processed frame to the output video file
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0: # Update FPS every 30 frames
                end_time = time.time()
                current_fps = 30 / (end_time - start_time)
                start_time = time.time()
                print(f"Frames processed: {frame_count}, Current FPS: {current_fps:.2f}, Faces detected: {len(faces)}")
            
    except KeyboardInterrupt:
        print("\nScript stopped by user (Ctrl+C).")
    finally:
        # Release everything when job is finished
        cap.release()
        out.release()
        print("Camera and video writer released.")

if __name__ == "__main__":
    main()

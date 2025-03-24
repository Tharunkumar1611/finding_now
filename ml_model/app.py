from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi'}

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def process_video(video_path, reference_face_encoding):
    video_capture = cv2.VideoCapture(video_path)
    face_found = False
    timestamp = None  # Initialize timestamp as None
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Get current timestamp of the frame in seconds
        current_frame_time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([reference_face_encoding], face_encoding)
            if True in matches:
                face_found = True
                timestamp = current_frame_time  # Save the timestamp when the face is found
                break  # Exit the loop if a match is found
        
        if face_found:
            break  # Exit the outer loop if a match is found
    
    video_capture.release()
    return face_found, timestamp  # Return face_found status and timestamp

@app.route('/check-face', methods=['POST'])
def check_face():
    print("Incoming request files:", request.files)
    
    if 'photo' not in request.files or 'video' not in request.files:
        print("Missing files in request")
        return jsonify({'error': 'Missing files'}), 400
    
    photo = request.files['photo']
    video = request.files['video']
    
    print("Received photo filename:", photo.filename)
    print("Received video filename:", video.filename)
    
    if not allowed_image_file(photo.filename):
        print("Invalid image file format")
        return jsonify({'error': 'Invalid image file format. Allowed formats: png, jpg, jpeg'}), 400
    
    if not allowed_video_file(video.filename):
        print("Invalid video file format")
        return jsonify({'error': 'Invalid video file format. Allowed formats: mp4, mov, avi'}), 400
    
    photo_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(photo.filename))
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video.filename))
    
    photo.save(photo_path)
    video.save(video_path)
    
    print("Photo saved to:", photo_path)
    print("Video saved to:", video_path)
    
    reference_image = face_recognition.load_image_file(photo_path)
    reference_face_encodings = face_recognition.face_encodings(reference_image)
    
    if not reference_face_encodings:
        print("No face found in the photo")
        return jsonify({'error': 'No face found in the photo'}), 400
    
    face_found, timestamp = process_video(video_path, reference_face_encodings[0])
    
    if face_found:
        result = {
            'result': 'Face Found',
            'timestamp': f'{timestamp:.2f} seconds'
        }
    else:
        result = {
            'result': 'Not Found'
        }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
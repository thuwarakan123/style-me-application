from mtcnn import MTCNN
import cv2
import numpy as np

# Loading the pre-trained face detector model
face_detector = MTCNN()

# List of allowed image extension 
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# This method used to check the face count of the input image 
def get_face_count(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb)
    if not faces:
        face_count = 0
    else:
        face_count = len(faces)  
    return face_count


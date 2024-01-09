from mtcnn import MTCNN
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Loading the pre-trained face detector model
face_detector = MTCNN()

# Loading the gender identification model 
gender_detector = load_model('models/gender_classification_model_EfficientNetV2M_Version_01.h5')

# This method used to identify the gender and return it.
def get_gender(faceImage):
    # This code used to detect the face and crop the head.
    face = face_detector.detect_faces(faceImage)
    for result in face:
        x, y, width, height = result['box']

        head_width = int(1.2 * width)
        head_height = int(1.6 * height)
        head_x = int(x - 0.1 * width)
        head_y = int(y - 0.5 * height)

        head = faceImage[max(head_y, 0):head_y+head_height, max(head_x, 0):head_x+head_width]

        # Saving the processed image.
        cv2.imwrite("static/images/preprocessed/preprocessed_image2.jpg", head)
    
    # In here we are loading image and pre procesing.
    image = load_img("static/images/preprocessed/preprocessed_image2.jpg", target_size=(224, 224))
    image = img_to_array(image)
    # Normalize pixel values
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)
 
    # Predicting the gender 
    predictions = gender_detector.predict(image)
    index = np.argmax(predictions, axis=1)
    return index[0]
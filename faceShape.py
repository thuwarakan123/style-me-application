from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_input
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet_input
import numpy as np

# Loading a face prediction model 
face_shape_predictor = load_model('models/face_shape_classification_model_Ensemble_Version_03.h5')

# List of face shapes
face_shape = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# This method used to predict the face shape and return it
def predict_face_shape():

    # In here we are loading image and pre procesing
    image = load_img("static/images/preprocessed/preprocessed_image2.jpg", target_size=(224, 224))
    image = img_to_array(image)
    # Normalize pixel values
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)

    # Predicting the face shape
    predictions = face_shape_predictor.predict(image)
    index = np.argmax(predictions, axis=1)
    return index[0]
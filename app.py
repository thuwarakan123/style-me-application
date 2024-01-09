from flask import Flask, render_template, request
from otherMethods import get_face_count, allowed_file
from gender import get_gender
from faceShape import predict_face_shape 
from data import hair_style_detail, beard_style_detail
import numpy as np
import cv2
import logging

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

#open help page
@app.route('/help')
def help():
    return render_template('help.html')

#open about page
@app.route('/about')
def about():
    return render_template('about.html')

# Api for suggesting hair and beard style
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        # Get the uploaded file from the form data
        file = request.files['file']

        # Check the file format and if it not allowed it will return error
        if not allowed_file(file.filename):
            return render_template('home.html', message ='Invalid file format. Please upload an image file (jpg, jpeg, png, gif).')

        # Convert file data to a numpy array
        file_data = np.fromstring(file.read(), np.uint8)

        # Decode the image data to a Mat object
        img = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
        cv2.imwrite("static/images/preprocessed/preprocessed_image.jpg", img)
        
        # Check if file has allowed extension
        if not allowed_file(file.filename):
            return render_template('home.html', message ='Invalid file format. Please upload an image file (jpg, jpeg, png, gif).')

        #getting face count for use the prediction 
        faceCount = get_face_count(img)
        print(faceCount)
        # Base on the face count show error messages 
        if(faceCount == 0):
            return render_template('home.html', message = "No faces detected in the image.")
        elif(faceCount > 1):
            return render_template('home.html', message = "There are many faces detected! please add a image which contain one face")
        else:
            # Getting the user gender 
            genderIdx = get_gender(img)

            # Checking if the user click the predict face shape button
            if request.form['action'] == "task1":
                # Getting the face shape 
                faceShape_idx  = predict_face_shape()
                # Getting the hair style based on the users gender and face shape
                data = hair_style_detail[genderIdx][faceShape_idx]
                return render_template('home.html', prediction = data)
            else:
                # Checking if the user is male or not
                if genderIdx  == 0:
                    # Getting the face shape 
                    faceShape_idx  = predict_face_shape()
                    # Getting the beard style based on the users face shape
                    data = beard_style_detail[faceShape_idx]
                    return render_template('home.html', prediction = data )
                else:
                    return render_template('home.html', message = "Please add men's images to suggest a suitable beard style.")
            

if __name__ == "__main__":
    app.run(debug=True)
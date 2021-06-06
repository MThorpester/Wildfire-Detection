
#Import Flast Library
from flask import Flask, jsonify,url_for

#Import datetime library  
import datetime as dt

#Import Tensorflow and Image Processing libraries
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2;
from PIL import Image
from tensorflow.keras.applications.xception import preprocess_input

#Miscellaneous Libraries
import os


########################################################################
#
#   LOAD MODE
#
########################################################################

#Load Model
model = keras.models.load_model('MediumXception', compile=True)
model.summary()

########################################################################
#
#   USER DEFINED FUNCTIONS
#
########################################################################

def getMyPrediction(name,model):
    image = tf.keras.preprocessing.image.load_img(name)
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch. 
    # Note: Xception expects a specific kind of input processing.
    # Before passing inputs to the model, call tf.keras.applications.xception.preprocess_input. 
    # It will scale scale input pixels between -1 and 1.
    x = preprocess_input(input_arr)
    predictions = model.predict(x)
    return(predictions[0][0])


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def buildImageArray(img_path):
    img = cv2.imread(img_path)
    img_h, img_w, img_channels = img.shape
    split_width = 299
    split_height = 299
    X_points = start_points(img_w, split_width, 0.01)
    Y_points = start_points(img_h, split_height, 0.01)
    split_array = [];
    count = 0
    for i in Y_points:
        for j in X_points:
            split = img[i:i+split_height, j:j+split_width]
            split_dict = {
                            'img':split,
                            'x':j,
                            'y':i,
                            'width':split_width,
                            'height':split_height
                        };
            split_array.append(split_dict)
#         cv2.imwrite('{}_{}.{}'.format(name, count, frmt), split)
            count += 1
    return split_array;

def predictArrayImages(pred_array, prediction_threshold):
    output_array = []
    output_dict = {}
    index = 0
    while index < len(pred_array):
        input_arr = keras.preprocessing.image.img_to_array(pred_array[index]['img'])
        input_arr = np.array([input_arr])
        x = preprocess_input(input_arr)
        prediction = model.predict(x)
        result = prediction[0][0]
        if result >= prediction_threshold:
            output_dict = {
                            'x': pred_array[index]['x'],
                            'y': pred_array[index]['y'],
                            'width': pred_array[index]['width'],
                            'height': pred_array[index]['height'],
                            'prediction' : str(round(result,4))
                          }
            output_array.append(output_dict)
        index += 1
    return(output_array)


#Configure Flask routes and configuration attributes
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

########################################################################
#
#   APPLICATION ROUTES
#
########################################################################

@app.route("/")
def welcome():
	return(
			f"<h1>Wildfire Predictor API Routes</h1>"
			f"/api/v1.0/predict/&lt;img_directory&gt;/&lt;img_name&gt;/&lt;pred_threshold&gt;<br/>"
	) 

@app.route("/api/v1.0/predict/<img_directory>/<img_name>/<pred_threshold>")
def prediction(img_directory,img_name,pred_threshold):
    img_path = os.path.join(img_directory,img_name)
    image_array = buildImageArray(img_path)
    return_results = predictArrayImages(image_array, float(pred_threshold))
    return jsonify(return_results)


#url_for('prediction', img_directory='D:\smoke_test', img_name='1528057977_+00660.jpg',pred_threshold=".50")


if __name__ == "__main__":
    app.run(debug=True)

#Import Flast Library
from flask import Flask, jsonify

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


#Load Model
model = keras.models.load_model('MediumXception', compile=True)
model.summary()

#Create Custom Functions
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

#Configure Flask routes and configuration attributes
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route("/")
def welcome():
	return(
			f"<h1>Hawaii Weather Station API Routes</h1>"
			f"/api/v1.0/stations<br/>"
			f"/api/v1.0/tobs<br/>"
			f"/api/v1.0/stats/&lt;start&gt;<br/>"
			f"/api/v1.0/stats/&lt;start&gt;/&lt;end&gt;<br/>"
	) 

@app.route("/api/v1.0/test")
def stations():
    name = r'D:\Wildfire\HPWren\HPWREN-FigLib_Output2\1512674224_+00240_centered_1.jpg' 
    return (
             str(getMyPrediction(name,model))
            )



if __name__ == "__main__":
    app.run(debug=True)
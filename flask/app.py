
########################################################################
#
#   IMPORT APPLICATION LIBRARY MODULES
#
########################################################################

#Import Flast Library
from flask import Flask, jsonify,url_for,request
from flask_cors import CORS, cross_origin

#Import datetime library  
import datetime as dt

#Import Tensorflow and Image Processing libraries
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2;
from PIL import Image, ImageDraw,ImageFont
from tensorflow.keras.applications.xception import preprocess_input

#Selenium Libraries
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.chrome.options import Options

#Miscellaneous Libraries
import os
import time

########################################################################
#
#   Application Constants
#
########################################################################
host_str = "http://metavision.tech"

########################################################################
#
#   LOAD MODEL
#
########################################################################

#Load Model
model = keras.models.load_model('Streamline1', compile=True)
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
            timestr = time.strftime("%Y%m%d-%H%M%S")
            filename = 'split' + str(count) + timestr + '.jpg'
            split_dict = {
                            'img':"images/processing/" + filename,
                            'x':j,
                            'y':i,
                            'width':split_width,
                            'height':split_height,
                            'cnt': "img" + str(count)
                
                        };
            split_array.append(split_dict)
            cv2.imwrite("images/processing/" + filename,split)
#           cv2.imwrite('{}_{}.{}'.format(name, count, frmt), split)
            count += 1
    return split_array;


def predictArrayImages(pred_array, prediction_threshold,img_path):
    output_array = []
    output_dict = {}
    index = 0
    smoke_detected = "N"
    url_filename = ""
    file = ""
    filepath = ""
    while index < len(pred_array):
        image = tf.keras.preprocessing.image.load_img(pred_array[index]['img'])
        input_arr = keras.preprocessing.image.img_to_array(image)
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
        if len(output_array) > 0:
            smoke_detected = "Y"
        else:
            smoke_detected = "N"
            
  
    #A call to crop function here in future
    img = Image.open(img_path)
    img1 = ImageDraw.Draw(img)
    print(img_path)
    if len(output_array) >= 1:
        for shape in output_array:
            x,y,width,height,pred = shape.values()
            img1.rectangle([(x,y),(x + width,y + height)],outline="red")
            font = ImageFont.truetype("LiberationMono-Bold.ttf", 30)
            img1.text((x + 80,y + height - 20), pred,font=font, fill="red")
        z = img_path.split('/')
        if len(z) >= 1:
            file = z[len(z) -1]
            print(file)
            filepath = r"/var/www/html/images/bound_images/" + file
            img.save(filepath)
            url_filename = "http://metavision.tech/images/bound_images/" + file       
        else:
            url_filename = img_path
    results = [{
        "img_url": url_filename,
        "smoke_detected": smoke_detected,
        "bounds": output_array}
    ]
    return(results)

def scrapeWildfireImage():
    # Set Browser Options
    options = Options()
    options.headless = True

    #Instantiate Headerless Browser
    driver = webdriver.Chrome(options=options, executable_path=ChromeDriverManager().install())
    url = 'http://www.alertwildfire.org/northcoast/index.html?camera=Axis-DeerHorn2&v=fd40728'
    driver.get(url) 

    #Resize WIndow
    driver.maximize_window()
    driver.set_window_size(1920, 1080)

    #Find Image Element
    source = driver.find_element_by_xpath("//div[@id = 'camera-block-image']/div[@class='leaflet-map-pane']/div[@class='leaflet-objects-pane']/div[@class='leaflet-overlay-pane']/img")

    #Set filename 
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'scrape_' + timestr
    filename_path = os.path.join('images','scrape',filename + ".png")
    filename_path_jpg = os.path.join('images','scrape',filename + ".jpg")   

    #Save PNG Convert To JPEG
    source.screenshot(filename_path)
    im1 = Image.open(r'images/scrape/' + filename + ".png")
    rgb_im = im1.convert('RGB')
    rgb_im.save(r'images/scrape/' + filename + '.jpg')
    export_url = os.path.join(host_str,'images','scrape',filename + '.jpg')

    return ( [{'img_url':export_url}]) 
    
#Configure Flask routes and configuration attributes
app = Flask(__name__)
cors = CORS(app)
app.config['JSON_SORT_KEYS'] = False
app.config['CORS_HEADERS'] = 'Content-Type'


########################################################################
#
#   APPLICATION ROUTES
#
########################################################################

@app.route("/api")
def welcome():
	return(
	    f"<h1>Wildfire Predictor API Routes<h1/>"
            f"<hr/>"
            f"<h2>/api/scrape<h2/>"
            f"<p>Scrapes image and stores scraped image on server.  Image URL sent in JSON response<p/>"
            f"<br/>"
	    f"<h2>/api/predict?img_directory=images&img_name=image.jpg&pred_threshold=.50<h2/>"
            f"<p>Processes server image for smoke. Returns,  Smoke flag, Image URL with boudary frames"
            f"(if smoke detected), prediction probability and smoke boundaries<p/>"
	) 

@app.route("/api/scrape2")
def scrape2():
    return_results = scrapeWildfireImage()
    return jsonify(return_results)

@app.route("/api/scrape")
def scrape():
    return_results = [];
    img_directory = "/var/www/html/images/scrape";
    img_name = "scrape.jpg"
    img_path = os.path.join(img_directory,img_name);
    print(img_path)
    image_array = buildImageArray(img_path)
    return_results = predictArrayImages(image_array,float(.50),img_path);
    if len(return_results[0]['bounds']) < 1:
            return_results = [{
                "img_url":"http://metavision.tech/images/scrape/scrape.jpg",
                "smoke_detected":"N",
                "bounds":[]
                }]
    return jsonify(return_results)

@app.route("/api/predict",methods=['GET'])
def prediction():
    img_directory = request.args.get("img_directory")
    img_name = request.args.get("img_name")
    pred_threshold = float(request.args.get("pred_threshold"))
    print(img_directory)
    print(img_name)

    img_path = os.path.join(img_directory,img_name)
    image_array = buildImageArray(img_path)
    return_results = predictArrayImages(image_array, pred_threshold,img_path)
    return jsonify(return_results)


#url_for('prediction', img_directory='D:\smoke_test', img_name='1528057977_+00660.jpg',pred_threshold=".50")


if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')

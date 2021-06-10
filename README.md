# Wildfire-Detection
 The [ALERT Wildfire consortium](http://www.alertwildfire.org/  "ALERT Wildfire website") has a network of over 800 pan-tilt-zoom (PTZ) cameras that are monitored by firefighters and the general public to detect wildfire ignition, verify & locate reported fires, and monitor wildfire behavior.<br> This project explores the possibility of automating the early detection of wildfires using these cameras. To do so, it creates a prototype that uses a convolutional neural network (CNN) to monitor PTZ fire camera images from ALERT Wildfire in real-time and automatically detect the small smoke plumes that typically signal an ignition event. <br><br>
**Prototype:**
The prototype uses: 
- Keras-Tensorflow, Python, Pillow and OpenCV to create and train the convolutional neural net,
- Javascript, Bootstrap, D3.js, HTML and CSS to create a web application that displays camera images and the results of model evaluation
- Flask, Python, Pillow & OpenCV to create a server-side application that scrapes or reads full-size camera images, splits them up into 299x299 subimages, and passes these subimages to the model for classification. It serves up requests for image classification via a Flask API.<br>

It also evaluates some historical fire sequences for demonstration purposes. The prototype can be accessed on the Wildfire Detection tab [here](http://metavision.tech/ "California Wildfire Dashboard").

**Findings**: Based on initial results with this prototype, we conclude that it is definitely a direction worth further exploration!  The current version of the neural net classifies subimages with a 94.66% true accuracy rate. It produces false negatives (incorrectly classifying early signs of wildfire) only 0.37% of the time, while it produces false postives (incorrectly classifying images that contain no clear sign of an ignition event) 4.97% of the time. The false positives usually include clouds, smoke or haze. We have created a new dataset of 35,000 images to improve the false positive rate, and plan to re-train the neural net with this dataset and re-evaluate our results in the near future. 

## Data Collection & Augmentation
We leveraged a couple of wildfire image datasets to jumpstart this project:
- [Open Climate Tech](https://openclimatetech.org/ "Open Climate Tech website") made a dataset of 1,800 smoke images w/bounding box annotations, and a dataset of 40,000 non-smoke images available on their [Github repository](https://github.com/open-climate-tech/firecam/tree/master/datasets/2019a/ "Open Climate Tech Github repository").
- The [High Performance and Wireless Research & Education Network (HPWREN)](http://hpwren.ucsd.edu/ "HPWREN website") made over 500 video sequences of fires from the PTZ cameras available along with all of the associated full-size image files on their [Fire Ignition Images Library](http://hpwren.ucsd.edu/HPWREN-FIgLib/ "HPWREN image archive").

We collected and annotated 500 early ignition images from the HPWREN archive, and along with the 1,800 from OCT created a boosted dataset of ~16,000 early ignition images by shifting and flipping the extracted 299x299 subimages. We then combined with an equal number of non-ignition event images to create a balanced dataset of ~32,000.


## The Convolutional Neural Net
The CNN is built with Keras-Tensorflow and it uses Transfer Learning - leveraging Xception as a trained base for feature extraction. The classification head consists of a Flatten layer, a Dense layer (with 128 neurons) and an output layer that uses the Sigmoid function to output the probability percentage of the image containing an early ignition event.
 The various versions of the convnet were trained on Colab Pro using GPUs and later TPUs, and the latest version is deployed on our Cloud server.
![Wildfire Detection Neural Net Architecture](https://github.com/MThorpester/Wildfire-Detection/blob/main/TrainTestCNN/Images/Streamline1-Architecture.jpg)

## Wildfire Detection API
The wildfire detection API is running on a Debian Linux server in the Google Cloud.  The domain http://metavision.tech points to our cloud server, which our HTML is being serviced by Apache and API by Python Flask.  The following are the endpoints at http://metavision.tech:5000/api/:
- **/** 
    - API documentation
 - **/predict**
    - Endpoint initiates the necessary processes to feed image into our prediction model.  Process includes the splitting of images innto 299X299 pixel segments and feeding each segmented image to model for prediction.  Based on the precition threshold passed into URL a smoke indicator value is set,  the images is annotated with bounding boxes and probability returned from model. The location of the generated/annotated image and bounding boxes are also sent with JSON.  The following is a sample JSON set genearated by endpoint.

```[
    {
        img_url: "http://metavision.tech/images/bound_images/image.jpg",
        smoke_detected: "Y",
        bounds: [
                    {
                    x: 888,
                    y: 592,
                    width: 299,
                    height: 299,
                    prediction: "0.8909"
                    },
                    {
                    x: 1184,
                    y: 592,
                    width: 299,
                    height: 299,
                    prediction: "0.5892"
                    }
        ]
    }
]
```
- **/scrape**
    - Calling the endpoint triggers a scrape of Deerhorn camera and dropping the scraped image onto server. A JSON with the location of scraped  image is returned.  Selenium is used to scrape image, but the endpoint does not work in our cloud environment.  We were not able to open a browser on a non graphical system. We tried creating a virtual display, but still could not startup the browser.  The following is the JSON returned by endpoint:
```[
    {
        img_url: "http://metavision.tech/images/bound_images/image.jpg"
    }
   }
   ```
## Project Artifacts
Key project files are organized as follows:
- All artifacts related to training and testing the neural net are in the Wildfire-Detection/TrainTestCNN directory:
    - Jupyter notebooks used to train and test all versions of the neural net
        - The notebooks used to train and test the currently deployed production model are: Train_Save_Streamline1.ipynb, Test_Saved_Streamline1.ipynb
    - Jupyter notebooks for generating the confusion matrix, extracting images, splitting images, sampling images and composing new image datasets
    - The actual saved models are too big to store in this repository. The saved version of the currently deployed model is here: https://drive.google.com/drive/folders/1e5x_d8IY2h36o5rIdzuhVkWM9cJ_1nJB?usp=sharing
    - The actual image datasets for training the CNN are too large to store in this repository (the latest contains 35,000 images). The latest dataset is here: https://drive.google.com/drive/folders/1mjEPKK596iGEUERgapdXmxCh-M3pK8UV?usp=sharingdataset 

- Web app
    - Our website html, CSS and javascript code is located in the **webassets** folder of our GitHub repository.
- API
    - Flask code is located in the **flask** folder of our GitHub repository.
- Image Augmentation 
    - Python code is in the **image_processing** folder of our GitHub repository.


## Getting Started**

To run this application visit the hosted version on the Wildfire Detection tab [here](http://metavision.tech/ "California Wildfire Dashboard").

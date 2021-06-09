# Wildfire-Detection
Most of the 650 traditional California fire watchtowers, manned by human wildfire lookouts, have been replaced over the last few decades by a combination of technology and remote monitoring. The [ALERT Wildfire consortium](http://www.alertwildfire.org/  "ALERT Wildfire website") has a network of over 800 pan-tilt-zoom (PTZ) cameras that are monitored by firefighters and the general public to detect wildfire ignition, verify & locate reported fires, and monitor wildfire behavior.<br> This project explores the possibility of automating the early detection of wildfires. To do so, it creates a prototype that uses a convolutional neural network (CNN) to monitor PTZ fire camera images from ALERT Wildfire in real-time and automatically detect the small smoke plumes that typically signal an ignition event. <br>The prototype uses: 
- Keras-Tensorflow, Python, Pillow and OpenCV and to create and train the convolutional neural net,
- Javascript, Bootstrap, HTML and CSS to create a web application that displays camera images and the results of model evaluation
- Flask, Python, Pillow & OpenCV to create an server-side application that scrapes full-size camera images, splits them up into 299x299 subimages, and passes these subimages to the model for classification. It serves up requests for image classification via a Flask API.

<br>  It also evaluates some historical fire sequences for demonstration purposes. The prototype can be accessed [here](http://35.193.188.227/ "California Wildfire Dashboard"). UPDATE LINK!   
 

**Findings**: Based on initial results with this prototype, we conclude that it is definitely a direction worth further exploration!  The current version of the neural net classifies subimages with a 94.66% true accuracy rate. It produces false negatives (incorrectly classifying early signs of wildfire) only 0.37% of the time, while it produces false postives (incorrectly classifying images that contain no clear sign of an ignition event) 4.97% of the time. We have created a new dataset of 35,000 images to improve the false positive rate, and plan to re-trainthe neural net with this dataset and re-evaluate our results in the future. 

## Data Collection & Augmentation
We leveraged a couple of wildfire image datasets to jumpstart this project:
- [Open Climate Tech](https://openclimatetech.org/ "Open Climate Tech website") made a dataset of 1,800 smoke images w/bounding box annotations, and a dataset of 40,000 non-smoke images available on their [Github repository](https://github.com/open-climate-tech/firecam/tree/master/datasets/2019a/ "Open Climate Tech Github repository").
- The [High Performance and Wireless Research & Education Network (HPWREN)](http://hpwren.ucsd.edu/ "HPWREN website") made over 500 video sequences of fires from the PTZ cameras available along with all of the associated full-size image files on their [Fire Ignition Images Library](http://hpwren.ucsd.edu/HPWREN-FIgLib/ "HPWREN image archive").

We collected and annotated 500 early ignition images from the HPWREN archive, and along with the 1,800 from OCT created a boosted dataset of ~16,000 early ignition images by shifting and flipping the extracted 299x299 subimages. We then combined with an equal number of non-ignition event images to create a balanced dataset of ~32,000.


## The Convolutional Neural Net
The CNN is built with Keras-Tensorflow and it uses Transfer Learning - leveraging Xception as a trained base for feature extraction. The classification head consists of a Flatten layer, a Dense layer (with 128 neurons) and an output layer that uses the Sigmoid function to output the probability percentage of the image containing an early ignition event.
![Wildfire Detection Neural Net Architecture](https://github.com/MThorpester/Wildfire-Detection/blob/main/Margaret/Streamline1-Architecture.jpg) 


## Wildfire Detection API
Several different requests are exposed by a Flask as API endpoints at http://metavision.tech/images/: UPDATE THIS!
- / 
    - API documentation
- /images
    - Returns the ?????
 - /evaluate
    - Returns the ??????
- /etc.
    - ???????? 
-
## Project Artifacts
Key project files are organized as follows:
- All artifacts related to training and testing the neural net are in the Wildfire-Detection/TrainTestCNN directory:
    - Jupyter notebooks used to train and test all versions of the neural net
    - Jupyter notebooks for extracting images, splitting images, sampling images and composing new image datasets
    - The actual saved models are too big to store in this repository. The saved version of the currently deployed model is here: https://drive.google.com/drive/folders/1e5x_d8IY2h36o5rIdzuhVkWM9cJ_1nJB?usp=sharing
    - The actual image datasets for training the CNN are too large to store in this repository (the latest contains 35,000 images). The latest dataset is here: https://drive.google.com/drive/folders/1mjEPKK596iGEUERgapdXmxCh-M3pK8UV?usp=sharingdataset 

- Web app
- API


## Getting Started

To run this application visit the hosted version [here](http://35.193.188.227/ "California Wildfire Dashboard").UPDATE THIS!

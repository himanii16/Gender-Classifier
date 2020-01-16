#Project : Gender Detection

#importing libraries
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils.data_utils import get_file
import numpy as np
import cv2
import os
import cvlib as cv

#taking user input for the image path
img_path = str (input("Enter name of the image:"))
image = cv2.imread(img_path)  #loading the image 

#condition if image is not found
if image is None:
    print("Could not read input image")
    exit()
    
#downloading dataset
dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
#model_path defines the path of the dataset 
model_path = get_file("gender_detection.model", dwnld_link,
                     cache_subdir="pre-trained", cache_dir=os.getcwd())
  
#loading dataset                   
model = load_model(model_path)

#function to detect faces 
face ,confidence= cv.detect_face(image) 
#confidence shows the percentage of accuracy of the detection of face 
#which depends on the quality and pixel of input

#loop to identify each face in the image 
for index, f in enumerate(face):
	#each face has 4 vertex which defines the face and are defined as follows:
    (startX, startY) = f[0], f[1] 
    (endX, endY) = f[2], f[3]
    
    #crop the detected face region
    face_crop = np.copy(image[startY:endY,startX:endX])

    #preprocessing for gender detection model
    #resize the cropped image 
    face_crop = cv2.resize(face_crop, (96,96)) 
     
    #typecasting the type of image to float
    face_crop = face_crop.astype("float") / 255.0
    
    #converting cropped image to array using keras library
    face_crop = img_to_array(face_crop)
    
    #Expand the shape of an array.
	#Insert a new axis that will appear at the axis position in the expanded array shape.
    face_crop = np.expand_dims(face_crop, axis=0)

    #applying gender detection on face using the above defined model
    conf = model.predict(face_crop)[0]
    
    #Returns the indices of the maximum values along an axis and return 0 if image is of a male and 1 if the image is of a female 
    index = np.argmax(conf) 
    
    if index==0:     
        print (0)    #male is detected
    
    else :
        print (1)    #female is detected


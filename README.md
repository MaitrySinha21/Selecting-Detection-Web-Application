# Selecting-Detection-Web-Application
### Multi-detection web application build using SSD Mobile-net to select & detect any video & objects

### An Automatic Selective object detection model wth 80 types of different objects trained on SSD Mobilenet and implemented on Flask api using opencv & HTML.We can choose any video from our system giving the path and select any object/objects we want to detect and it will able to detect those object/objects


![final](https://user-images.githubusercontent.com/52413661/151869188-453dcb97-2dca-4979-84d4-c40795e396d5.gif)




## Architecture Used:

# Flask , SSD Mobilenet, OpenCV, HTML.

## All the requirements are mentioned in the requirements.txt file.


### How to run:--
### 1) After clonning first open in pycharm & create an conda enviornment with python 3.6 or python 3.7

### 2) Install requirements.txt
      pip install -r requirements.txt

### 3) Now run  app.py

### 4) You can use any video as "street.mp4" or  any video of your choice or can directly use live camera.

### 5) To run the dockerfile pass the following command --
    docker run -it --name vehicle_ssd -p 8888:5000 imageid


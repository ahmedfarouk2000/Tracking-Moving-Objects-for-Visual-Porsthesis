
import cv2 as cv
import numpy as np
import os 

allVideos=os.listdir('C:/Users/AHMED FAROUK/Documents/python computer vision/video')
capture =cv.VideoCapture('video/cat.mp4') 

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
frame_size = (frame_width,frame_height)
fps = 20
output = cv.VideoWriter('output_video_from_file.avi', cv.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)


# camera=cv.VideoCapture(0)
def ReadFromCamera():
    while True :
        _,frame=capture.read()
        # cv.imshow("Display",FirstImage)
        output.write(frame)
        cv.imshow('asdasd',frame)
         
        if cv.waitKey(1)=='a':
            cv.destroyAllWindows() 
            break




ReadFromCamera()
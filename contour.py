from copy import copy
from http.client import CannotSendHeader
from msilib.schema import Media
from pickle import TRUE
from sqlite3 import apilevel
from statistics import median
import cv2 as cv 

import sys
import random


import glob
import numpy as np 
from matplotlib import pyplot as plt
import math 
import sys          
import argparse                                                                                
                                                                                                                                                                                                                                                               
                                                                                                    
                                                                  
                                                                                                    
# Create the Qt Application                                                                         
# app = QApplication(sys.argv)                                                                        
# # Create a button, connect it and show it                                                           
# button = QPushButton("Click me")                                                                    
# button.clicked.connect(say_hello)                                                                    
# button.show()                                                                                       
# Run the main Qt loop                                                                              
# app.exec_()

def nothing(args):
    pass

def nextimage(args):
    pass

# def back(*args):
#     pass

# Prints PySide6 version


cv.namedWindow("Control Panel" , cv.WINDOW_NORMAL)
cv.resizeWindow("Control Panel", 1920, 150)
cv.moveWindow("Control Panel", 0,0)

cv.namedWindow("Display2")



# import cv
  
# # Path

  
# # Reading an image in grayscale mode

  
# # Naming a window
# cv.namedWindow("Resize", cv.WINDOW_NORMAL)
  
  
# # Using resizeWindow()
# cv.resizeWindow("Resize", 700, 200)
  
# Displaying the image



cv.createTrackbar('Upper','Control Panel',175,255,nothing)
cv.createTrackbar('lower','Control Panel',125,255,nothing)

cv.createTrackbar('Blur','Control Panel',11,101,nothing)

# cv.createTrackbar('Width','Display',500,1000,nothing)
# cv.createTrackbar('Height','Display',400,800,nothing)
cv.createTrackbar('Thickness','Control Panel',2,20,nothing)

cv.createTrackbar('Thickness fake','Control Panel',0,1000,nothing)

cv.createTrackbar('Next','Control Panel',0,1,nextimage)
cv.createTrackbar('Previous','Control Panel',0,1,nextimage)
# cv.createButton('Next','Display',nothing,nothing,1)
# cv.createButton ('next',nothing)
# cvCreateButton("button6",callbackButton2,NULL,CV\_PUSH\_BUTTON,1);

# cv.createButton('hello',nothing)

# cv.createButton("Back",back,None,cv.QT_PUSH_BUTTON,1) #need at least cv3

# camera=cv.VideoCapture(0)
# img =cv.imread('photo/bird2.jpg')
# # img=cv.cvtColor(img,cv.COLOR_BGR2RGB) #must first conevrt it from bgr to rgb 
# gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# MedianBlur = cv.medianBlur(gray, ksize=21)
# CannyFliter =cv.Canny(MedianBlur , 125,175) #to remove noises from an image 

# ret,thresh = cv.threshold(CannyFliter, 0, 100, cv.THRESH_BINARY) #same as the canny doing to apply the threshhold
# visualize the binary image


# cv.imshow('Binary image', CannyFliter)


#this one applyed on image original image but cant use blur 
path = glob.glob("photo/*")
cv_img = []
for img in path:
    n = cv.imread(img)
    cv_img.append(n)


def ReadFromImage():
    totalImages=len(cv_img)
    counter=0 
    switchNext=0 
    switchPrev=0 
    
    while True:
      
        nextImg =cv.getTrackbarPos('Next','Control Panel')
        prevImg =cv.getTrackbarPos('Previous','Control Panel')
       
        if   nextImg==1 and switchNext==1 :
            counter+=1
            img =cv_img[counter%totalImages]
            switchNext=0 
        elif nextImg==0  and switchNext==0 :
            counter+=1
            img =cv_img[counter%totalImages]
            switchNext=1
          
        elif prevImg==0 and  switchPrev==0 : 
             counter-=1
             img =cv_img[counter%totalImages]
             switchPrev=1
        elif prevImg==1 and  switchPrev==1 :
             counter-=1
             img =cv_img[counter%totalImages]
             switchPrev=0

        lower=cv.getTrackbarPos('lower','Control Panel')
        upper=cv.getTrackbarPos('Upper','Control Panel')
        # width=cv.getTrackbarPos('Width','Display')
        # height=cv.getTrackbarPos('Height','Display')
        # down_width =cv.getTrackbarPos('Width','Display')
        # down_height =cv.getTrackbarPos('Width','Display')
        thickness =cv.getTrackbarPos('Thickness','Control Panel')

        thicknessfake =cv.getTrackbarPos('Thickness fake','Control Panel')
        blur=cv.getTrackbarPos('Blur','Control Panel')
        if blur%2==0:
            blur+=1
        # down_points = (down_width, down_height)

      
        # CannyFliter = cv.resize(CannyFliter, down_points, interpolation= cv.INTER_LINEAR)
        # ScaleSize=cv.getTrackbarPos('Scale','Display')
        # CannyFliter=rescaleFrame(CannyFliter,scale=size)
        # gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # MedianBlur = cv.medianBlur(gray, ksize=blur)
        # CannyFliter =cv.Canny(MedianBlur , lower,upper) #to remove noises from an image 

        # img_dilation = cv.dilate(CannyFliter, (5,5), iterations=thicknessfake)
        # img_dilation = cv.dilate(CannyFliter, (5,5), iterations=thickness)
        img_copy = img.copy()
        #three diff modes are for finding contour are RETR_TREE ,  RETR_EXTERNAL ,RETR_LIST ,RETR_CCOMP
        #External will discard the children and that is what we want 
        #List wont make any relations at all
        # contours, _ = cv.findContours(image=CannyFliter, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        # cv.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=thickness, lineType=cv.LINE_AA)
       
        
      
        # PhospheneImage=cv.circle(blank.copy(),(img.shape[1]//2,img.shape[0]//2),100,(255,255,255),-1) #-1 means fill the circle
        fix=cv.cvtColor(img_copy,cv.COLOR_BGR2GRAY)
        _, result = cv.threshold(fix, 150, 255, cv.THRESH_BINARY_INV) # this means will replace each white color in phosphene to the gray scale
        mask=cv.bitwise_and(img,img,mask=result)
        print(img.shape)
        print(result.shape)
            #    mask=cv.bitwise_and(img,img,mask=img_dilation)

        blank=np.zeros(img.shape[:2],dtype=np.uint8)

       

    

        # biggest_contour = max(contours, key = cv.contourArea)
        # x, y, w, h = cv.boundingRect(biggest_contour)
        # rect=cv.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # # print(len(rect))
        # # print(len(rect[0]))
        # # print(contours[0])
        # cv.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=2)
        # cv.imshow('Display2',img_copy)
        # cv.waitKey(0)
        # img2 = cv.imread('photo/real_cat.jpg')
        img_copy2=img.copy()
        gray = cv.cvtColor(img_copy2,cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray,thicknessfake,0.0000001,upper)
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            cv.circle(img_copy2,(x,y),3,255,-1)

        cv.imshow('Display2',img_copy2)
        # cv.waitKey(0)
        # for i in range(len(contours)):
            
        #     cv.drawContours(blank,contours,i,(0,255,0),3) #will display it from the index cotour till the length of contour
          
        #     cv.imshow('Display2',blank)
        #     cv.waitKey(0)
        # graynew=cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
        # cv.imshow('Display2', img_copy)
        
        # print(img.shape)

        # _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
        # mask2=cv.bitwise_and(graynew,graynew,mask=thresh)

        # cv.imshow('asdas',mask2)

        # cv.imshow('Display2',img_copy)
        

        if cv.waitKey(1)==27:
            cv.destroyAllWindows() 
            break         


#this one applyed on blured imaged after the gray scale so i can adjust blur easily 
def ReadFromImageWithBlur():
    cv.createTrackbar('Blur','Display',21,201,nothing)
    while True:
        x=cv.getTrackbarPos('lower','Display')
        y=cv.getTrackbarPos('Upper','Display')
        # width=cv.getTrackbarPos('Width','Display')
        # height=cv.getTrackbarPos('Height','Display')
        # down_width =cv.getTrackbarPos('Width','Display')
        # down_height =cv.getTrackbarPos('Width','Display')
        thickness =cv.getTrackbarPos('Thickness','Display')
        # down_points = (down_width, down_height)
        Blurksize =cv.getTrackbarPos('Blur','Display')
        if Blurksize%2==0:
            Blurksize+=1
      
        # CannyFliter = cv.resize(CannyFliter, down_points, interpolation= cv.INTER_LINEAR)
        # ScaleSize=cv.getTrackbarPos('Scale','Display')
        # CannyFliter=rescaleFrame(CannyFliter,scale=size)
        gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        MedianBlur = cv.medianBlur(gray, ksize=Blurksize)
        CannyFliter =cv.Canny(MedianBlur , x,y) #to remove noises from an image 
        # img_dilation = cv.dilate(CannyFliter, (5,5), iterations=thickness)
        img_copy = MedianBlur.copy()
        contours, _ = cv.findContours(image=CannyFliter, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        cv.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=thickness, lineType=cv.LINE_AA)

        cv.imshow('Display',img_copy)
        if cv.waitKey(1)==27:
            cv.destroyAllWindows() 
            break    

   




# def ReadFromCamera():
#     cv.createTrackbar('Blur','Display',21,201,nothing)
#     while True:
#         _,img=camera.read()
           
#         x=cv.getTrackbarPos('lower','Display')
#         y=cv.getTrackbarPos('Upper','Display')
#         # width=cv.getTrackbarPos('Width','Display')
#         # height=cv.getTrackbarPos('Height','Display')
#         # down_width =cv.getTrackbarPos('Width','Display')
#         # down_height =cv.getTrackbarPos('Width','Display')
#         thickness =cv.getTrackbarPos('Thickness','Display')
#         # down_points = (down_width, down_height)
#         Blurksize =cv.getTrackbarPos('Blur','Display')
#         if Blurksize%2==0:
#             Blurksize+=1
      
#         # CannyFliter = cv.resize(CannyFliter, down_points, interpolation= cv.INTER_LINEAR)
#         # ScaleSize=cv.getTrackbarPos('Scale','Display')
#         # CannyFliter=rescaleFrame(CannyFliter,scale=size)
#         # gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#         MedianBlur = cv.medianBlur(img, ksize=Blurksize)
#         CannyFliter =cv.Canny(MedianBlur , x,y) #to remove noises from an image 
#         img_dilation = cv.dilate(CannyFliter, (5,5), iterations=thickness)
#         img_copy = MedianBlur.copy()
#         contours, _ = cv.findContours(image=img_dilation, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
#         cv.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv.LINE_AA)

#         cv.imshow('Display',img_copy)
#         if cv.waitKey(1)==27:
#             cv.destroyAllWindows() 
#             break

# ReadFromCamera()
ReadFromImageWithBlur()
# ReadFromImage()
# ReadFromImage()
# ReadFromCamera()

images=[] 
def addImage(path):
    images.append(path)


addImage('photo/real_Car.jpg')
addImage('photo/bird2.jpg')
addImage('photo/butter.png')
addImage('photo/rappit.jpg')
# addImage('photo/tiger.jpg')
# addImage('photo/tiger.jpg')
# addImage('photo/tiger.jpg')
# addImage('photo/tiger.jpg')
# addImage('photo/tiger.jpg')
# addImage('photo/tiger.jpg')

# def nextImage(args):

# def drawRectangle(action, x, y, flags, *userdata):
#   # Referencing global variables 
#   global top_left_corner, bottom_right_corner
#   # Mark the top left corner when left mouse button is pressed
#   if action == cv.EVENT_LBUTTONDOWN:
#     top_left_corner = [(x,y)]
#     # When left mouse button is released, mark bottom right corner
#   elif action == cv.EVENT_LBUTTONUP:
#     bottom_right_corner = [(x,y)]    
#     # Draw the rectangle
#     cv.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0,255,0),2, 8)
#     cv.imshow("Window",image)


# image = cv.imread("photo/dog2.jpg")
# # Make a temporary image, will be useful to clear the drawing
# temp = image.copy()
# # Create a named window
# cv.namedWindow("Window")
# # highgui function called when mouse events occur
# cv.setMouseCallback("Window", drawRectangle)

# k=0
# # Close the window when key q is pressed
# while k!=113:
#   # Display the image
#   cv.imshow("Window", image)
#   k = cv.waitKey(0)
#   # If c is pressed, clear the window, using the dummy image
#   if (k == 99):
#     image= temp.copy()
#     cv.imshow("Window", image)

# cv.destroyAllWindows()





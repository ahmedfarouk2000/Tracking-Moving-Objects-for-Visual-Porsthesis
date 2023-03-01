from http.client import CannotSendHeader
from msilib.schema import Media
from sqlite3 import apilevel
import cv2 as cv 
import numpy as np 
from matplotlib import pyplot as plt
import math 

def nothing(x):
    pass 
cv.namedWindow("Display")
cv.createTrackbar('lower','Display',0,255,nothing)
cv.createTrackbar('Upper','Display',0,255,nothing)
# cv.createTrackbar('Width','Display',500,1000,nothing)
# cv.createTrackbar('Height','Display',400,800,nothing)
cv.createTrackbar('Thickness','Display',0,20,nothing)
cv.createTrackbar('Blur','Display',21,201,nothing)

camera=cv.VideoCapture(0)

def rescaleFrame(frame,scale=0.01):
    #used for images and videos and live videos
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)

    dimentions=(width,height)

    return cv.resize(frame,dimentions,interpolation=cv.INTER_AREA)


img =cv.imread('photo/lion.jpg')
img=cv.cvtColor(img,cv.COLOR_BGR2RGB) #must first conevrt it from bgr to rgb 
gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
GausianBlur=cv.GaussianBlur(gray,(21,21),cv.BORDER_DEFAULT)
NoramlBlur = cv.blur(gray, (21,21)) # Using the blur function to blur an image where ksize is the kernel size

MedianBlur = cv.medianBlur(gray, ksize=21)

CannyFliter =cv.Canny(MedianBlur , 125,175) #to remove noises from an image 

# sobelxy = cv.Sobel(src=MedianBlur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
resized_img=rescaleFrame(CannyFliter,scale=0.18)
# cv.imshow("SobelFilter", sobelxy)
# cv.imshow("aaa",resized_img)
# img_erosion = cv.erode(CannyFliter, (5,5), iterations=1)
img_dilation = cv.dilate(resized_img, (5,5), iterations=50)
# cv.imshow("aaa",img_dilation)
# cv.imshow('yoo',img_dilation)
# cv.imshow('yoo',img_erosion)
# down_width=900
# down_height=900

# down_points = (down_width, down_height)
# CannyFliter = cv.resize(CannyFliter, down_points, interpolation= cv.INTER_LINEAR)


# while True:
#         _,img=camera.read()
#         gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#         MedianBlur = cv.medianBlur(gray, ksize=21)
#         x=cv.getTrackbarPos('lower','Display')
#         y=cv.getTrackbarPos('Upper','Display')
#         width=cv.getTrackbarPos('Width','Display')
#         height=cv.getTrackbarPos('Height','Display')
#         down_width =cv.getTrackbarPos('Width','Display')
#         down_height =cv.getTrackbarPos('Width','Display')
        
#         down_points = (down_width, down_height)
#         CannyFliter = cv.resize(CannyFliter, down_points, interpolation= cv.INTER_LINEAR)
#         # ScaleSize=cv.getTrackbarPos('Scale','Display')
#         # CannyFliter=rescaleFrame(CannyFliter,scale=size)
#         CannyFliter =cv.Canny(MedianBlur , x,y) #to remove noises from an image 
    
#         cv.imshow('Display',CannyFliter)
#         if cv.waitKey(1)==27:
#             cv.destroyAllWindows() 
#             break 


    
def ReadFromCamera():
    while True:
        _,img=camera.read()
        
        x=cv.getTrackbarPos('lower','Display')
        y=cv.getTrackbarPos('Upper','Display')
        # width=cv.getTrackbarPos('Width','Display')
        # height=cv.getTrackbarPos('Height','Display')
        # down_width =cv.getTrackbarPos('Width','Display')
        # down_height =cv.getTrackbarPos('Width','Display')
        # down_points = (down_width, down_height)
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
        img_dilation = cv.dilate(CannyFliter, (5,5), iterations=thickness)

        cv.imshow('Display',img_dilation)
        if cv.waitKey(1)==27:
            cv.destroyAllWindows() 
            break 



def ReadFromImage():
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
        MedianBlur = cv.medianBlur(gray, ksize=21)
        CannyFliter =cv.Canny(MedianBlur , x,y) #to remove noises from an image 
        # cv.imshow('canny',CannyFliter)
        # cv.waitKey(0)
        # img_dilation = cv.dilate(CannyFliter, (5,5), iterations=thickness)

        cv.imshow('Display',CannyFliter)
        if cv.waitKey(1)==27:
            cv.destroyAllWindows() 
            break         



# ReadFromCamera()
ReadFromImage()
    
# cv.imshow("new one", CannyFliter)
# cv.waitKey(0)


# cv.waitKey(0)

# img_blur = cv.GaussianBlur(img,(9,9), sigmaX=10, sigmaY=10)


# blur3 = cv.blur(img, (7,7)) # Using the blur function to blur an image where ksize is the kernel size



# bilateral_filter = cv.bilateralFilter(img, d=51, sigmaColor=20, sigmaSpace=20)
# cv.imshow('aaa',bilateral_filter)
# cv.waitKey(0)


#this used to identify the edges of any image  ( to get less edges jsut pass the blur image)
# canny =cv.Canny(blur , 125,175) 
# canny2 =cv.Canny(blur2 , 125,175) 



# canny4 =cv.Canny(bilateral_filter , 125,175) 

#dilated image (expanding the edges resulted from canny )
# dilated=cv.dilate(img,(1,1),iterations=50) 








# lap =cv.Laplacian(gray,cv.CV_64F,Ksize=3)
# lap =np.uint8(np.absolute(lap))






titles=[]
images=[]
def appendBoth(name,image):
    titles.append(name) 
    images.append(image)

appendBoth("origianl",img)  
appendBoth("gray",gray)
appendBoth("noraml blur by 21",NoramlBlur)    
appendBoth("Gaus blur by 21",GausianBlur)    
appendBoth("Median blur by 21",MedianBlur)     
appendBoth("Canny after Median",CannyFliter)    
# appendBoth("Dilation after Median 5 iteratons",resized_img) #used to make the line thicker     
# appendBoth("Canny after Median",img_erosion)    
    
# appendBoth("sobelxy after Median",sobelxy)    
# appendBoth("canny after median blur by 21",canny3)    


# appendBoth("laplacian",lap)    



for i in range(len(images)):
    plt.subplot(math.ceil(len(images)/5),5,i+1) #will always be 3 columns 
    plt.imshow(images[i],'gray') #by default gray for some how 
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
# plt.show()    






  

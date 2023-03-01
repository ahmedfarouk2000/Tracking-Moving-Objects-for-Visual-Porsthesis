import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt
import math 
# img =cv.imread('photo/cat.jpg')
# cv.imshow('CAT',img)
# cv.waitKey(0) #used to wait until pressing any button on screen 
# print("wow") 



# used to display any video desired 
capture =cv.VideoCapture('video/bat1.mp4') 
while True:
   isTrue,frame =capture.read()
   bbox=cv.selectROI("Display",frame,False)  # keep pressing on space so that the vidoe keep going 
   cv.imshow('Video',frame) 

   if cv.waitKey(20) & 0xFF==ord('s'):
      break 

# capture.release()
# cv.destroyAllWindows() 
# titles=["wow"]
# videos=[capture]
# for i in range(len(videos)):
#     plt.subplot(math.ceil(len(videos)/5),5,i+1) #will always be 3 columns 
#     plt.imshow(videos[i],'gray') #by default gray for some how 
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])

# plt.show()    
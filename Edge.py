
from ast import Return
from unittest import result
import cv2 as cv 
import numpy as np
import glob



def nothing(args):
    pass

def nextimage(args):
    pass




path = glob.glob("photo/*")
cv_img = []
for img in path:
    n = cv.imread(img)
    cv_img.append(n)

def ReadFromImageBlackAndWhite(): #this one is uselss now 
    cv.namedWindow("Control" , cv.WINDOW_NORMAL)
    cv.resizeWindow("Control", 1920, 150)
    cv.moveWindow("Control", 0,0)
    cv.namedWindow("Display2")
    cv.createTrackbar('nuCircles','Control',175,1000,nothing)
    cv.createTrackbar('DistanceBet','Control',125,500,nothing)
    cv.createTrackbar('accuracy','Control',1,10,nothing)
    cv.createTrackbar('radius','Control',4,30,nothing)
    cv.createTrackbar('Next','Control',0,1,nextimage)
    cv.createTrackbar('Previous','Control',0,1,nextimage)

    totalImages=len(cv_img)
    counter=0 
    switchNext=0 
    switchPrev=0 
    
    while True:
      
        nextImg =cv.getTrackbarPos('Next','Control')
        prevImg =cv.getTrackbarPos('Previous','Control')
       
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

       
        nuCircles =cv.getTrackbarPos('nuCircles','Control')
        DistanceBet =cv.getTrackbarPos('DistanceBet','Control')


        accuracy =cv.getTrackbarPos('accuracy','Control')/100
        raduis =cv.getTrackbarPos('radius','Control')
        if accuracy==0 :
            accuracy=1/100
            cv.setTrackbarPos('accuracy','Control',1) #only even num are allowed bec of output numpy array
        if raduis==0:
            raduis+=1
            cv.setTrackbarPos('radius','Control',raduis) #only even num are allowed bec of output numpy array


        img_copy2=img.copy()
        # blank=np.zeros(img.shape[:2],dtype=np.uint8)
        blank =np.zeros(img_copy2.shape[:2],dtype=np.uint8)
        gray = cv.cvtColor(img_copy2,cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray,nuCircles,accuracy,DistanceBet) # shi-tomasi algo corner detection
        #will return x,y postion of the phos places and i can choose any desired radius
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            circle = np.zeros((img.shape[0],img.shape[1]), np.uint8) #Creamos mascara (matriz de ceros) del tamano de la imagen original
            cv.circle(circle,(x,y),raduis,255,-1)
            avg = cv.mean(img, mask=circle)[::-1]
            avg_value=(avg[1]+avg[2]+avg[3])//3
            #previous 4 steps used only to find the avg color to used inside the phos
            cv.circle(blank,(x,y),raduis,avg_value,-1)



        cv.imshow('Display2',blank)
      
        

        if cv.waitKey(1)==27:
            cv.destroyAllWindows() 
            break 








def ReadFromImageColored():
    cv.namedWindow("Control" , cv.WINDOW_NORMAL)
    cv.resizeWindow("Control", 1920, 150)
    cv.moveWindow("Control", 0,0)
    cv.namedWindow("Display2")
    cv.createTrackbar('nuCircles','Control',175,1000,nothing)
    cv.createTrackbar('DistanceBet','Control',125,500,nothing)
    cv.createTrackbar('accuracy','Control',1,10,nothing)
    cv.createTrackbar('radius','Control',4,30,nothing)
    cv.createTrackbar('Next','Control',0,1,nextimage)
    cv.createTrackbar('Previous','Control',0,1,nextimage)
    totalImages=len(cv_img)
    counter=0 
    switchNext=0 
    switchPrev=0 
    
    while True:
      
        nextImg =cv.getTrackbarPos('Next','Control')
        prevImg =cv.getTrackbarPos('Previous','Control')
       
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

       

      
       
        nuCircles =cv.getTrackbarPos('nuCircles','Control')
        DistanceBet =cv.getTrackbarPos('DistanceBet','Control')
        accuracy =cv.getTrackbarPos('accuracy','Control')/100
        raduis =cv.getTrackbarPos('radius','Control')
        if accuracy==0 :
            accuracy=1/100
        if raduis==0:
            raduis+=1

        img_copy2=img.copy()
        # blank=np.zeros(img.shape[:2],dtype=np.uint8)
        blank =np.zeros(img_copy2.shape[:2],dtype=np.uint8)
        gray = cv.cvtColor(img_copy2,cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray,nuCircles,accuracy,DistanceBet)
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            # circle = np.zeros((img.shape[0],img.shape[1]), np.uint8) #Creamos mascara (matriz de ceros) del tamano de la imagen original
            cv.circle(img_copy2,(x,y),raduis,255,-1)
            # avg = cv.mean(img, mask=circle)[::-1]
            # avg_value=(avg[1]+avg[2]+avg[3])//3
            # cv.circle(blank,(x,y),raduis,avg_value,-1)

        cv.imshow('Display2',img_copy2)
       
   

        if cv.waitKey(1)==27:
            cv.destroyAllWindows() 
            break            



# ReadFromImageBlackAndWhite()
# ReadFromImageColored()
from ast import Add
from audioop import add
from cgi import print_environ
from concurrent.futures import thread
from copy import copy
from distutils.log import info
from enum import unique
from http.client import CannotSendHeader
from msilib.schema import Media
from operator import invert
from pickle import TRUE
from pickletools import read_unicodestring1
from sqlite3 import apilevel
from traceback import print_tb
from turtle import circle
from Edge import ReadFromImageBlackAndWhite, ReadFromImageColored
import cv2 as cv 

import sys
import random


import glob
import numpy as np 
from matplotlib import pyplot as plt
import math 
import sys                                                                                          
from PIL import Image
import plotly.express as px
# cv.namedWindow("Display2")
def nothing(args):
    pass
def nextimage(args):
    pass


cv.namedWindow("Control Panel" , cv.WINDOW_AUTOSIZE )
cv.resizeWindow("Control Panel", 1000, 300)
cv.moveWindow("Control Panel", 0,0)


cv.namedWindow("Display2")
cv.moveWindow("Display2", 0,350)

# cv.namedWindow("All Info")
# cv.moveWindow("All Info",200,500)

cv.createTrackbar('In Width','Control Panel',32,255,nothing)
cv.createTrackbar('In Height','Control Panel',32,255,nothing)
cv.createTrackbar('Pho Size','Control Panel',12,16,nothing)
# cv.createTrackbar('Spacing','Control Panel',0,100,nothing) # in pixel
# cv.createTrackbar('Space Between Phosphens in Pixel','Control Panel',3,5,nothing)
cv.createTrackbar('Inverse','Control Panel',1,1,nothing)
cv.createTrackbar('Gray Bits','Control Panel',1,8,nextimage)
cv.createTrackbar('Next','Control Panel',0,1,nextimage)
cv.createTrackbar('Previous','Control Panel',0,1,nextimage)


# cv.namedWindow("Control" , cv.WINDOW_NORMAL)
# cv.resizeWindow("Control", 1920, 150)
# cv.moveWindow("Control", 0,0)
cv.createTrackbar('nuPhos','Control Panel',175,1000,nothing)
cv.createTrackbar('Distance','Control Panel',31,500,nothing)
cv.createTrackbar('accuracy','Control Panel',1,10,nothing)
cv.createTrackbar('radius','Control Panel',4,30,nothing)





path = glob.glob("photo/*")
cv_img = []
for img in path:
    n = cv.imread(img)
    cv_img.append(n)





def PhosLocations(img_input):
    while True:
           
        nuPhos =cv.getTrackbarPos('nuPhos','Control Panel')
        Distance =cv.getTrackbarPos('Distance','Control Panel')
        accuracy =cv.getTrackbarPos('accuracy','Control Panel')/100
        raduis =cv.getTrackbarPos('radius','Control Panel')
        if accuracy==0:
            accuracy=0.01
        # nuPhos =175
        # Distance =17
        # accuracy =0.01
        # raduis =4
        
        img_copy2=img_input.copy()
        List=[]

        blank =np.zeros(img_copy2.shape[:2],dtype=np.uint8)
        gray = cv.cvtColor(img_copy2,cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray,nuPhos,accuracy,Distance)
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            circle = np.zeros((img_copy2.shape[0],img_copy2.shape[1]), np.uint8) #Creamos mascara (matriz de ceros) del tamano de la imagen original
            cv.circle(circle,(x,y),raduis,255,-1)
            avg = cv.mean(img_copy2, mask=circle)[::-1]
            avg_value=(avg[1]+avg[2]+avg[3])//3
            # cv.circle(blank,(x,y),raduis,avg_value,-1)
            Element=[x,y,avg_value] 
            List.append(Element)
        # cv.imshow('dis3',circle)
        return List

def CreateFinalPhos(): #only one bit reprresentation black and white
    #this list will contain the info about Phosphene x , y and color 3 things
   
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
            

      
        PhospheneSize = cv.getTrackbarPos('Pho Size','Control Panel')
        if PhospheneSize%2==1: #only even number are allowed
            PhospheneSize+=1
            cv.setTrackbarPos('Pho Size','Control Panel',PhospheneSize) #only even num are allowed bec of output numpy array
        if PhospheneSize==0:
            PhospheneSize=2
            cv.setTrackbarPos('Pho Size','Control Panel',PhospheneSize) #only even num are allowed bec of output numpy array
        # widthPhos =PhospheneSize
        # heightPhos = PhospheneSize

        # SpaceBetween = cv.getTrackbarPos('Spacing','Control Panel')
        # if widthPhos==0:
        #     widthPhos+=1
        # if heightPhos==0:
        #     heightPhos+=1
      
        # dimPhos = (widthPhos, heightPhos)
        # PhospheneImage= cv.imread('C:/Users/AHMED FAROUK/Documents/python computer vision/Phosphenes/test2.png')

        # blank=np.zeros((400,400,3),dtype=np.uint8)
        # PhospheneImage=cv.circle(blank.copy(),(200,200),200,(255,255,255),-1) #-1 means fill the circle
        # cv.imshow('asdsad',PhospheneImage)
        # resizedPhos = cv.resize(PhospheneImage, dimPhos, interpolation = cv.INTER_AREA)
      
        Inverse = cv.getTrackbarPos('Inverse','Control Panel')

        width = cv.getTrackbarPos('In Width','Control Panel')
        height = cv.getTrackbarPos('In Height','Control Panel')

        if width ==0:
            width+=1
            cv.setTrackbarPos('In Width','Control Panel',width) #only even num are allowed bec of output numpy array
        if height==0:
            height+=1
            cv.setTrackbarPos('In Height','Control Panel',height) #only even num are allowed bec of output numpy array
        
        dim = (width, height)

        img_copy=img.copy()
        List=PhosLocations(img) #return the pos of each phos and the color
        # print(len(Lis))
        # cv.imshow('adasd',TestImage)
        resized = cv.resize(img_copy, dim, interpolation = cv.INTER_AREA)
        
        gray =cv.cvtColor(resized,cv.COLOR_BGR2GRAY) # the gray scale image
     
        original_height=img_copy.shape[0] #must be the original value this one is an example
        original_width=img_copy.shape[1]
        # print(original_height)
        # print(original_width)

        phos_height= PhospheneSize # after the scale
        phos_width= PhospheneSize #after the scale
        input_height= len(gray) #after the sacel
        input_width= len(gray[0]) #after the scale
        
        # AdditionalSpaceWidth = (input_width-1) * SpaceBetween
        # AdditionalSpaceHeight = (input_height -1)*SpaceBetween
        output = np.zeros(((input_height*phos_height) , (input_width*phos_width) , 3), dtype=np.uint8)

        # distance= SpaceBetween
        bit = cv.getTrackbarPos('Gray Bits','Control Panel')
        StepSize=256/2**bit #if bit equal to 3 then 2^3=8 8 diff values of gray scale lets say its 32 for now
        if Inverse==1:
            InverseValue=255
            convert=1
        else :
            InverseValue=0
            convert=-1
        AllColorsList=[] 
       
        for i in range(len(List)): 
                x=List[i][0]  
                y=List[i][1]  
                color=List[i][2]  
                ratioy = original_height/y #must be obtained and others must respect this ratio 
                ratiox = original_width/x
                x=round(output.shape[1]/ratiox) 
                y=round(output.shape[0]/ratioy) 

                PixelColor =color//StepSize
                #colored will be sent from the edge class
                
                # _, result = cv.threshold(resizedPhos, 150, (InverseValue-(PixelColor*StepSize))*convert, cv.THRESH_BINARY) # this means will replace each white color in phosphene to the gray scale
                color =(InverseValue-(PixelColor*StepSize))*convert
                Allcolor=(color,color,color)
                AllColorsList.append((InverseValue-(PixelColor*StepSize))*convert)
                try:
                    # output[y-PhospheneSize//2:y+PhospheneSize//2 ,x-PhospheneSize//2:x+PhospheneSize//2]=result
                    PhospheneRadius=PhospheneSize//2 #radius must be an integer 
                    cv.circle(output,(x,y),PhospheneRadius,Allcolor,-1) 
                except:
                    print('error')
                    # distance+=10
                    # dataSpace[i*PhospheneSize:(i*PhospheneSize)+PhospheneSize ,j*PhospheneSize:(j*PhospheneSize)+PhospheneSize]=resizedPhos

        # data = np.zeros((h, w, 3), dtype=np.uint8)
        # data[0:15, 0:15] = circle  # red patch in upper left
        #height , width 
        # data[0:750, 0:750] = [0,0,0]  # red patch in upper left
        # data[15:30, 15:30] = circle  # red patch in upper left

        # fig = px.imshow(data)
        # fig.show()

        # fig = px.imshow(data)
        # fig.show('',fig)
                
        # FinalImage = Image.fromarray(data, 'RGB')
       
            # FinalImage.save('C:/Users/AHMED FAROUK/Desktop/becholer research/Output.jpg')
            # View=cv.imread('C:/Users/AHMED FAROUK/Desktop/becholer research/Output.jpg')
            # # cv.imwrite('photo/OutPutSave.jpg',View)
            # cv.imshow('Display2',View)

            # img10 = np.zeros((512,512,3), np.uint8)
       
      
        
       
        
        
        # space=30
        cv.imshow('Display2',output)
        # cv.imshow('Display3',img)
        # cv.imshow('Display2',output)

        # info=cv.imread('C:/Users/AHMED FAROUK/Documents/python computer vision/background/background.png')
        info = np.zeros((500,500,3), dtype=np.uint8) # create the black background
        InfoShow('Input Size',str(input_width),str(input_height),1,info)
        InfoShow('Phosphene Size',str(phos_width),str(phos_height),2,info)
        # InfoShow('Phosphene Space',SpaceBetween,'',110,info)
        InfoShow('Output Size',str(output.shape[1]),str(output.shape[0]),3,info)
        InfoShow('Nu Phosphenes',str(input_width),str(input_height),4,info)
        InfoShow('inverse',Inverse,'',5,info)
        InfoShow('Gray Bits',bit,'',6,info)
        InfoShow('Diff Color',2**bit,len(np.unique(np.array(AllColorsList))),7,info)
        
        InfoShow('Expected nuPhos',cv.getTrackbarPos('nuPhos','Control Panel'),'',8,info) #not less than that number 
        InfoShow('Expected Disbetween',cv.getTrackbarPos('Distance','Control Panel'),'',9,info) #each phosphene distance between
        InfoShow('Expected Accuracy',cv.getTrackbarPos('accuracy','Control Panel')/100,'',10,info) #less the number the less accuracy will be 1 -->means am sure abour the corner
        InfoShow('Expected Corner Radius',cv.getTrackbarPos('radius','Control Panel'),'',11,info) #less the number the less accuracy will be 1 -->means am sure abour the corner
        InfoShow('Util',len(List),input_height*input_width,12,info) #less the number the less accuracy will be 1 -->means am sure abour the corner
        
    
        
        if cv.waitKey(1)==27:
            cv.destroyAllWindows() 
            break         
            
              
def InfoShow(Attribute,Value1,Value2,space,image):
   
        FinalValue=''
        if Attribute=='Input Size' or Attribute=='Phosphene Size' or Attribute=='Output Size' :
            FinalValue=str(Value1)+'x'+str(Value2)+'px'
        elif  Attribute=='Nu Phosphenes':
            FinalValue=str(Value1)+'x'+str(Value2)
        elif Attribute=='Phosphene Space':
            FinalValue=str(Value1)+'px'
        elif Attribute=='Diff Color':
            FinalValue=str(Value2)+' outof '+str(Value1)
        elif Attribute =='Util':
            FinalValue=str(Value1) +' outof '+str(Value2)+':'+str(int((Value1/Value2)*100))+'%'
        else:
            FinalValue=str(Value1)


        
        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (0,space*40)
        fontScale              = 1
        fontColor              = (255,255,255)
        thickness              = 3
        lineType               = 2

        cv.putText(image,Attribute+':'+FinalValue, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)        
        # cv.namedWindow("All info" , cv.WINDOW_AUTOSIZE )    
        # cv.resizeWindow("All info", 600, 300)
        cv.imshow('All info',image)





# List=[[100,100,250],[200,200,100]]
# OriginalSize=[100,100] #width and height
#will pass 2 arrays first will contain the coordinates of each phosphenes+color -- and the other list is the original image size so we can rescpect the new postions of phophens depending on the new size of image
CreateFinalPhos()
# ReadFromImageBlackAndWhite()





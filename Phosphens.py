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
cv.resizeWindow("Control Panel", 1920, 230)
cv.moveWindow("Control Panel", 0,0)


cv.namedWindow("Display2")
cv.moveWindow("Display2", 0,270)

# cv.namedWindow("All Info")
# cv.moveWindow("All Info",200,500)

cv.createTrackbar('In Width','Control Panel',32,255,nothing)
cv.createTrackbar('In Height','Control Panel',32,255,nothing)
cv.createTrackbar('Pho Size','Control Panel',18,30,nothing)
cv.createTrackbar('Spacing','Control Panel',5,100,nothing) # in pixel


# cv.createTrackbar('Space Between Phosphens in Pixel','Control Panel',3,5,nothing)
cv.createTrackbar('Inverse','Control Panel',0,1,nothing)
cv.createTrackbar('Gray Bits','Control Panel',3,8,nextimage)
cv.createTrackbar('Next','Control Panel',0,1,nextimage)
cv.createTrackbar('Previous','Control Panel',0,1,nextimage)





path = glob.glob("photo/*")
cv_img = []
for img in path:
    n = cv.imread(img)
    cv_img.append(n)


SaveFile=0
def alternate():
    global SaveFile
    SaveFile+=1

def ReadFromImage(): #only one bit reprresentation black and white
   
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
            
        # cv.imshow('orig',img)
        
      
        
        PhospheneSize = cv.getTrackbarPos('Pho Size','Control Panel')
        SpaceBetween = cv.getTrackbarPos('Spacing','Control Panel')
        if PhospheneSize==0:
            PhospheneSize+=1
            cv.setTrackbarPos('Pho Size','Control Panel',PhospheneSize) #only even num are allowed bec of output numpy array

        # if widthPhos==0:
        #     widthPhos+=1
        # if heightPhos==0:
        #     heightPhos+=1
        widthPhos = cv.getTrackbarPos('Pho Size','Control Panel')
        heightPhos = cv.getTrackbarPos('Pho Size','Control Panel')
      
        dimPhos = (widthPhos, heightPhos)
        # PhospheneImage= cv.imread('C:/Users/AHMED FAROUK/Documents/python computer vision/Phosphenes/test2.png')

        blank=np.zeros((400,400,3),dtype=np.uint8)
        PhospheneImage=cv.circle(blank.copy(),(200,200),200,(255,255,255),-1) #-1 means fill the circle
        # cv.imshow('asdsad',PhospheneImage)
        resizedPhos = cv.resize(PhospheneImage, dimPhos, interpolation = cv.INTER_AREA)
        cv.imwrite('photo/The Phosphene.jpg',resizedPhos)
      
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
        resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        cv.imwrite('photo/resized.jpg',resized)
        gray =cv.cvtColor(resized,cv.COLOR_BGR2GRAY) # the gray scale image
        cv.imwrite('photo/gray.jpg',gray)
        # cv.imshow('gray',gray)
        # print(gray)
        #print(gray[0][0])
        #any value greater than the value 150 will be setted to 255 (white) else to black
       
        
        # print(gray[0][5])
        # if Inverse ==1: 
        #     _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
        # else :
        #      _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
       
        phos_height= len(resizedPhos) # after the scale
        phos_width= len(resizedPhos[0]) #after the scale
        input_height= len(gray) #after the sacel
        input_width= len(gray[0]) #after the scale
        
        AdditionalSpaceWidth = (input_width-1) * SpaceBetween
        AdditionalSpaceHeight = (input_height -1)*SpaceBetween
        output = np.zeros(((input_height*phos_height)+AdditionalSpaceHeight , (input_width*phos_width) +AdditionalSpaceWidth, 3), dtype=np.uint8)
        cv.imwrite('photo/output.jpg',output)
        distance= SpaceBetween
        bit = cv.getTrackbarPos('Gray Bits','Control Panel')
        StepSize=256/2**bit #if bit equal to 3 then 2^3=8 8 diff values of gray scale lets say its 32 for now
        if Inverse==1:
            InverseValue=255
            convert=1
        else :
            InverseValue=0
            convert=-1
        AllColorsList=[] 
        UsedPhosCounter=0
        for i in range(input_height):
            for j in range(input_width):
                PixelColor =gray[i][j]//StepSize
                # print(PixelColor*32)
                if gray[i][j]==255: 
                     _, result = cv.threshold(resizedPhos, 150, (InverseValue-255)*convert, cv.THRESH_BINARY) # this means will replace each white color in phosphene to the gray scale
                     AllColorsList.append((InverseValue-255)*convert) 
                else:
                    _, result = cv.threshold(resizedPhos, 150, (InverseValue-(PixelColor*StepSize))*convert, cv.THRESH_BINARY) # this means will replace each white color in phosphene to the gray scale
                    AllColorsList.append((InverseValue-(PixelColor*StepSize))*convert) 
                    if (InverseValue-(PixelColor*StepSize))*convert !=0: #all non black phosphenes will be counted
                       UsedPhosCounter+=1
                # print(len(result)) 
                # print('asd') 
                # print(len(result[0])) 
                # if gray[i][j]==255 : # means its white spot so i will replace it with circle in new image 
                    # SpacingWidth= PhospheneSize+AdditionalSpaceWidth
                    # SpacingHeight= PhospheneSize+AdditionalSpaceHeight
              
                
                output[(i*PhospheneSize)+(distance*i):((i*PhospheneSize)+(distance*i)+PhospheneSize) ,(j*PhospheneSize)+(distance*j):((j*PhospheneSize)+(distance*j)+PhospheneSize)]=result
                # cv.imwrite('photo/output.jpg',output)
                    # distance+=10
                    # dataSpace[i*PhospheneSize:(i*PhospheneSize)+PhospheneSize ,j*PhospheneSize:(j*PhospheneSize)+PhospheneSize]=resizedPhos

        # print(UsedPhosCounter)
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
       
      
        
       
        
        
        # cv.imwrite('C:/Users/AHMED FAROUK/Desktop/becholer research/img rep 1/rep.png', output)     
        cv.imshow('Display2',output)
        cv.imwrite('photo/final output.jpg',output)
        # info=cv.imread('C:/Users/AHMED FAROUK/Documents/python computer vision/background/background.png')
        info = np.zeros((400,400,3), dtype=np.uint8) # create the black background
        InfoShow('Input Size',str(input_width),str(input_height),30,info)
        InfoShow('Phosphene Size',str(phos_width),str(phos_height),70,info)
        InfoShow('Phosphene Space',SpaceBetween,'',110,info)
        InfoShow('Output Size',str((input_width*phos_width) +AdditionalSpaceWidth),str((input_height*phos_height)+AdditionalSpaceHeight),150,info)
        InfoShow('Nu Phosphenes',str(input_width),str(input_height),190,info)
        InfoShow('inverse',Inverse,'',230,info)
        InfoShow('Gray Bits',bit,'',270,info)
        InfoShow('Diff Color',2**bit,len(np.unique(np.array(AllColorsList))),310,info)
        InfoShow('Util',UsedPhosCounter,input_height*input_width,350,info) #less the number the less accuracy will be 1 -->means am sure abour the corner
    
        
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
        bottomLeftCornerOfText = (0,space)
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
        
        cv.imshow('All info',image)




# ReadFromImage()
def ReadFromImageTestSpace(): #only one bit reprresentation black and white
   
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
            
        # cv.imshow('orig',img)
        
      
        
        PhospheneSize = cv.getTrackbarPos('Pho Size','Control Panel')
        
        if PhospheneSize==0:
            PhospheneSize+=1
            cv.setTrackbarPos('Pho Size','Control Panel',PhospheneSize) #only even num are allowed bec of output numpy array

        # if widthPhos==0:
        #     widthPhos+=1
        # if heightPhos==0:
        #     heightPhos+=1
        widthPhos = cv.getTrackbarPos('Pho Size','Control Panel')
        heightPhos = cv.getTrackbarPos('Pho Size','Control Panel')
        SpaceBetween = cv.getTrackbarPos('Spacing','Control Panel')
      
        dimPhos = (widthPhos, heightPhos)
        # PhospheneImage= cv.imread('C:/Users/AHMED FAROUK/Documents/python computer vision/Phosphenes/test2.png')

        blank=np.zeros((widthPhos,heightPhos,3),dtype=np.uint8)
        if SpaceBetween >= widthPhos//2:
            SpaceBetween= (widthPhos//2)-1 
        resizedPhos=cv.circle(blank.copy(),(widthPhos//2,heightPhos//2),(widthPhos//2)-SpaceBetween ,(255,255,255),-1) #-1 means fill the circle
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
        resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        gray =cv.cvtColor(resized,cv.COLOR_BGR2GRAY) # the gray scale image
        # cv.imshow('gray',gray)
        # print(gray)
        #print(gray[0][0])
        #any value greater than the value 150 will be setted to 255 (white) else to black
       
        
        # print(gray[0][5])
        # if Inverse ==1: 
        #     _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
        # else :
        #      _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
       
        phos_height= len(resizedPhos) # after the scale
        phos_width= len(resizedPhos[0]) #after the scale
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
        UsedPhosCounter=0
        for i in range(input_height):
            for j in range(input_width):
                PixelColor =gray[i][j]//StepSize
                # print(PixelColor*32)
                if gray[i][j]==255: 
                     _, result = cv.threshold(resizedPhos, 150, (InverseValue-255)*convert, cv.THRESH_BINARY) # this means will replace each white color in phosphene to the gray scale
                     AllColorsList.append((InverseValue-255)*convert) 
                else:
                    _, result = cv.threshold(resizedPhos, 150, (InverseValue-(PixelColor*StepSize))*convert, cv.THRESH_BINARY) # this means will replace each white color in phosphene to the gray scale
                    AllColorsList.append((InverseValue-(PixelColor*StepSize))*convert) 
                    if (InverseValue-(PixelColor*StepSize))*convert !=0: #all non black phosphenes will be counted
                       UsedPhosCounter+=1
                # print(len(result)) 
                # print('asd') 
                # print(len(result[0])) 
                # if gray[i][j]==255 : # means its white spot so i will replace it with circle in new image 
                    # SpacingWidth= PhospheneSize+AdditionalSpaceWidth
                    # SpacingHeight= PhospheneSize+AdditionalSpaceHeight
              
                
                output[(i*PhospheneSize):((i*PhospheneSize)+PhospheneSize) ,(j*PhospheneSize):((j*PhospheneSize)+PhospheneSize)]=result
                    # distance+=10
                    # dataSpace[i*PhospheneSize:(i*PhospheneSize)+PhospheneSize ,j*PhospheneSize:(j*PhospheneSize)+PhospheneSize]=resizedPhos

        # print(UsedPhosCounter)
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
       
      
        

        # cv.imwrite('C:/Users/AHMED FAROUK/Desktop/becholer research/img rep 1/rep.png', output)
        cv.imshow('Display2',output)
        # info=cv.imread('C:/Users/AHMED FAROUK/Documents/python computer vision/background/background.png')
        info = np.zeros((400,400,3), dtype=np.uint8) # create the black background
        InfoShow('Input Size',str(input_width),str(input_height),30,info)
        InfoShow('Phosphene Size',str(phos_width),str(phos_height),70,info)
        InfoShow('Phosphene Space',SpaceBetween,'',110,info)
        InfoShow('Output Size',str(input_width*phos_width) ,str(input_height*phos_height),150,info)
        InfoShow('Nu Phosphenes',str(input_width),str(input_height),190,info)
        InfoShow('inverse',Inverse,'',230,info)
        InfoShow('Gray Bits',bit,'',270,info)
        InfoShow('Diff Color',2**bit,len(np.unique(np.array(AllColorsList))),310,info)
        InfoShow('Util',UsedPhosCounter,input_height*input_width,350,info) #less the number the less accuracy will be 1 -->means am sure abour the corner
    
        
        if cv.waitKey(1)==27:
            cv.destroyAllWindows() 
            break     



# ReadFromImageBlackAndWhite()

# ReadFromImageTestSpace()
ReadFromImage()


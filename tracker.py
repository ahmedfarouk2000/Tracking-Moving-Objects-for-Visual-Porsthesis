from ctypes import resize
from mimetypes import init
from platform import python_version
from posixpath import split
from sre_constants import SUCCESS
import cv2 as cv 
from matplotlib import pyplot as plt
import numpy as np

from skimage.io import imread_collection

import imageio

import os



import glob

def nothing(args):
    pass
def nextimage(args):
    pass



cv.namedWindow("Control" , cv.WINDOW_NORMAL)
cv.resizeWindow("Control", 1920, 290)

cv.moveWindow("Control", 0,0)
cv.createTrackbar('MIL','Control',0,1,nothing)
cv.createTrackbar('KCF','Control',0,1,nothing)

cv.createTrackbar('CSRT','Control',0,1,nothing)
cv.createTrackbar('MOSSE','Control',0,1,nothing)
cv.createTrackbar('MedianFlow','Control',0,1,nothing)
cv.createTrackbar('Boosting','Control',0,1,nothing)
cv.createTrackbar('TLD','Control',0,1,nothing)
cv.createTrackbar('GOTURN','Control',0,1,nothing)

cv.createTrackbar('Width','Control',550,2000,nothing)
cv.createTrackbar('Height','Control',350,2000,nothing)

cv.createTrackbar('Next','Control',0,1,nextimage)
cv.createTrackbar('Previous','Control',0,1,nextimage)
# cv.createTrackbar('width','Control',1,,nothing)
# cv.createTrackbar('height','Control',1,1,nothing)


def concat_vh(list_2d):
    
      # return final image
    return cv.vconcat([cv.hconcat(list_h) 
                        for list_h in list_2d])


def drawbox(img,bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv.rectangle(img, p1, p2, (255,0,255), 2, 1)
    # x,y,w,h =bbox[0],bbox[1],bbox[2],bbox[3]
    # cv.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1) 

def InfoShow(Value,space,image,color):
        # FinalValue=''
        # if Attribute=='Input Size' or Attribute=='Phosphene Size' or Attribute=='Output Size' :
        #     FinalValue=str(Value1)+'x'+str(Value2)+'px'
        # elif  Attribute=='Nu Phosphenes':
        #     FinalValue=str(Value1)+'x'+str(Value2)
        # elif Attribute=='Phosphene Space':
        #     FinalValue=str(Value1)+'px'
        # elif Attribute=='Diff Color':
        #     FinalValue=str(Value2)+' outof '+str(Value1)
        # elif Attribute =='Util':
        #     FinalValue=str(Value1) +' outof '+str(Value2)+':'+str(int((Value1/Value2)*100))+'%'    
        # else:
        FinalValue=str(Value)
        outcolor =(0,0,0)
        if color =='red':
            outcolor=(0,0,255)
        elif color =='green':
            outcolor=(0,255,0)
        elif color=='yellow':
            outcolor=(0,255,255)    
        else :
            outcolor=(255,255,255)

        
        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (0,space)
        fontScale              = 1
        fontColor              = outcolor
        thickness              = 3
        lineType               = 2

        cv.putText(image,FinalValue, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)        
        
        # cv.imshow('Display3',image)
def removeGap(inputarray):
    outarray =[]
    for i in range(len(inputarray)):
        if(inputarray[i]!='empty'):
            # print('aaa')
            outarray.append(inputarray[i])
    
    return outarray

def splitequal(inputarray):
    firstarray=[]
    sectarray=[]
    for i in range(len(inputarray)):
        if i%2==0:
            firstarray.append(inputarray[i])
        else:
            sectarray.append(inputarray[i])
    return firstarray,sectarray


path = glob.glob("photo/*")
cv_img = []
for img in path:
    n = cv.imread(img)
    cv_img.append(n)


def ReadFromCamera():
    camera=cv.VideoCapture(0)
    while True :
        _,FirstImage=camera.read()
        # cv.imshow("Display",FirstImage)
        bbox=cv.selectROI("Display",FirstImage,False)  # keep pressing on space so that the vidoe keep going 
        if bbox!=(0,0,0,0):    
            break
           
    TrackerKCF = cv.legacy.TrackerKCF_create()
    TrackerMIL=cv.TrackerMIL_create() #this one worked fine
    TrackerCSRT = cv.TrackerCSRT_create() #best one till now 
    TrackerMOSSE = cv.legacy.TrackerMOSSE_create()
    TrackerMedianFlow = cv.legacy.TrackerMedianFlow_create() # not that good
    TrackerBoosting = cv.legacy.TrackerBoosting_create()
    TrackerTLD = cv.legacy.TrackerTLD_create()
    TrackerGOTURN = cv.TrackerGOTURN_create()




    TrackerKCF.init(FirstImage,bbox)
    TrackerMIL.init(FirstImage,bbox)

    TrackerCSRT.init(FirstImage,bbox)
    TrackerMOSSE.init(FirstImage,bbox)

    TrackerMedianFlow.init(FirstImage,bbox)
    TrackerBoosting.init(FirstImage,bbox)
    TrackerTLD.init(FirstImage,bbox)

    TrackerGOTURN.init(FirstImage,bbox)

    AllTrack=['empty']*8   
    while True:
        
        width = cv.getTrackbarPos('Width','Control')
        height = cv.getTrackbarPos('Height','Control')
        dim = (width, height)

        timer=cv.getTickCount()
        KCF=cv.getTrackbarPos('KCF','Control')
        if KCF ==1 :
            _,TrackerKCFimg=camera.read() #success is a flag that return true if can read from cam and false if not
            TrackerKCFSucces,KCFbbox=TrackerKCF.update(TrackerKCFimg)
            
            if TrackerKCFSucces:
                drawbox(TrackerKCFimg,KCFbbox)  
                resizedTrackerKCFimg = cv.resize(TrackerKCFimg, dim, interpolation = cv.INTER_AREA)        
                InfoShow('TRACKING',30,resizedTrackerKCFimg,'green')
                InfoShow('KCF',60,resizedTrackerKCFimg,'white')
            else :
                resizedTrackerKCFimg = cv.resize(TrackerKCFimg, dim, interpolation = cv.INTER_AREA)   
                InfoShow('LOST',30,resizedTrackerKCFimg,'red') 
                InfoShow('KCF',60,resizedTrackerKCFimg,'white')
            AllTrack[0]=resizedTrackerKCFimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerKCFimg,'yellow')
            
        else :
            AllTrack[0]='empty'


        timer=cv.getTickCount()
        MIL=cv.getTrackbarPos('MIL','Control')
        if MIL ==1 :
            _,TrackerMILimg=camera.read() #success is a flag that return true if can read from cam and false if not
            TrackerMILSucces,MILbbox=TrackerMIL.update(TrackerMILimg)
        
            if TrackerMILSucces:
                drawbox(TrackerMILimg,MILbbox)
                resizedTrackerMILimg = cv.resize(TrackerMILimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerMILimg,'green')
                InfoShow('MIL',60,resizedTrackerMILimg,'white')
            else :
                resizedTrackerMILimg = cv.resize(TrackerMILimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerMILimg,'red')
                InfoShow('MIL',60,resizedTrackerMILimg,'white')
                
            AllTrack[1]=resizedTrackerMILimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerMILimg,'yellow')
            
        else :
            AllTrack[1]='empty'
            
        timer=cv.getTickCount()
        CSRT=cv.getTrackbarPos('CSRT','Control')
        if CSRT ==1 :
            _,TrackerCSRTimg=camera.read() #success is a flag that return true if can read from cam and false if not
            TrackerCSRTSucces,CSRTbbox=TrackerCSRT.update(TrackerCSRTimg)
        
            if TrackerCSRTSucces:
                drawbox(TrackerCSRTimg,CSRTbbox)
                resizedTrackerCSRTimg = cv.resize(TrackerCSRTimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerCSRTimg,'green')
                InfoShow('CSRT',60,resizedTrackerCSRTimg,'white')
            else :
                resizedTrackerCSRTimg = cv.resize(TrackerCSRTimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerCSRTimg,'red')
                InfoShow('CSRT',60,resizedTrackerCSRTimg,'white')
            AllTrack[2]=resizedTrackerCSRTimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerCSRTimg,'yellow')
         
        else :
            AllTrack[2]='empty'


        timer=cv.getTickCount()
        MOSSE=cv.getTrackbarPos('MOSSE','Control')
        if MOSSE ==1 :
            _,TrackerMOSSEimg=camera.read() #success is a flag that return true if can read from cam and false if not
            TrackerMOSSESucces,MOSSEbbox=TrackerMOSSE.update(TrackerMOSSEimg)
        
            if TrackerMOSSESucces:
                drawbox(TrackerMOSSEimg,MOSSEbbox)
                resizedTrackerMOSSEimg = cv.resize(TrackerMOSSEimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerMOSSEimg,'green')
                InfoShow('MOSSE',60,resizedTrackerMOSSEimg,'white')
            else :
                resizedTrackerMOSSEimg = cv.resize(TrackerMOSSEimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerMOSSEimg,'red')
                InfoShow('MOSSE',60,resizedTrackerMOSSEimg,'white')
            AllTrack[3]=resizedTrackerMOSSEimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerMOSSEimg,'yellow')
         
        else :
            AllTrack[3]='empty'
        
        timer=cv.getTickCount()
        MedianFlow=cv.getTrackbarPos('MedianFlow','Control')
        if MedianFlow ==1 :
            _,TrackerMedianFlowimg=camera.read() #success is a flag that return true if can read from cam and false if not
            TrackerMedianFlowSucces,MedianFlowbbox=TrackerMedianFlow.update(TrackerMedianFlowimg)
        
            if TrackerMedianFlowSucces:
                drawbox(TrackerMedianFlowimg,MedianFlowbbox)
                resizedTrackerMedianFlowimg = cv.resize(TrackerMedianFlowimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerMedianFlowimg,'green')
                InfoShow('MedianFlow',60,resizedTrackerMedianFlowimg,'white')
            else :
                resizedTrackerMedianFlowimg = cv.resize(TrackerMedianFlowimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerMedianFlowimg,'red')
                InfoShow('MedianFlow',60,resizedTrackerMedianFlowimg,'white')
            AllTrack[4]=resizedTrackerMedianFlowimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerMedianFlowimg,'yellow')
         
        else :
            AllTrack[4]='empty'


        timer=cv.getTickCount()
        Boosting=cv.getTrackbarPos('Boosting','Control')
        if Boosting ==1 :
                _,TrackerBoostingimg=camera.read() #success is a flag that return true if can read from cam and false if not
                TrackerBoostingSucces,Boostingbbox=TrackerBoosting.update(TrackerBoostingimg)
            
                if TrackerBoostingSucces:
                    drawbox(TrackerBoostingimg,Boostingbbox)
                    resizedTrackerBoostingimg = cv.resize(TrackerBoostingimg, dim, interpolation = cv.INTER_AREA)
                    InfoShow('TRACKING',30,resizedTrackerBoostingimg,'green')
                    InfoShow('Boosting',60,resizedTrackerBoostingimg,'white')
                else :
                    resizedTrackerBoostingimg = cv.resize(TrackerBoostingimg, dim, interpolation = cv.INTER_AREA)
                    InfoShow('LOST',30,resizedTrackerBoostingimg,'red')
                    InfoShow('Boosting',60,resizedTrackerBoostingimg,'white')
                AllTrack[5]=resizedTrackerBoostingimg
                fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
                InfoShow(str(int(fps)),90,resizedTrackerBoostingimg,'yellow')
            
        else :
                AllTrack[5]='empty'

        timer=cv.getTickCount()
        TLD=cv.getTrackbarPos('TLD','Control')
        if TLD ==1 :
            _,TrackerTLDimg=camera.read() #success is a flag that return true if can read from cam and false if not
            TrackerTLDSucces,TLDbbox=TrackerTLD.update(TrackerTLDimg)
        
            if TrackerTLDSucces:
                drawbox(TrackerTLDimg,TLDbbox)
                resizedTrackerTLDimg = cv.resize(TrackerTLDimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerTLDimg,'green')
                InfoShow('TLD',60,resizedTrackerTLDimg,'white')
            else :
                resizedTrackerTLDimg = cv.resize(TrackerTLDimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerTLDimg,'red')
                InfoShow('TLD',60,resizedTrackerTLDimg,'white')
            AllTrack[6]=resizedTrackerTLDimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerTLDimg,'yellow')
        else :
            AllTrack[6]='empty'

        timer=cv.getTickCount()
        GOTURN=cv.getTrackbarPos('GOTURN','Control')
        if GOTURN ==1 :
            _,TrackerGOTURNimg=camera.read() #success is a flag that return true if can read from cam and false if not
            TrackerGOTURNSucces,GOTURNbbox=TrackerGOTURN.update(TrackerGOTURNimg)
        
            if TrackerGOTURNSucces:
                drawbox(TrackerGOTURNimg,GOTURNbbox)
                resizedTrackerGOTURNimg = cv.resize(TrackerGOTURNimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerGOTURNimg,'green')
                InfoShow('GOTURN',60,resizedTrackerGOTURNimg,'white')
            else :
                resizedTrackerGOTURNimg = cv.resize(TrackerGOTURNimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerGOTURNimg,'red')
                InfoShow('GOTURN',60,resizedTrackerGOTURNimg,'white')
            AllTrack[7]=resizedTrackerGOTURNimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerGOTURNimg,'yellow')
        else :
            AllTrack[7]='empty'    

        out=removeGap(AllTrack)
        arr1,arr2=splitequal(out)
        blank=np.zeros((height,width,3),dtype=np.uint8)
        if len(arr1)!= len(arr2):
            arr2.append(blank)
            # img_tile = concat_vh([arr1])
        # if len(arr2)==0: #means we only has one pic at a time
        #     img_tile = concat_vh([arr1])

        # else :
        img_tile = concat_vh([arr1,arr2])
         
        # else :
        #     img_tile = concat_vh(
        #               [[resized, resized],
        #               [resized, resized]])
         
        # array=[1,2,3,4]
        # first,sec=splitequal(array)
        # print(first)
        # print(sec)
            # cv.imshow('Display2',   img_tile)
        # hor =np.hstack((img,img2))
        cv.imshow('Display2',   img_tile)
        # cv.imshow('Display2',img2)
        if cv.waitKey(1)=='a':
            cv.destroyAllWindows() 
            break





def ReadFromVideo():
    allVideos=os.listdir('C:/Users/AHMED FAROUK/Documents/python computer vision/video')
    totalImages=len(allVideos)
    counter=0 
    switchNext=0 
    switchPrev=0 
    while True :
        nextImg =cv.getTrackbarPos('Next','Control')
        prevImg =cv.getTrackbarPos('Previous','Control')
       
        if   nextImg==1 and switchNext==1 :
            counter+=1
            capture =cv.VideoCapture('video/'+allVideos[counter%totalImages]) 
            # capture =cv_img[counter%totalImages]
            switchNext=0 
          
        elif nextImg==0  and switchNext==0 :
            counter+=1
            capture =cv.VideoCapture('video/'+allVideos[counter%totalImages]) 
            switchNext=1
           
        elif prevImg==0 and  switchPrev==0 : 
             counter-=1
             capture =cv.VideoCapture('video/'+allVideos[counter%totalImages]) 
             switchPrev=1
            
        elif prevImg==1 and  switchPrev==1 :
             counter-=1
             capture =cv.VideoCapture('video/'+allVideos[counter%totalImages]) 
             switchPrev=0


        _,FirstImage=capture.read()
        # cv.imshow("Display",FirstImage)
        bbox=cv.selectROI("Display",FirstImage,False)  # keep pressing on space so that the vidoe keep going 
        if bbox!=(0,0,0,0):    
            break
           
    TrackerKCF = cv.legacy.TrackerKCF_create()
    TrackerMIL=cv.TrackerMIL_create() #this one worked fine
    TrackerCSRT = cv.TrackerCSRT_create() #best one till now 
    TrackerMOSSE = cv.legacy.TrackerMOSSE_create()
    TrackerMedianFlow = cv.legacy.TrackerMedianFlow_create() # not that good
    TrackerBoosting = cv.legacy.TrackerBoosting_create()
    TrackerTLD = cv.legacy.TrackerTLD_create()
    TrackerGOTURN = cv.TrackerGOTURN_create()




    TrackerKCF.init(FirstImage,bbox)
    TrackerMIL.init(FirstImage,bbox)

    TrackerCSRT.init(FirstImage,bbox)
    TrackerMOSSE.init(FirstImage,bbox)

    TrackerMedianFlow.init(FirstImage,bbox)
    TrackerBoosting.init(FirstImage,bbox)
    TrackerTLD.init(FirstImage,bbox)

    TrackerGOTURN.init(FirstImage,bbox)

    AllTrack=['empty']*8   
    while True:
        
        width = cv.getTrackbarPos('Width','Control')
        height = cv.getTrackbarPos('Height','Control')
        dim = (width, height)

        timer=cv.getTickCount()
        KCF=cv.getTrackbarPos('KCF','Control')
        if KCF ==1 :
            _,TrackerKCFimg=capture.read() #success is a flag that return true if can read from cam and false if not
            TrackerKCFSucces,KCFbbox=TrackerKCF.update(TrackerKCFimg)

            if TrackerKCFSucces:
                drawbox(TrackerKCFimg,KCFbbox)  
                resizedTrackerKCFimg = cv.resize(TrackerKCFimg, dim, interpolation = cv.INTER_AREA)        
                InfoShow('TRACKING',30,resizedTrackerKCFimg,'green')
                InfoShow('KCF',60,resizedTrackerKCFimg,'white')
            else :
                resizedTrackerKCFimg = cv.resize(TrackerKCFimg, dim, interpolation = cv.INTER_AREA)   
                InfoShow('LOST',30,resizedTrackerKCFimg,'red') 
                InfoShow('KCF',60,resizedTrackerKCFimg,'white')
            AllTrack[0]=resizedTrackerKCFimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerKCFimg,'yellow')
        else :
            AllTrack[0]='empty'


        timer=cv.getTickCount()
        MIL=cv.getTrackbarPos('MIL','Control')
        if MIL ==1 :
            _,TrackerMILimg=capture.read() #success is a flag that return true if can read from cam and false if not
            TrackerMILSucces,MILbbox=TrackerMIL.update(TrackerMILimg)
        
            if TrackerMILSucces:
                drawbox(TrackerMILimg,MILbbox)
                resizedTrackerMILimg = cv.resize(TrackerMILimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerMILimg,'green')
                InfoShow('MIL',60,resizedTrackerMILimg,'white')
            else :
                resizedTrackerMILimg = cv.resize(TrackerMILimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerMILimg,'red')
                InfoShow('MIL',60,resizedTrackerMILimg,'white')
            AllTrack[1]=resizedTrackerMILimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerMILimg,'yellow') 
        else :
            AllTrack[1]='empty'
            
        timer=cv.getTickCount()
        CSRT=cv.getTrackbarPos('CSRT','Control')
        if CSRT ==1 :
            _,TrackerCSRTimg=capture.read() #success is a flag that return true if can read from cam and false if not
            TrackerCSRTSucces,CSRTbbox=TrackerCSRT.update(TrackerCSRTimg)
        
            if TrackerCSRTSucces:
                drawbox(TrackerCSRTimg,CSRTbbox)
                resizedTrackerCSRTimg = cv.resize(TrackerCSRTimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerCSRTimg,'green')
                InfoShow('CSRT',60,resizedTrackerCSRTimg,'white')
            else :
                resizedTrackerCSRTimg = cv.resize(TrackerCSRTimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerCSRTimg,'red')
                InfoShow('CSRT',60,resizedTrackerCSRTimg,'white')
            AllTrack[2]=resizedTrackerCSRTimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerCSRTimg,'yellow')
         
        else :
            AllTrack[2]='empty'


        timer=cv.getTickCount()
        MOSSE=cv.getTrackbarPos('MOSSE','Control')
        if MOSSE ==1 :
            _,TrackerMOSSEimg=capture.read() #success is a flag that return true if can read from cam and false if not
            TrackerMOSSESucces,MOSSEbbox=TrackerMOSSE.update(TrackerMOSSEimg)
        
            if TrackerMOSSESucces:
                drawbox(TrackerMOSSEimg,MOSSEbbox)
                resizedTrackerMOSSEimg = cv.resize(TrackerMOSSEimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerMOSSEimg,'green')
                InfoShow('MOSSE',60,resizedTrackerMOSSEimg,'white')
            else :
                resizedTrackerMOSSEimg = cv.resize(TrackerMOSSEimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerMOSSEimg,'red')
                InfoShow('MOSSE',60,resizedTrackerMOSSEimg,'white')
            AllTrack[3]=resizedTrackerMOSSEimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerMOSSEimg,'yellow')
         
        else :
            AllTrack[3]='empty'
        
        timer=cv.getTickCount()
        MedianFlow=cv.getTrackbarPos('MedianFlow','Control')
        if MedianFlow ==1 :
            _,TrackerMedianFlowimg=capture.read() #success is a flag that return true if can read from cam and false if not
            TrackerMedianFlowSucces,MedianFlowbbox=TrackerMedianFlow.update(TrackerMedianFlowimg)
        
            if TrackerMedianFlowSucces:
                drawbox(TrackerMedianFlowimg,MedianFlowbbox)
                resizedTrackerMedianFlowimg = cv.resize(TrackerMedianFlowimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerMedianFlowimg,'green')
                InfoShow('MedianFlow',60,resizedTrackerMedianFlowimg,'white')
            else :
                resizedTrackerMedianFlowimg = cv.resize(TrackerMedianFlowimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerMedianFlowimg,'red')
                InfoShow('MedianFlow',60,resizedTrackerMedianFlowimg,'white')
            AllTrack[4]=resizedTrackerMedianFlowimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerMedianFlowimg,'yellow')
         
        else :
            AllTrack[4]='empty'


        timer=cv.getTickCount()
        Boosting=cv.getTrackbarPos('Boosting','Control')
        if Boosting ==1 :
                _,TrackerBoostingimg=capture.read() #success is a flag that return true if can read from cam and false if not
                TrackerBoostingSucces,Boostingbbox=TrackerBoosting.update(TrackerBoostingimg)
            
                if TrackerBoostingSucces:
                    drawbox(TrackerBoostingimg,Boostingbbox)
                    resizedTrackerBoostingimg = cv.resize(TrackerBoostingimg, dim, interpolation = cv.INTER_AREA)
                    InfoShow('TRACKING',30,resizedTrackerBoostingimg,'green')
                    InfoShow('Boosting',60,resizedTrackerBoostingimg,'white')
                else :
                    resizedTrackerBoostingimg = cv.resize(TrackerBoostingimg, dim, interpolation = cv.INTER_AREA)
                    InfoShow('LOST',30,resizedTrackerBoostingimg,'red')
                    InfoShow('Boosting',60,resizedTrackerBoostingimg,'white')
                AllTrack[5]=resizedTrackerBoostingimg
                fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
                InfoShow(str(int(fps)),90,resizedTrackerBoostingimg,'yellow')
        else :
                AllTrack[5]='empty'

        timer=cv.getTickCount()
        TLD=cv.getTrackbarPos('TLD','Control')
        if TLD ==1 :
            _,TrackerTLDimg=capture.read() #success is a flag that return true if can read from cam and false if not
            TrackerTLDSucces,TLDbbox=TrackerTLD.update(TrackerTLDimg)
        
            if TrackerTLDSucces:
                drawbox(TrackerTLDimg,TLDbbox)
                resizedTrackerTLDimg = cv.resize(TrackerTLDimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerTLDimg,'green')
                InfoShow('TLD',60,resizedTrackerTLDimg,'white')
            else :
                resizedTrackerTLDimg = cv.resize(TrackerTLDimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerTLDimg,'red')
                InfoShow('TLD',60,resizedTrackerTLDimg,'white')
            AllTrack[6]=resizedTrackerTLDimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerTLDimg,'yellow')
         
        else :
            AllTrack[6]='empty'

        timer=cv.getTickCount()
        GOTURN=cv.getTrackbarPos('GOTURN','Control')
        if GOTURN ==1 :
            _,TrackerGOTURNimg=capture.read() #success is a flag that return true if can read from cam and false if not
            TrackerGOTURNSucces,GOTURNbbox=TrackerGOTURN.update(TrackerGOTURNimg)
        
            if TrackerGOTURNSucces:
                drawbox(TrackerGOTURNimg,GOTURNbbox)
                resizedTrackerGOTURNimg = cv.resize(TrackerGOTURNimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('TRACKING',30,resizedTrackerGOTURNimg,'green')
                InfoShow('GOTURN',60,resizedTrackerGOTURNimg,'white')
            else :
                resizedTrackerGOTURNimg = cv.resize(TrackerGOTURNimg, dim, interpolation = cv.INTER_AREA)
                InfoShow('LOST',30,resizedTrackerGOTURNimg,'red')            
                InfoShow('GOTURN',60,resizedTrackerGOTURNimg,'white')
            AllTrack[7]=resizedTrackerGOTURNimg
            fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
            InfoShow(str(int(fps)),90,resizedTrackerGOTURNimg,'yellow')
        else :
            AllTrack[7]='empty'    

        out=removeGap(AllTrack)
        arr1,arr2=splitequal(out)
        blank=np.zeros((height,width,3),dtype=np.uint8)
        if len(arr1)!= len(arr2):
            arr2.append(blank)
            # img_tile = concat_vh([arr1])
        # if len(arr2)==0: #means we only has one pic at a time
        #     img_tile = concat_vh([arr1])

        # else :
        img_tile = concat_vh([arr1,arr2])
         
        # else :
        #     img_tile = concat_vh(
        #               [[resized, resized],
        #               [resized, resized]])
         
        # array=[1,2,3,4]
        # first,sec=splitequal(array)
        # print(first)
        # print(sec)
            # cv.imshow('Display2',   img_tile)
        # hor =np.hstack((img,img2))
        cv.imshow('Display2',   img_tile)
        # cv.imshow('Display2',img2)
        if cv.waitKey(1)=='a':
            cv.destroyAllWindows() 
            break




# ReadFromCamera()
ReadFromVideo()


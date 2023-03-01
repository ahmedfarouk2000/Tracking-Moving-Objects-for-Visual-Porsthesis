import imghdr
from tokenize import blank_re
from traceback import print_tb
from turtle import delay, width, window_width
from typing import Counter
from unittest.result import failfast
import cv2 as cv
import numpy as np 
import math
from kalmanfilter import KalmanFilter
import os
import glob
import time
from hungarian_algo import *

from Phosphene_Exepermint import *

def nothing(args):
    pass
def nextimage(args):
    pass




def InfoShow(image,area,percantage,x,y,IOU_percantage,x2,y2):
        # if area=='percantage':
        #     finalresult="Percantage:"+str(percantage)+"%"
        #     outcolor=(0,255,255)
        #     location=(x,y)
        # elif area=='counter':
        #     finalresult="Counter:"+str(percantage)
        #     outcolor=(0,255,255)
        #     location=(x,y)
    
        finalresult=str(area)+":"+str(percantage)+"%"
        outcolor=(0,255,0)
        location=(x-10,y-10)
       

        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = location
        fontScale              = 0.6
        fontColor              = outcolor
        thickness              = 1
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)    

        fontColor=(255,255,255)
        location=(x2,y2)
        bottomLeftCornerOfText = location
        finalresult=str("{:.2f}".format(IOU_percantage))
        cv.putText(image,finalresult, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)


def InfoShow2(image,area,x,y):
        # if area=='percantage':
        #     finalresult="Percantage:"+str(percantage)+"%"
        #     outcolor=(0,255,255)
        #     location=(x,y)
        # elif area=='counter':
        #     finalresult="Counter:"+str(percantage)
        #     outcolor=(0,255,255)
        #     location=(x,y)
    
        finalresult="Avg:"+str(area)
        outcolor=(0,255,0)
        location=(x-10,y-30)
       

        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = location
        fontScale              = 0.6
        fontColor              = outcolor
        thickness              = 1
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)    

def OrderShow(image,Order,x,y):
       
    
        finalresult="Order:"+str(Order)
        outcolor=(255,255,0)
        location=(x,y+20)
       

        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = location
        fontScale              = 0.6
        fontColor              = outcolor
        thickness              = 1
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)    


def infoshowRate(image,area,x2,y2):
        # if area=='percantage':
        #     finalresult="Percantage:"+str(percantage)+"%"
        #     outcolor=(0,255,255)
        #     location=(x,y)
        # elif area=='counter':
        #     finalresult="Counter:"+str(percantage)
        #     outcolor=(0,255,255)
        #     location=(x,y)
    
        finalresult="Rate:"+str(area)
        outcolor=(255,255,0)
        location=(x2,y2+20)
       

        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = location
        fontScale              = 0.6
        fontColor              = outcolor
        thickness              = 1
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)  


def infoshowRateAvg(image,area,SizeCover,x2,y2):
       
        Current_Size=0 
        if SizeCover>=4:
            Current_Size=4
        else :
            Current_Size=SizeCover

    
        finalresult="Avg Rate:"+str(area)+":"+str(Current_Size)
        outcolor=(255,255,0)
        location=(x2,y2+40)
       

        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = location
        fontScale              = 0.6
        fontColor              = outcolor
        thickness              = 1
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType) 


def infoshowRate4frames(image,area4frames,x2,y2):
       
      
        finalresult="RateNew:"+str(area4frames)  # using the prof method to see if it appr or going away
        outcolor=(255,255,0)
        location=(x2,y2+60)
       

        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = location
        fontScale              = 0.6
        fontColor              = outcolor
        thickness              = 1
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType) 


def infoshowRate4framesSign(image,Sign4frames,x2,y2):
       
      
        finalresult="Sign:"+str(Sign4frames)  # using the prof method to see if it appr or going away
        outcolor=(255,255,0)
        location=(x2,y2+80)
       

        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = location
        fontScale              = 0.6
        fontColor              = outcolor
        thickness              = 1
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType) 

def infoshowRate4frameCenterRate(image,CenterRate,x2,y2):
       
      
        finalresult='dis:'+str(CenterRate)  # using the prof method to see if it appr or going away
        outcolor=(255,255,0)
        # location=(x2,y2+100)
        location=(x2,y2+40)

        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = location
        fontScale              = 0.6
        fontColor              = outcolor
        thickness              = 1
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType) 
def infoshowRate4framesSignFinal(image,finalDec,x2,y2):

        finalresult='fin:'+str(finalDec)  # using the prof method to see if it appr or going away
        outcolor=(255,255,0)
        # location=(x2,y2+100)
        location=(x2,y2+60)

        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = location
        fontScale              = 0.6
        fontColor              = outcolor
        thickness              = 1
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType) 

def infoshowSpeed(image,speed,x2,y2):

        finalresult='S:'+str(speed)  # using the prof method to see if it appr or going away
        outcolor=(255,255,0)
        # location=(x2,y2+100)
        location=(x2-10,y2-50)

        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = location
        fontScale              = 0.6
        fontColor              = outcolor
        thickness              = 1
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType) 





def fail(image,amount):
        
           
        outcolor=(0,0,255)
        finalresult='failures:'+str(amount)
        location=(0,30)
        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = location
        fontScale              = 1
        fontColor              = outcolor
        thickness              = 2
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType) 


def Conflict(image,Conflictuse):
        
           
        outcolor=(0,0,255)
        finalresult=str(Conflictuse)
        location=(0,60)
        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = location
        fontScale              = 1
        fontColor              = outcolor
        thickness              = 2
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)   

def FrameNum(image,FrameCounter):
        
           
        outcolor=(0,0,255)
        finalresult='FrameNu'+str(FrameCounter)
        location=(0,90)
        # outcolor=(0,255,0)
        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = location
        fontScale              = 1
        fontColor              = outcolor
        thickness              = 2
        lineType               = 2

        cv.putText(image,finalresult, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)  


#didnt make yet the Sorted algo on the video functionnnnnnn



Old_Path = glob.glob("D:/python computer vision/new frame objects/*")
New_Path = glob.glob("D:/python computer vision/new frame objects/*")
Old_Objetcs = []
for img in Old_Path:
    n = cv.imread(img)
    Old_Objetcs.append(n)

New_Objetcs = []
for img in New_Path:
    n = cv.imread(img)
    New_Objetcs.append(n) 



def TrackVideoTEST():

    global margin
    # margin=10
    margin=10

    global CurrentIndex # length of current index
    CurrentIndex=6 # so will insert start from 5th index
    
    cv.namedWindow("Control" , cv.WINDOW_NORMAL)
    cv.resizeWindow("Control", 1920, 290)
    cv.createTrackbar('Percentage','Control',10,10,nothing)
    cv.createTrackbar('Switch','Control',0,1,nothing)


    capture =cv.VideoCapture('video/car5_Trim.mp4') 
    # capture = cv.VideoCapture(0)
    # capture.set(cv.CAP_PROP_FPS, 8)
    # capture.set(cv.cv.CV_CAP_PROP_FPS, 10)
    net = cv.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    frame_size = (frame_width,frame_height)
    print('FRAAAAAAAME_ SSSSSSSSSSSSSSIZEEEEEEEEEE111111111q')
    print(frame_size)
  
    # print(frame_size)
    global output
    output = cv.VideoWriter('video/trackresult/output_video_from_file.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, frame_size)
   
    global frame_size_mask
    frame_size_mask =getDimentions()
    print('FRAAAAAAAME_SZIEEEEEEEEEEEEEEE_MASK')
    print(frame_size_mask)

    global output_Phos
    output_Phos = cv.VideoWriter('video/trackresult/output_video_from_file_Pho.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_size_mask[0],frame_size_mask[1]))
    
    global output_Mask
    output_Mask = cv.VideoWriter('video/trackresult/output_video_from_file_masked.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_size[0]+margin+margin,frame_size[1]+margin+margin))

    global output_Border_Phos
    output_Border_Phos = cv.VideoWriter('video/trackresult/output_video_from_file_Pho_Temp.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_size_mask[0],frame_size_mask[1]))

    global output_Border_Phos_NoFlash
    output_Border_Phos_NoFlash = cv.VideoWriter('video/trackresult/output_video_from_file_Pho_NoFlash.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_size_mask[0],frame_size_mask[1]))


    global OriginalVideo
    OriginalVideo = cv.VideoWriter('video/trackresult/output_video_from_file_Pho_Original.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_size_mask[0],frame_size_mask[1]))
    # output_Mask_Phosphene = cv.VideoWriter('video/trackresult/output_video_from.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, frame_size)
    

    First_Frame=0 
    Counter=0
    only_first_clip =False
    # Alldim=[]
    while True :
       
  
        # img=cv.imread('photo/humans.jpg')
        _,img=capture.read() #success is a flag that return true if can read from cam and false if not
        # time.sleep(1)
        img_bbox=img.copy() 

        OrginalVideoPho=ReadFromImage(img)
        OriginalVideo.write(OrginalVideoPho)  

        

       
     
        height,width,_= img.shape
        blank=np.zeros((height,width),np.uint8)
   

        blob =cv.dnn.blobFromImage(img,swapRB=True)
        net.setInput(blob)
        boxes,masks =net.forward(['detection_out_final',"detection_masks"])
        detection_count=boxes.shape[2] #always will detect max of 100 object in the scene
        # print(detection_count)
        # postprocess(boxes, masks)
        # t, _ = net.getPerfProfile()
        img_copy=img.copy()

        Percentage =cv.getTrackbarPos('Percentage','Control')
            #box[1]-->classtypeId, box[2]-->confidance
        Object_Counter=0 
        
        saveMe_blank=np.zeros((frame_size_mask[0],frame_size_mask[1]),np.uint8) # where i wll save the framed produced by saveMe
        saveMe_blank = cv.cvtColor(saveMe_blank,cv.COLOR_GRAY2RGB)
        for i in range (detection_count):
            #box[1]-->classtypeId, box[2]-->confidance
            box=boxes[0,0,i]
            # print(box[0])
            #class_id =0 are the humans
            #first box[0] is a flag that define 0 not under any class of objects otherwise 1 under the class he  detected for
            class_id=box[1]
            print("class_id=",class_id)
            if ( class_id!=0): # only cars and humans # 2 car #17 dog #16 cat
                continue
            # if class_id==0 : #0 this means its a human
            score=box[2]
            if score < 0.01:
                continue
            x=int(box[3]*width)
            y=int(box[4]*height)
            x2=int(box[5]*width)
            y2=int(box[6]*height)
            BboxWidth=x2-x
            BboxHeight=y2-y
            BboxArea = BboxHeight* BboxWidth
            # print(width)
            WindowArea=width*height
            # if (BboxArea<50000):
            #     continue
            # current_percantage=int((BboxArea/WindowArea)*100)
        
            # cv.rectangle(img_bbox, (x, y), (x2, y2), (255, 255, 255), 3) 
            dimensions={'x':x,'x2':x2,'y':y,'y2':y2}

            add_y=0
            if y2 +margin >img_bbox.shape[0] :
                add_y=margin
                print('iffffff ya trash')
                        #here
            add_x=0
            if x2 +margin >img_bbox.shape[1] :
                add_x=margin
                print('iffffff ya trash22')
                

            add_y_before=0
            if y -margin <0 :
                add_y_before=margin
                print('iffffff ya trash3') 
                #here

            add_x_before=0
            if x -margin <0 :
                add_x_before=margin
                print('iffffff ya trash4')
              
            bonus_y=0
            if y -margin <0 and  y2 +margin >img_bbox.shape[0] :
                bonus_y=margin
                print('iffffff ya trash5')
                #here

            bonus_x=0
            if x -margin <0 and  x2 +margin >img_bbox.shape[1] :
                bonus_x=margin
                print('iffffff ya trash6')


            cropped_img_mask=img_bbox[y-margin-add_y+add_y_before+bonus_y:y2+margin -add_y+add_y_before-bonus_y,x-margin-add_x+add_x_before+bonus_x:x2+margin-add_x+add_x_before+bonus_x]
            print('y==',y)
            print('y2==',y2)
            print('x==',x)
            print('x2==',x2)
            print('y2+margin',y2+margin)
            print('y-margin',y-margin)
            print('height==',(y2+margin)-(y-margin))
            print('cropppped==',cropped_img_mask.shape)
            # cropped_img_mask =cv.cvtColor(cropped_img_mask,cv.COLOR_BGR2GRAY)



            roi =blank[y:y2 , x:x2] # this is the small object of interset we have execlusded from the image one by one
            roi_height,roi_wdith= roi.shape
            mask =masks[i,int(class_id)] ## very very small its just 15x15 pixel size so we must scale it back to be bigger or the same original size
            mask=cv.resize(mask,(roi_wdith,roi_height))
            _,mask=cv.threshold(mask,0.1,255,cv.THRESH_BINARY)  
            contours,_=cv.findContours(np.array(mask, np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for i in contours:  
                cv.fillPoly(roi, [i], (255,255,255)) # this will fill the gaps no diff between it and mask lets test that
            # MedianBlur = cv.medianBlur(blank, ksize=21)
          
            # margin_right=10
            height_temp= (roi_height )+margin+margin
            width_temp= (roi_wdith )+margin+margin
          
            blank_temp=np.zeros((height_temp,width_temp),np.uint8)
            # blank_temp2=np.zeros((height_temp,width_temp,3),np.uint8)
        
            print('roi size===',roi.shape)
            print('blank_temp===',blank_temp.shape)

            print('x and y',y2-y, x2-x)

          
            
            
         
            # cv.imshow('current mask',mask)
            # cv.imshow('roi',roi)
            # cv.imshow('cropped_img_mask',cropped_img_mask)
         

            # print('cropped_img_mask==',cropped_img_mask.shape)
            # print('mask==',mask.shape)
            MedianBlur = cv.medianBlur(roi, ksize=51)
            # cv.imshow('medina',MedianBlur)
            # cv.imwrite('C:/Users/AHMED FAROUK/Documents/python computer vision/video/trackresult/expermint/test.jpg',MedianBlur)

            # roi = cv.resize(roi,(roi_wdith,roi_height), interpolation = cv.INTER_AREA)
            # cv.imshow('roi',roi)

            # roi=cv.resize(roi,(roi_wdith,roi_height))
           
            blank_temp[ margin:margin+roi_height , margin:margin+roi_wdith ]=MedianBlur
            # cv.imshow('bblank temp',blank_temp)

            CannyFliter =cv.Canny(blank_temp , 0,255)
            # cv.imshow('canny',CannyFliter)

            kernel = np.ones((5,5),np.uint8)

            kernel_gradient = np.ones((5,5),np.uint8)

            dilation = cv.dilate(CannyFliter,kernel,iterations = 1) # to increase the thickness of canny edge 
            # gradient = cv.morphologyEx(blank_temp, cv.MORPH_GRADIENT, kernel_gradient) #same as dilation but on mask
            # cv.imshow('gradient',gradient)

            # cv.imshow('dilation',dilation)
          

            dilation_huge= cv.dilate(CannyFliter,kernel,iterations = 1) # to increase the thickness of canny edge # sent to pho as(mask)
            # cv.imshow('dilation_huge',dilation_huge)
            inverted_image = cv.bitwise_not(dilation_huge)

            

            # cv.imshow('inverted_image',inverted_image)
            print('cropped size=',cropped_img_mask.shape)
            print('inverted=',inverted_image.shape)
       
            # print('all the calc',y-margin-add_y+add_y_before+bonus_y)
            # print('all the calc2',y2+margin -add_y+add_y_before-bonus_y)
            test_tempo=cv.bitwise_and(cropped_img_mask,cropped_img_mask,mask=inverted_image)
            
                # cropped_img_mask=img_bbox[y-margin-add_y+add_y_before+bonus_y:y2+margin -add_y+add_y_before-bonus_y,x-margin-add_x+add_x_before+bonus_x:x2+margin-add_x+add_x_before+bonus_x]

            # cv.imshow('test_mask',test_tempo)
           
            blank_temp = cv.cvtColor(blank_temp,cv.COLOR_GRAY2RGB)

            ImgWithoutBorder=cv.bitwise_and(test_tempo,blank_temp) # to be sent to pho expermint (img)
            BorderImg = cv.cvtColor(dilation_huge,cv.COLOR_GRAY2RGB) # the border mask

            FinalBorderImg=cv.bitwise_or(ImgWithoutBorder,BorderImg) # el final result fl pho this line  elmask el3ady mn 8eer border
          

            # cv.imshow('test_mask_final',test_mask)



            # cv.imshow('test_mask',test_mask)
            # test_mask_new=cv.bitwise_and(cropped_img_mask,cropped_img_mask,mask=test_mask)
            # cv.imshow('test_mask_new',test_mask_new)

            # saveMe_Border=ReadFromImageTEST(ImgWithoutBorder,BorderImg,height,width,margin,x,x2,y,y2)
            # saveMe_blank=cv.bitwise_or(saveMe_blank,saveMe_Border)
          

            # cv.imshow('readdddd,', saveMe_Border)
            
            # chnage this so it can save more than one object together .....


            backtorgb = cv.cvtColor(dilation,cv.COLOR_GRAY2RGB)
            final_mask=FinalBorderImg # With white bordeeeeeeer elmask el3ady mn 8eer border so will get resized 3ady and white wont appear
            # final_mask=test_img  # without white border 

            # final_mask=cv.bitwise_and(cropped_img_mask,cropped_img_mask,mask=MedianBlur)
            # print('finallllll mask======')
            # cv.imshow('numpyy',final_mask)
            # cv.waitKey(0)
            # print(final_mask)
            #this mask is the one will be sent with each object so that can retrive it later to be shown
            # cv.imshow('cropped_img_mask',final_mask)

            # cv.imshow('final_mask',final_mask)
            # cv.waitKey(0)
        
           


            if(BboxWidth>100): #wont fail to detect 
                # print('if')
                cropped_img=img_bbox[y:y2,x:x2]
                if First_Frame==0 :
                        # cv.imwrite('C:/Users/AHMED FAROUK/Documents/python computer vision/old frame objects/old'+str(Object_Counter)+'.jpg',cropped_img)
                        # Current_insert ={'image':cropped_img ,'scale_percent':1}
                        # Current_dimensions={'x':x,'y':y,'x2':x2,'y2':y2}
                        Current_insert=(cropped_img,1,dimensions,class_id,final_mask,ImgWithoutBorder,BorderImg)
                        
                        Old_Objetcs.append(Current_insert)
                        # Object_Counter+=1
                else :
                        # cv.imwrite('C:/Users/AHMED FAROUK/Documents/python computer vision/new frame objects/new'+str(Object_Counter)+'.jpg',cropped_img)
                        # Object_Counter+=1
                        # Current_dimensions={'x':x,'y':y,'x2':x2,'y2':y2}
                        # Current_insert ={'image':cropped_img ,'scale_percent':1}
                        Current_insert=(cropped_img,1,dimensions,class_id,final_mask,ImgWithoutBorder,BorderImg)
                        New_Objetcs.append(Current_insert)
                        # Alldim.append(dimensions)
                    
            else: # must scale the image so orb can detect 
                # print('else ')
                cropped_img=img_bbox[y:y2,x:x2]
                scale_percent = 2 # percent of original size the best approach till now is the 2 percantage
                cropped_width,cropped_height,_=cropped_img.shape
                widthCrop = int(cropped_width * scale_percent )
                heightCrop = int(cropped_height * scale_percent )
                dimCrop = (heightCrop, widthCrop)
                # try:
                resized = cv.resize(cropped_img, dimCrop, interpolation = cv.INTER_AREA)
                # except:
                #     pass
                if First_Frame==0 :
                    # cv.imwrite('C:/Users/AHMED FAROUK/Documents/python computer vision/old frame objects/old'+str(Object_Counter)+'.jpg',cropped_img)
                    # Current_insert ={'image':resized ,'scale_percent':scale_percent}
                    # Current_dimensions={'x':x,'y':y,'x2':x2,'y2':y2}
                    Current_insert=(resized,scale_percent,dimensions,class_id,final_mask,ImgWithoutBorder,BorderImg)
                    Old_Objetcs.append(Current_insert)
                    # Object_Counter+=1
                else :
                    # cv.imwrite('C:/Users/AHMED FAROUK/Documents/python computer vision/new frame objects/new'+str(Object_Counter)+'.jpg',cropped_img)
                    # Object_Counter+=1
                    # Current_dimensions={'x':x,'y':y,'x2':x2,'y2':y2}
                    Current_insert=(resized,scale_percent,dimensions,class_id,final_mask,ImgWithoutBorder,BorderImg)
                    New_Objetcs.append(Current_insert)
                    # Alldim.append(dimensions)

        # output_Temp_Phos.write(saveMe_blank) #saving the threshold border video        
        # cv.imshow('readdddd,', saveMe_blank)

        if First_Frame==0:
            First_Frame=1 
        else: #must now compare each frame objetcs old/new
            Feature_Det(img_bbox)
             
                # cv.waitKey(0)
            # Alldim.clear()
            # time.sleep(1) 
            removeAllOld()
            # time.sleep(1)
            NewtoOld()
                
                    
        cv.imshow('asdasd',img_bbox)    
        output.write(img_bbox)
        print('SEEEEEEEEE TYPEEEEEEEE ')
        print(img_bbox.shape)
            # cv.imshow('frameeee',img_bbox)      

                # cv.imshow("Image", img_bbox) 
        if cv.waitKey(1)=='a':
                    cv.destroyAllWindows() 
                    break

def removeAllOld():
    # files = glob.glob('C:/Users/AHMED FAROUK/Documents/python computer vision/old frame objects/*')
    # for i in files:
    #     os.remove(i)
    Old_Objetcs.clear()   
def NewtoOld():
    #  newfiles = glob.glob('C:/Users/AHMED FAROUK/Documents/python computer vision/new frame objects/*')
    #  oldfiles = glob.glob('C:/Users/AHMED FAROUK/Documents/python computer vision/old frame objects/*')
     Object_Counter=0
     for i in range(len(New_Objetcs)):
        # current_img = cv.imread(img)
        # cv.imwrite('C:/Users/AHMED FAROUK/Documents/python computer vision/old frame objects/old'+str(Object_Counter)+'.jpg',current_img)
        Old_Objetcs.append(New_Objetcs[i])
        # os.remove(img)
        # Object_Counter+=1
     New_Objetcs.clear()


# print(len(Old_Objetcs))
# print(len(New_Objetcs))
# removeAllOld()
# NewtoOld()
# print(len(Old_Objetcs))
# print(len(New_Objetcs))

Fail_counter=0
wowarray=[]
current_frame=0
Diff=abs(len(New_Objetcs)-len(Old_Objetcs))
New_Greater=False

 
   

def Feature_Det(frame):
    frame_height,frame_width,_=frame.shape
    frame_height=int(frame_height)
    frame_width=int(frame_width)
    frame_size = (frame_width,frame_height)
    print('FRAAAAAAAME_ SSSSSSSSSSSSSSIZEEEEEEEEEE22222222222')
    print(frame_size)

    # Phos_vidoe= cv.VideoWriter('video/trackresult/output_video_from_file_mask.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, frame_size)


    Mask_frame=frame.copy()
    All=[]
    Best_matches=[]
    global Fail_counter
    global current_frame

    try :
        MaxAmong= max(len(Old_Objetcs),len(New_Objetcs))
        IOU_Array =[[0 for i in range(MaxAmong)] for j in range(MaxAmong)]
        IOU_Array=np.array(FillUnionOverInterSection(IOU_Array)) # done filling the gaps now u just must apply the hungerian algo so that u see the best intersection value
        profit_matrix= IOU_Array.copy()# done filling the gaps now u just must apply the hungerian algo so that u see the best intersection value
        max_value = np.max(IOU_Array)
        cost_matrix = max_value - IOU_Array
        ans_pos = hungarian_algorithm(cost_matrix.copy())#Get the element position.
        _, IOU_Array_Result = ans_calculation(profit_matrix, ans_pos)#Get the minimum or maximum value and corresponding matrix.#Show the result	print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")
        print(IOU_Array)
        print('aaaaa')
        print(IOU_Array_Result)
        print('spaaaaaaaaaaaaace')

        Current_Matching =[[0 for i in range(MaxAmong)] for j in range(MaxAmong)]
        Current_Matching=np.array(FillMatching(Current_Matching)) # done filling the gaps now u just must apply the hungerian algo so that u see the best intersection value
        profit_matrix_mataching= Current_Matching.copy()# done filling the gaps now u just must apply the hungerian algo so that u see the best intersection value

        max_value_new = np.max(Current_Matching)
        cost_matrix_new = max_value_new - Current_Matching
        ans_pos_new = hungarian_algorithm(cost_matrix_new.copy())#Get the element position.
        _, Current_Matching_Reuslt = ans_calculation(profit_matrix_mataching, ans_pos_new)#Get the minimum or maximum value and corresponding matrix.#Show the result	print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")
        print(Current_Matching)
        print('aaaaa')
        print(Current_Matching_Reuslt)
    except :
            print('fail in hungerian algo')
    # cv.waitKey(0)
    # kf = KalmanFilter()

    

    # print(New_Objetcs)
    Compare_String=''
    for i in range(len(New_Objetcs)):
       
        New_Object=New_Objetcs[i][0]
        scale_percent_new=New_Objetcs[i][1]
        New_area=New_Object.shape[0]*New_Object.shape[1]
        New_Dimensions=New_Objetcs[i][2]
        New_Class=New_Objetcs[i][3]
        New_Mask=New_Objetcs[i][4]
        ImgWithoutBorderNew=New_Objetcs[i][5]
        BorderImgNew=New_Objetcs[i][6]
        
        
        # print(New_Dimensions)
        
        for j in range(len(Old_Objetcs)):
        
            Old_Object=Old_Objetcs[j][0]
            scale_percent_old=Old_Objetcs[j][1]
            Old_area=Old_Object.shape[0]*Old_Object.shape[1]
            Old_Dimensions=Old_Objetcs[j][2]
            Old_Class=Old_Objetcs[j][3]
            Old_Mask=Old_Objetcs[j][4]
            ImgWithoutBorderOld=Old_Objetcs[j][5]
            BorderImgOld=Old_Objetcs[j][6]
            # IOU_Score=UnionOverInterSection(Old_Dimensions,New_Dimensions)
            IOU_Score=IOU_Array_Result[i][j]
            Matching_Score=Current_Matching_Reuslt[i][j]

            # if value <0:
            #     continue
          
            sift = cv.SIFT_create()
            # surf = cv.xfeatures2d.SURF_create(500)

            # orb = cv.ORB_create()
            kpOld, desOld = sift.detectAndCompute(Old_Object, None)
            kpNew, desNew = sift.detectAndCompute(New_Object, None)

            bf=cv.BFMatcher()
            try:
               
                matches=bf.knnMatch(desOld,desNew,k=2)
                good_matches=[]
                for m,n in matches:
                    if m.distance<0.75*n.distance and Old_Class==New_Class:
                        #TO BE SURE THAT MISS MATCH WONT HAPPEND BETWEEN DIFFERENT CLASSES
                        good_matches.append([m])

                # kf= KalmanFilter ()
                # pre=kf.predict(x_old,y_old)
                # pre=kf.predict(x_new,y_new)
                # print('cooredinates 1='+str(pre))
                # print('cooredinates 2='+str(pre))
                # print(pre)
                # cv.circle(frame,pre,50,(255,0,0),-1)
                deresized_new=New_area//(scale_percent_new*scale_percent_new)
                deresized_old=Old_area//(scale_percent_old*scale_percent_old)

                Weighted_Area=WeightedAcross2Frames(deresized_old,deresized_new) #HEEEEEEEEEEEEREEEEEEEEEEEEE
                
                Current_match ={'New_index':i ,'Old_index':j,'good_match':len(good_matches),'Old_area':Old_area,'New_area':New_area,'Weighted_Area':Weighted_Area
                ,'scale_percent_old':scale_percent_old,'scale_percent_new':scale_percent_new 
                ,'New_Dimensions':New_Dimensions,'Old_Dimensions':Old_Dimensions ,'IOU_Score':IOU_Score,'Matching_Score':Matching_Score,'New_Object':False
                ,'New_Class':New_Class,'Old_Class':Old_Class,'SpeedY':0,'CenterOfMassY':0,'New_Mask':New_Mask,'Old_Mask':Old_Mask
                ,'ImgWithoutBorderNew':ImgWithoutBorderNew,'ImgWithoutBorderOld':ImgWithoutBorderOld ,'BorderImgNew':BorderImgNew,'BorderImgOld':BorderImgOld}
                # print(j)
                # print('heere man')
                All.append(Current_match)
            except:
                print('oppppppps')
                
            # print(matches)

           

            # img_Final=cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=2)
        # print(len(All))        
        All.sort( reverse=True,key=myFunc2) # this means the first or greater value of good_match is the same object
        EnteredLoop =False
        Result_String=(Conflict_Result(IOU_Array_Result,Current_Matching_Reuslt,IOU_Array,Current_Matching))
        print(Conflict_Result(IOU_Array_Result,Current_Matching_Reuslt,IOU_Array,Current_Matching))
        
        try:
            if Result_String=='USE IOU1' or Result_String=='USE IOU2':
                    Compare_String='IOU_Score'
            else : #either use mtaching or no conflict then will use matching score
                    Compare_String='Matching_Score'
            for k in range(len(All)):
                if All[k][Compare_String] !=0 : # to double check that the one with most descriptors matching has the number found from hangerian algo
                    #HERE I AM SURE THAT OBJECT MATCHED CORRECTLY
                    # print('Old Objecttttttttt')
                    # print('current i='+str(k))
                    # print(All[k]['IOU_Score'])
                    New_area=All[k]['New_area']
                    Old_area=All[k]['Old_area']
                    scale_percent_new=All[k]['scale_percent_new']
                    scale_percent_old=All[k]['scale_percent_old']
                    
                    deresized_new=New_area//(scale_percent_new*scale_percent_new)
                    deresized_old=Old_area//(scale_percent_old*scale_percent_old)

                    new_index=All[k]['New_index']
                    Weighted_Area_new=All[k]['Weighted_Area']
                    Current_New_Rate=  RateOfChangeCal(deresized_old,deresized_new) # to be stored like weighted
                    

                    # print('old area='+str(deresized_old)) 
                    # print('new area='+str(deresized_new)) 
                    # print(Current_New_Rate)
                 
                    New_Dimensions=All[k]['New_Dimensions']
                    x_new,x2_new,y_new,y2_new=New_Dimensions['x'],New_Dimensions['x2'],New_Dimensions['y'],New_Dimensions['y2']
                    BboxWidth_New=x2_new-x_new
                    BboxHeight_New=y2_new-y_new
                    # Current_CenteOfMass=((x_new+(BboxWidth_New//2)),(y_new+(BboxHeight_New//2)))
                    Current_CenteOfMass=((x_new+(BboxWidth_New//2)),y2_new) #x will be from center of mass and y from y2



                    try : # using this to get the sign and afterwards storres it !!
                        old_index=All[k]['Old_index']
                        temp_array=Old_Objetcs[old_index][CurrentIndex+4] #will found all previous areas here
                        temp_array= [*temp_array,deresized_new]
                        ResultOfNewAvgAreasSign= int(RateAcross4Frames(temp_array))
                        print('ResultOfNewAvgAreasSign',ResultOfNewAvgAreasSign)
                    except :
                        ResultOfNewAvgAreasSign=0 # so that it wont cuz a problem
                        print('failed ResultOfNewAvgAreasSign')


                    try :
                        old_index=All[k]['Old_index']
                        temp_array=Old_Objetcs[old_index][CurrentIndex+3] #to store the current center of mass
                        temp_array= [*temp_array,Current_CenteOfMass] # contain current(new) and previous center of mass so now i can calc speed here
                        ResultSpeed=SpeedInY4Frames(temp_array)
                        # if (ResultSpeed=='not 4 frames yet Speed'):
                        #     ResultSpeed=(0,0)
                        print('try result of speeds',ResultSpeed)
                    except :
                        ResultSpeed=(0,0)
                        print('failed to calc speed')


                    try :
                        old_index=All[k]['Old_index']
                        All_Previous_States=Old_Objetcs[old_index][CurrentIndex+7] #to store the current center of mass
                        ResultState=FindTheNewState(All_Previous_States)
                        print('try result of ResultState',ResultState)
                    except :
                        ResultState=256 #first time its white by default
                        print('failed to Find Previous State')



                    # Middle_Point2=((x_old+(BboxWidth_Old//2)),(y_old+(BboxHeight_Old//2)))
                    # Weighted_Area_new_tuple=(Weighted_Area_new,)
                    try :
                        old_index=All[k]['Old_index']
                        # print('i am here beforeeeee')
                        All_Old_Weighted_Avg=Old_Objetcs[old_index][CurrentIndex+1] # will be found as list of previous weighted # was 5
                        All_Old_Rate_Avg=Old_Objetcs[old_index][CurrentIndex+2] # will be found as list of previous Rate
                        All_CenterOfMass=Old_Objetcs[old_index][CurrentIndex+3] # will be found as list of previous Cnter of masses
                        All_Areas=Old_Objetcs[old_index][CurrentIndex+4] #will found all previous areas here
                        All_Areas_Signs=Old_Objetcs[old_index][CurrentIndex+5] #will found all previous signs the weighted areas 0.6 0.3 0.1
                        All_Speeds=Old_Objetcs[old_index][CurrentIndex+6] # will store all the speeds using centerof mass #was 10
                        All_PreviousStates=Old_Objetcs[old_index][CurrentIndex+7] # will store all the previous states whether it was white or black 
                        # All_Old_Weighted_Avg_tuple=(All_Old_Weighted_Avg,)
                        # print('i am here ')
                        # print(All_Old_Weighted_Avg_tuple)
                        try:
                            All_together= [*All_Old_Weighted_Avg,Weighted_Area_new]
                            All_together_Rate= [*All_Old_Rate_Avg,Current_New_Rate]
                            All_together_CenterOfMass= [*All_CenterOfMass,Current_CenteOfMass]

                            All_Areas= [*All_Areas,deresized_new]
                            All_Areas_Signs= [*All_Areas_Signs,ResultOfNewAvgAreasSign]

                            All_Speeds= [*All_Speeds,ResultSpeed]

                            All_PreviousStates= [*All_PreviousStates,ResultState] # resultState is the new state that must be followed (inverse of last state)
                            print('try')
                        except:
                            print('catch1')
                            All_together= [All_Old_Weighted_Avg,Weighted_Area_new]
                            All_together_Rate= [All_Old_Rate_Avg,Current_New_Rate]
                            All_together_CenterOfMass= [All_CenterOfMass,Current_CenteOfMass]
                            All_Areas= [All_Areas,deresized_new]
                            All_Areas_Signs= [All_Areas_Signs,ResultOfNewAvgAreasSign]
                            All_Speeds= [All_Speeds,ResultSpeed]

                            All_PreviousStates= [All_PreviousStates,ResultState]
                            print('catch2')
                        
                        print(All_together)
                        print(All_together_Rate)
                        print(All_together_CenterOfMass)
                        print('All_areas==',All_Areas)
                        print('All_Signs==',All_Areas_Signs)
                        print('All_Speeds==',All_Speeds)
                        print('All_PreviousStates==',All_PreviousStates)
                        

                    except :
                        print('no old till now wait')
                        All_together=[]
                        All_together_Rate=[]
                        All_together_CenterOfMass=[]
                        All_Areas=[]
                        All_Areas_Signs=[]
                        All_Speeds=[]
                        All_PreviousStates=[]
                       
              
                    Current_insert= New_Objetcs[new_index]+(All_together,All_together_Rate,All_together_CenterOfMass,All_Areas,All_Areas_Signs,All_Speeds,All_PreviousStates)
                    #now just replace this object in new objetcs with this tuple
                    New_Objetcs[new_index]=Current_insert
                    #now the new object will become old object and will contain the old weighted avg

                    
                    # print(Current_insert)
                    # print(Weighted_Area_new_tuple)

                  
                    Best_matches.append(All[k])
                    EnteredLoop=True
                    break 
            if EnteredLoop is False : # didnt match with any element (all row is zeros) means new object joined the framea
                print('New Object Foundddddd')
                #must add the type of this object but for now we will ignore it
                Current_insert ={'New_index':i ,'New_area':New_area,'scale_percent_new':scale_percent_new 
                ,'New_Dimensions':New_Dimensions,'New_Object':True,'New_Class':New_Class,'SpeedY':0,'CenterOfMassY':0,'New_Mask':New_Mask
                 ,'ImgWithoutBorderNew':ImgWithoutBorderNew ,'BorderImgNew':BorderImgNew}
                Best_matches.append(Current_insert)
                # All[0]['New_Object']=True #insert any dummy data but go later check the new_object flag and see if it a new object or not
            # print('current match:'+str(All[0]['good_match']))
            
        except:
            print('i fialllllllleddddddddddddddd')
        All.clear()
       
            # print(width)
    print('frame number='+str(current_frame))      
    current_frame=current_frame+1  
    height,width,_= frame.shape
    WindowArea=width*height








    for c in range(len(Best_matches)): #will use this to obtain each new object and get cal avgrate and then sort them
        NewOrNot=Best_matches[c]['New_Object'] 
        new=Best_matches[c]['New_index']  

        if NewOrNot is False: # not a new object izi
            try: 
                    # RateOfChangeAvg=RateOfChangeTracer(New_Objetcs[new][6])
                    CurrentSpeed=SpeedInY4Frames(New_Objetcs[new][CurrentIndex+3])
                    CurrentSpeedInY=CurrentSpeed[1]
                    Best_matches[c]['SpeedY']=CurrentSpeedInY 

                    CurrentCenterOfMass=New_Objetcs[new][CurrentIndex+3][-1] # to get the last record with me
                    CurrentCenterOfMassY=CurrentCenterOfMass[1]
                    Best_matches[c]['CenterOfMassY']=CurrentCenterOfMassY 
                  


                  
                  
            except: # this means that new object just became old so no old weighted for him 
                    print("I CANNNNNTtttttttttttttttttttttttt")

    print('BEFORRRRRRRE')
    # print(Best_matches)
    Best_matches.sort( reverse=True,key=SortMeSpeed)  # first sort depending on wt is faster 
    Best_matches.sort( reverse=True,key=SortMeCenterOfMass)  # second sort depending on how close the object is
    print('AFTEEEEEEER')
    # print(Best_matches)

    


    Greater_to_Smaller=0
    saveMe_blank=np.zeros((frame_size_mask[0],frame_size_mask[1]),np.uint8) # where i wll save the framed produced by saveMe
    saveMe_blank = cv.cvtColor(saveMe_blank,cv.COLOR_GRAY2RGB)

    saveMe_blank2=np.zeros((frame_size_mask[0],frame_size_mask[1]),np.uint8) # where i wll save the framed produced by saveMe
    saveMe_blank2 = cv.cvtColor(saveMe_blank2,cv.COLOR_GRAY2RGB)

    black_background=np.zeros((height+margin+margin,width+margin+margin,3),np.uint8) # to put the resultant masks on 
    print('BLACCCCCCCCCK SIZE')
    print(black_background.shape)
    for k in range(len(Best_matches)):
     
        NewOrNot=Best_matches[k]['New_Object'] 
        New_Dimensions=Best_matches[k]['New_Dimensions']
        x_new,x2_new,y_new,y2_new=New_Dimensions['x'],New_Dimensions['x2'],New_Dimensions['y'],New_Dimensions['y2']
        BboxWidth_New=x2_new-x_new
        BboxHeight_New=y2_new-y_new
        BboxArea_New = BboxWidth_New* BboxHeight_New
        new_area= Best_matches[k]['New_area']//(scale_percent_new*scale_percent_new)
        scale_percent_new=Best_matches[k]['scale_percent_new']
        new_percantage=math.ceil((new_area/WindowArea)*100)
             #this is the frame itself
        new=Best_matches[k]['New_index']  
        New_Mask=Best_matches[k]['New_Mask']  

        ImgWithoutBorderNew=Best_matches[k]['ImgWithoutBorderNew']
        BorderImgNew=Best_matches[k]['BorderImgNew']
         
        Current_CenteOfMass=((x_new+(BboxWidth_New//2)),(y_new+(BboxHeight_New//2))) ## check if the center of mass located in first half or sec half of screen
        
        Lower_center_dim= ((x2_new -(BboxWidth_New//2)),(y2_new -(BboxHeight_New//2)))
        cv.circle(frame,Lower_center_dim, 2, (0,0,255), -1)


        # cv.line(frame, (0,int(height//2)), (int(width),int(height//2)), (0,255,255), 2) # middle line

        # Upper_Point = (int(0.33*width),int(0)) 
        # Lower_Point = (int(0.1*width),int(height))
        # cv.line(frame, Upper_Point, Lower_Point, (0,255,255), 2) # line  starts from the 0.25 width of screen

        # Upper_Point2 = (int(0.66*width),int(0)) 
        # Lower_Point2 = (int(0.9*width),int(height))
        # cv.line(frame, Upper_Point2, Lower_Point2, (0,255,255), 2) # line  starts from the 0.75 width of screen
       


        try :
            new=Best_matches[k]['New_index']   
            All_Previous_States=New_Objetcs[new][CurrentIndex+7] #to store the current center of mass
            CurrentState=All_Previous_States[-1] # suppose that we have just inserted the new state so dont need to alternate just take the last in array
            print('the new try all previous states')
            print('CurrentState==',CurrentState)
        except :
            print('faileeeed the new except all previous states')
            CurrentState=256

        ExceedYThreshold =False
        YThreshold=int(height)-int((height)*0.25)
        # cv.line(frame, (0,YThreshold), (int(width),YThreshold), (0,0,255), 2) # imaginary threshold
        if (y2_new>=YThreshold):
            ExceedYThreshold=True
        else :
            ExceedYThreshold=False



        draw_or_not=EquationOfLineParallel((0.33*width,0),(0.1*width,height),(0.66*width,0),(0.9*width,height),Lower_center_dim)
        # if (y2_new >height/2 and draw_or_not==True): # means in down part of screen any object above this line wont be tracked
        # or will be tracked but with diff brightness
        Greater_to_Smaller+=1   # if greater to smaller is ==1 then this means that this object of order1
        OrderShow(frame,Greater_to_Smaller,x2_new-(BboxWidth_New),y2_new)


        

        if NewOrNot is False:
        # current_percantage=((BboxArea/WindowArea)*100)
            old=Best_matches[k]['Old_index']       # this is the frame itself
        
            # new=Best_matches[k]['New_index']     
            best=Best_matches[k]['good_match']

        
            Old_Dimensions=Best_matches[k]['Old_Dimensions']
        
            x_old,x2_old,y_old,y2_old=Old_Dimensions['x'],Old_Dimensions['x2'],Old_Dimensions['y'],Old_Dimensions['y2']
            IOU_percantage=UnionOverInterSection(Old_Dimensions,New_Dimensions)
        

        

            BboxWidth_Old=x2_old-x_old
            BboxHeight_Old=y2_old-y_old
            # BboxArea_New = BboxWidth_New* BboxHeight_New

            Middle_Point1=((x_new+(BboxWidth_New//2)),(y_new+(BboxHeight_New//2)))
            Middle_Point2=((x_old+(BboxWidth_Old//2)),(y_old+(BboxHeight_Old//2)))
            # Middle_Point2=((x2_old-x_old)//2,(y2_old-y_old)//2)
            # print(Middle_Point2)
        
            # x=Alldim[k]['x']
            # x2=Alldim[k]['x2']
            # y=Alldim[k]['y']
            # y2=Alldim[k]['y2']
            JoinedFlag=False
            new=Best_matches[k]['New_index']   
            try: 
                New_Weighted_Avg=Best_matches[k]['Weighted_Area']   
                InfoShow2(frame,New_Weighted_Avg,x_new,y_new)
                # Old_Weighted_Avg=Old_Objetcs[old][4][-1] # in place 4 we got the old weighted avg as we stored it
                ApprochingOrNotVar=ApprouchingOrNot(New_Objetcs[new][CurrentIndex+1])
                # print(New_Objetcs[new][4])
                # print('app or nooooooooot======'+str(ApprochingOrNotVar))
                RateOfChangeAvg=RateOfChangeTracer(New_Objetcs[new][CurrentIndex+2])
            
                print('heere MANA')
                print(New_Objetcs[new][CurrentIndex+2])
                # print('crurent_ength==='+str(len(New_Objetcs[new][5])))
                print('CURRENT_AVG_RATE='+str(RateOfChangeAvg))
                infoshowRateAvg(frame,RateOfChangeAvg,len(New_Objetcs[new][CurrentIndex+2]),x2_new,y2_new)


                ResultOfNewAvgAreas= int(RateAcross4Frames(New_Objetcs[new][CurrentIndex+4]))
                infoshowRate4frames(frame,ResultOfNewAvgAreas,x2_new,y2_new)
                print('ResultOfNewAvgAreas==',ResultOfNewAvgAreas)

                

                All_Signs_result=RateAcross4FramesSign(New_Objetcs[new][CurrentIndex+5])
                print('ResultOfNewAvgAreasSign==',All_Signs_result)
                infoshowRate4framesSign(frame,All_Signs_result,x2_new,y2_new)


                RateInCenterOfMass=RateInCenterOfMass4frames(New_Objetcs[new][CurrentIndex+3])
                print('RateInCenterOfMass==',RateInCenterOfMass)
                infoshowRate4frameCenterRate(frame,RateInCenterOfMass,x2_new-(BboxWidth_New),y2_new)
                DoubleCheck= DoubleCheckApproaching(RateInCenterOfMass)
                TripleCheck= TripleCheckApproaching(New_Objetcs[new][CurrentIndex+3])

                CurrentSpeed=SpeedInY4Frames(New_Objetcs[new][CurrentIndex+3])
                infoshowSpeed(frame,CurrentSpeed,x_new,y_new)



                
                JoinedFlag=True
            except: # this means that new object just became old so no old weighted for him 
                print("I CANNNNNT no previous Weight found")
        
            if JoinedFlag is False :
                Old_Weighted_Avg=0  # so that it wont cause i problem when comparing both the old_weight and new_weight
                ApprochingOrNotVar=1
                All_Signs_result='approaching sure'
                DoubleCheck='+veY'
                TripleCheck='not crossing'


            # try :
            #     SlopeTracer(New_Objetcs[new][6],frame,height)
            # except:
            #     print('excpeeing in slope')
            
                
            

        
            
            # cv.imshow('old',Old_Objetcs[old][0])      
            # cv.imshow('new',New_Objetcs[new][0])    
            # cv.waitKey(0)
            scale_percent_new=Best_matches[k]['scale_percent_new']
            scale_percent_old=Best_matches[k]['scale_percent_old']
            
            old_area= Best_matches[k]['Old_area'] //(scale_percent_old*scale_percent_old)
            new_area= Best_matches[k]['New_area']//(scale_percent_new*scale_percent_new)

            RateOfChangeShow(frame,old_area,new_area,x2_new,y2_new)

            print('oldddd area='+str(old_area))
            print('newwww area='+str(new_area))
        
            new_percantage=math.ceil((new_area/WindowArea)*100)
            BboxArea_New = BboxWidth_New* BboxHeight_New
        
            InfoShow(frame,BboxArea_New,new_percantage,x_new,y_new,IOU_percantage,x2_new,y2_new)
            #we could compare by the percantage better plus or minus 3% is still approaching

            New_Mask=Best_matches[k]['New_Mask']  
            Old_Mask=Best_matches[k]['Old_Mask']  
            print('y_new-margin==',y_new-margin)
            print('y2_new-margin==',y2_new+margin)
            print('x_new-margin==',x_new-margin)
            print('x2_new-margin==',x2_new-margin)
            add_y=0
            if y_new -margin <0 :
                add_y=margin

            add_x=0
            if x_new -margin <0 :
                add_x=margin

            ImgWithoutBorderNew=Best_matches[k]['ImgWithoutBorderNew']
            BorderImgNew=Best_matches[k]['BorderImgNew']

            ImgWithoutBorderOld=Best_matches[k]['ImgWithoutBorderOld'] # wont use this anyways
            BorderImgOld=Best_matches[k]['BorderImgOld'] # wont use this anyways

        

            # if (ApprochingOrNotVar==1 and IOU_percantage>0) : # approaching
            if (All_Signs_result=='approaching sure'   and IOU_percantage>0 and TripleCheck=='not crossing') : # approaching
                infoshowRate4framesSignFinal(frame,'Sapp1',x2_new-(BboxWidth_New),y2_new)
                cv.rectangle(frame, (x_new, y_new), (x2_new, y2_new), (0, 255, 0), 3) 
                black_background[y_new-margin+add_y:y2_new+margin +add_y, x_new-margin+add_x:x2_new+margin+add_x]=New_Mask

                saveMe_Border=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'approaching',ExceedYThreshold,Greater_to_Smaller,CurrentState,True)
                saveMe_blank=cv.bitwise_or(saveMe_blank,saveMe_Border)

                saveMe_Border2=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'approaching',ExceedYThreshold,Greater_to_Smaller,CurrentState,False)
                saveMe_blank2=cv.bitwise_or(saveMe_blank2,saveMe_Border2)

            elif ( All_Signs_result=='approaching maybe'and DoubleCheck=='+veY'and IOU_percantage>0  and TripleCheck=='not crossing') : # approaching sure
                infoshowRate4framesSignFinal(frame,'Sapp2',x2_new-(BboxWidth_New),y2_new)
                cv.rectangle(frame, (x_new, y_new), (x2_new, y2_new), (0, 255, 0), 3) 
                black_background[y_new-margin+add_y:y2_new+margin +add_y, x_new-margin+add_x:x2_new+margin+add_x]=New_Mask

                saveMe_Border=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'approaching',ExceedYThreshold,Greater_to_Smaller,CurrentState,True)
                saveMe_blank=cv.bitwise_or(saveMe_blank,saveMe_Border)

                saveMe_Border2=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'approaching',ExceedYThreshold,Greater_to_Smaller,CurrentState,False)
                saveMe_blank2=cv.bitwise_or(saveMe_blank2,saveMe_Border2)

            elif ( All_Signs_result=='approaching maybe'and DoubleCheck=='-veY'and IOU_percantage>0  and TripleCheck=='not crossing') : # maybe approaching both are -ve
                infoshowRate4framesSignFinal(frame,'Mapp',x2_new-(BboxWidth_New),y2_new)
                cv.rectangle(frame, (x_new, y_new), (x2_new, y2_new), (0, 0.5*255, 0), 3) 
                black_background[y_new-margin+add_y:y2_new+margin +add_y, x_new-margin+add_x:x2_new+margin+add_x]=New_Mask   

                saveMe_Border=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'approaching',ExceedYThreshold,Greater_to_Smaller,CurrentState,True)
                saveMe_blank=cv.bitwise_or(saveMe_blank,saveMe_Border)

                saveMe_Border2=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'approaching',ExceedYThreshold,Greater_to_Smaller,CurrentState,False)
                saveMe_blank2=cv.bitwise_or(saveMe_blank2,saveMe_Border2)
                
            # elif (ApprochingOrNotVar==0 and IOU_percantage>0):# goinf away

            elif (All_Signs_result=='going away' and DoubleCheck=='+veY' and IOU_percantage>0  and TripleCheck=='not crossing'):# maybe will be removed idk
                infoshowRate4framesSignFinal(frame,'Sapp3',x2_new-(BboxWidth_New),y2_new)
                cv.rectangle(frame,  (x_new, y_new), (x2_new, y2_new), (0, 255, 0), 3) 
                black_background[y_new-margin+add_y:y2_new+margin +add_y, x_new-margin+add_x:x2_new+margin+add_x]=New_Mask

                saveMe_Border=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'approaching',ExceedYThreshold,Greater_to_Smaller,CurrentState,True)
                saveMe_blank=cv.bitwise_or(saveMe_blank,saveMe_Border)

                saveMe_Border2=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'approaching',ExceedYThreshold,Greater_to_Smaller,CurrentState,False)
                saveMe_blank2=cv.bitwise_or(saveMe_blank2,saveMe_Border2)
            
            elif (All_Signs_result=='going away' and IOU_percantage>0  and TripleCheck=='not crossing'):# goinf away
                infoshowRate4framesSignFinal(frame,'away2',x2_new-(BboxWidth_New),y2_new)
                cv.rectangle(frame,  (x_new, y_new), (x2_new, y2_new), (255, 0, 0), 3) 
                black_background[y_new-margin+add_y:y2_new+margin +add_y, x_new-margin+add_x:x2_new+margin+add_x]=New_Mask

                saveMe_Border=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'away',ExceedYThreshold,Greater_to_Smaller,CurrentState,True)
                saveMe_blank=cv.bitwise_or(saveMe_blank,saveMe_Border)

                saveMe_Border2=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'away',ExceedYThreshold,Greater_to_Smaller,CurrentState,False)
                saveMe_blank2=cv.bitwise_or(saveMe_blank2,saveMe_Border2)

            elif (IOU_percantage>0   and TripleCheck=='crossing'):
                infoshowRate4framesSignFinal(frame,'crossing',x2_new-(BboxWidth_New),y2_new)
                cv.rectangle(frame,  (x_new, y_new), (x2_new, y2_new), (255, 0, 255), 3) 
                black_background[y_new-margin+add_y:y2_new+margin +add_y, x_new-margin+add_x:x2_new+margin+add_x]=New_Mask

                saveMe_Border=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'crossing',ExceedYThreshold,Greater_to_Smaller,CurrentState,True)
                saveMe_blank=cv.bitwise_or(saveMe_blank,saveMe_Border)

                saveMe_Border2=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'crossing',ExceedYThreshold,Greater_to_Smaller,CurrentState,False)
                saveMe_blank2=cv.bitwise_or(saveMe_blank2,saveMe_Border2)

            
            else : # miss macthed the red bbox
                #this means a miss match happens and must take further steps 
                Fail_counter+=1
                cv.line(frame, Middle_Point2, Middle_Point1, (255,255,255), 2)
                cv.rectangle(frame, (x_new, y_new), (x2_new, y2_new), (0, 0, 255), 3) 
                black_background[y_new-margin+add_y:y2_new+margin +add_y, x_new-margin+add_x:x2_new+margin+add_x]=New_Mask

                saveMe_Border=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'crossing',ExceedYThreshold,Greater_to_Smaller,CurrentState,True) # maybe we can consicder it as crossing but here miss match happens
                saveMe_blank=cv.bitwise_or(saveMe_blank,saveMe_Border)
                
                saveMe_Border2=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'crossing',ExceedYThreshold,Greater_to_Smaller,CurrentState,False)
                saveMe_blank2=cv.bitwise_or(saveMe_blank2,saveMe_Border2)
                # print('third If')
                
                # time.sleep(10)
                
                print('failed match='+str(best))
                # cv.imshow('old',Old_Objetcs[old][0])      
                # cv.imshow('new',New_Objetcs[new][0])    
                # cv.waitKey(0)
        else : # means new object 
            add_y=0
            if y_new -margin <0 :
                add_y=margin

            add_x=0
            if x_new -margin <0 :
                add_x=margin
            scale_percent_new=Best_matches[k]['scale_percent_new']
            # scale_percent_old=Best_matches[k]['scale_percent_old']
            new_area= Best_matches[k]['New_area']//(scale_percent_new*scale_percent_new)
            new_percantage=math.ceil((new_area/WindowArea)*100)
            BboxWidth_New=x2_new-x_new
            BboxHeight_New=y2_new-y_new
            BboxArea_New = BboxWidth_New* BboxHeight_New
            # print('new objecttttttt else')
            # print('new area='+str(new_area))
            InfoShow(frame,BboxArea_New,new_percantage,x_new,y_new,0,x2_new,y2_new)
            cv.rectangle(frame, (x_new, y_new), (x2_new, y2_new), (0, 255, 255), 3) 
            black_background[y_new-margin+add_y:y2_new+margin +add_y, x_new-margin+add_x:x2_new+margin+add_x]=New_Mask

            saveMe_Border=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'new',ExceedYThreshold,Greater_to_Smaller,CurrentState,True) # we will consider new as crossing till it goes to appraoching state
            saveMe_blank=cv.bitwise_or(saveMe_blank,saveMe_Border)

            saveMe_Border2=ReadFromImageTEST(ImgWithoutBorderNew,BorderImgNew,height,width,margin,x_new,x2_new,y_new,y2_new,'new',ExceedYThreshold,Greater_to_Smaller,CurrentState,False)
            saveMe_blank2=cv.bitwise_or(saveMe_blank2,saveMe_Border2)
            # cv.imshow('new object',New_Objetcs[new][0])    
            # cv.waitKey(0)
    
    fail(frame,Fail_counter) 
    Conflict(frame,Compare_String) 
    FrameNum(frame,current_frame) 
    black_background_To_Phosphene=ReadFromImage(black_background)
    print('typppppppppppppppppppppppe')
    print(black_background_To_Phosphene.shape)
    
    # black_background_To_Phosphene =cv.cvtColor(black_background_To_Phosphene,cv.COLOR_BGR2GRAY)

    convertme=cv.cvtColor(black_background_To_Phosphene,cv.COLOR_BGRA2BGR) ################### i hate u
    # output.write(black_background)

    output_Mask.write(black_background) # will save mask video here 
    output_Phos.write(black_background_To_Phosphene) # will save mask video here
    output_Border_Phos.write(saveMe_blank) #saving the threshold border video        
    output_Border_Phos_NoFlash.write(saveMe_blank2) #saving the threshold border video        
    cv.imshow('black background Mask',black_background) # this is a frame produced just turn frame into phosphene and test it
    cv.imshow('Without Border Pho',black_background_To_Phosphene) # this is a frame produced just turn frame into phosphene and test it
    cv.imshow('Border Pho',saveMe_blank) #this is phosphene with white border less go
   
    # return black_background_To_Phosphene





def FillUnionOverInterSection(IOU_Array) :
    # global IOU_Array
    for i in range(len(New_Objetcs)):
        New_Dimensions=New_Objetcs[i][2]
        for j in range(len(Old_Objetcs)):
            Old_Dimensions=Old_Objetcs[j][2]
            insert_value=UnionOverInterSection(Old_Dimensions,New_Dimensions)
            if (len(Old_Objetcs) > j):
                IOU_Array[i][j]=insert_value
            else :
                IOU_Array[i][j]=0
    if len(New_Objetcs) <len(Old_Objetcs):
        Current_insert=[0]*len(Old_Objetcs)
        IOU_Array[len(New_Objetcs)]=Current_insert

    # print(IOU_Array)  
    return IOU_Array


def FillMatching(Current_Matching) :
    # global IOU_Array
    # print(len(New_Objetcs))
    # print(len(Old_Objetcs))
    MaxAmong= max(len(Old_Objetcs),len(New_Objetcs))
    for i in range(len(New_Objetcs)):
        New_Object=New_Objetcs[i][0]
        for j in range(len(Old_Objetcs)):
            Old_Object=Old_Objetcs[j][0]
            sift = cv.SIFT_create()
            # surf = cv.xfeatures2d.SURF_create(500)

            # orb = cv.ORB_create()
            kpOld, desOld = sift.detectAndCompute(Old_Object, None)
            kpNew, desNew = sift.detectAndCompute(New_Object, None)

            bf=cv.BFMatcher()
            try:
               
                matches=bf.knnMatch(desOld,desNew,k=2)
                # good_matches_New =[[0 for i in range(MaxAmong)] for j in range(MaxAmong)]
                good_matches_New=[]
                for m,n in matches:
                    if m.distance<0.75*n.distance:
                        #TO BE SURE THAT MISS MATCH WONT HAPPEND BETWEEN DIFFERENT CLASSES
                        good_matches_New.append([m])
                
                Current_Matching[i][j]=len(good_matches_New)
            except:
                print('oppppppps fill function')
            # if (len(Old_Objetcs) > j):
            #     IOU_Array[i][j]=insert_value
            # else :
            #     IOU_Array[i][j]=0
    if len(New_Objetcs) <len(Old_Objetcs):
        Current_insert=[0]*len(Old_Objetcs)
        Current_Matching[len(New_Objetcs)]=Current_insert

    # print(IOU_Array)  
    return Current_Matching


def UnionOverInterSection(Old_Dimensions,New_Dimensions):
    x_new,x2_new,y_new,y2_new=New_Dimensions['x'],New_Dimensions['x2'],New_Dimensions['y'],New_Dimensions['y2']
    x_old,x2_old,y_old,y2_old=Old_Dimensions['x'],Old_Dimensions['x2'],Old_Dimensions['y'],Old_Dimensions['y2']
    First_Corner=(max(x_new,x_old),max(y_new,y_old))
    Second_Corner=(min(x2_new,x2_old),min(y2_new,y2_old))
    width=Second_Corner[0]-First_Corner[0]
    height=Second_Corner[1]-First_Corner[1]
    # print('current width'+str(width)+' current height:'+str(height))
    Inter_Area=width *height

    First_box_Area=(x2_new-x_new)* (y2_new-y_new)
    Second_box_Area=(x2_new-x_new)* (y2_new-y_new)
    Union_Area=(First_box_Area+Second_box_Area) - Inter_Area
    return(Inter_Area/Union_Area)
    #will return -ve value when miss matching happens .. could use this to detect if miss matching happens caused by orb

def Conflict_Result(IOU_Array,Matching_Array,IOU_Original,Matching_Original):
    Conflict_Counter=0 
    Location_Of_Conflict_IOU=-1 # if bith has changed means u have both location 
    Location_Of_Conflict_Matching=-1 
    Current_Column=-1
    for i in range(len(IOU_Array)):
        
        for j in range(len(IOU_Array)):
            IOU_Value =IOU_Array[j][i]
            Matching_Value =Matching_Array[j][i]
            if (Matching_Value!=0 and IOU_Value==0)  :
                Location_Of_Conflict_Matching=j # current row
                Current_Column=i
            elif (Matching_Value==0 and IOU_Value!=0) :
                Location_Of_Conflict_IOU=j
                Current_Column=i
            elif Location_Of_Conflict_IOU!=-1 and Location_Of_Conflict_Matching!=-1 and Current_Column!=-1: #now i have each location 
                Matching_result=Search_Another_Array1(Location_Of_Conflict_Matching,Current_Column,IOU_Original)
                IOU_result=Search_Another_Array2(Location_Of_Conflict_IOU,Current_Column,Matching_Original)

                IOU_Negative=All_Negative(Location_Of_Conflict_IOU,Current_Column,IOU_Original)
                if IOU_Negative is False: # akeed msh h match with -ve iou value
                    return 'USE MATCHING1'
                elif Matching_result is False : #mathcing for sure fails # must also check if iou is negative or all elemnt are negative
                    return 'USE IOU1'
                # elif IOU_result is False: #further 
                else :
                    Greater_Result_IOU=Greater_than(Location_Of_Conflict_IOU,Current_Column,IOU_Original)
                    Greater_Result_Matching=Greater_than(Location_Of_Conflict_Matching,Current_Column,Matching_Original)
                    if Greater_Result_Matching is False: # only matching fails means here iou wins
                        return 'USE IOU2'
                    elif Greater_Result_Matching is True :#here mathcing wins                   
                        return 'USE MATCHING2'
    # print('no conflict')            
    return 'no conflict'

def Search_Another_Array1(Location_Of_Conflict_Matching,Current_Column,IOU_Original):
    if IOU_Original[Location_Of_Conflict_Matching][Current_Column]<0: #-ve value this means matching array will lose 
        return False
    else :
        return True

def Search_Another_Array2(Location_Of_Conflict_Matching,Current_Column,Matching_Original):
    if Matching_Original[Location_Of_Conflict_Matching][Current_Column]<=5: #-ve value this means matching array will lose 
        return False
    else :
        return True
def Greater_than(Location_Of_Conflict,Current_Column,Current_Array):
    Compare_With=Current_Array[Location_Of_Conflict][Current_Column]
    for i in range(len(Current_Array)):
        Current_Element=Current_Array[Location_Of_Conflict][i]
        if(Current_Element>Compare_With): # means it will fail since another value were grater than it
            return False
    return True    

def All_Negative(Location_Of_Conflict_IOU,Current_Column,IOU_Original):
    Compare_With=IOU_Original[Location_Of_Conflict_IOU][Current_Column]
    Counter_Negative=0
    if Compare_With<0: # means currrent number is negative so no further calculation are needed
        return False
    for i in range(len(IOU_Original)):
        Current_Element=IOU_Original[Location_Of_Conflict_IOU][i]
        if(Current_Element!=Compare_With) and Current_Element<0: # means it will fail since another value were grater than it
            Counter_Negative+=1
    if (Counter_Negative == len(IOU_Original)-1): #means all are negative numbers
        return False        
    return True  

def WeightedAcross2Frames(Old_Area,New_Area):
    result = int(Old_Area*0.4 + New_Area*0.6)
    return result

def ApprouchingOrNot(Old_Weighted_Avg_Array):
    try :
        if len(Old_Weighted_Avg_Array) <4 :
            x=10/0 # to go for except
        else :
            for i in range((len(Old_Weighted_Avg_Array)-5),(len(Old_Weighted_Avg_Array)-1)):
                current= Old_Weighted_Avg_Array[i]
                next= Old_Weighted_Avg_Array[i+1]
                if next >=current:
                    # print((Old_Weighted_Avg_Array))
                    print('am in try bb')
                    return 1  #if only 1 out of the 4 frames is approaching so will consider it approaching 
                            #all 4 must fail so that we can now say its not apparoaching

            return 0 #means all 4 fails so its going away
      
    except :
        print('am in except bb')
        # print((Old_Weighted_Avg_Array))
        if len(Old_Weighted_Avg_Array) ==1 :
            return 1
        for i in range(len(Old_Weighted_Avg_Array)-1): # less than 4 frames are here so compare with 
            print('HEEEEEERE')
            current= Old_Weighted_Avg_Array[i]
            next= Old_Weighted_Avg_Array[i+1]
            print('current='+str(current))
            print('next='+str(next))
            if next >=current:
                return 1  #if only 1 out of the 4 frames is approaching so will consider it approaching 
        return 0 

def RateAcross4Frames(Old_Areas):
    try :
        if len(Old_Areas) <4 :
            x=10/0 # to go for except means not ready to produce cuz no 4 frames are ready 
        else :
            Result= calculateAccross4Frames(Old_Areas)
            return Result
    except :  
        return 'not 4 frames yet'

def calculateAccross4Frames(Old_area):
    result = Old_area[len(Old_area)-1]*1 - Old_area[len(Old_area)-2]*0.6 -Old_area[len(Old_area)-3]*0.3 -Old_area[len(Old_area)-4]*0.1 
    return result 

def RateAcross4FramesSign(Old_Areas):
    try :
        if len(Old_Areas) <4 :
            x=10/0 # to go for except means not ready to produce cuz no 4 frames are ready 
        else :
            if Old_Areas[len(Old_Areas)-1]>=0 : # only the current frame is +ve then am sure not need to trace the previous 4 frames too
                return 'approaching sure' 
            if (Old_Areas[len(Old_Areas)-1]>=0 and Old_Areas[len(Old_Areas)-2]>=0 and Old_Areas[len(Old_Areas)-3]>=0 and Old_Areas[len(Old_Areas)-4]>=0) :
                return 'approaching sure' #here am sure that its app 
            for i in range(len(Old_Areas)-4,len(Old_Areas)):
                current= Old_Areas[i] 
                if current>0 :
                    return 'approaching maybe' # one or more are +ve in the previous 4 frames
            return 'going away' # traced the 4 frames and all of them were -ve so its going away  
         
    except :  
        return 'not 4 frames yet'

def RateInCenterOfMass4frames(CenterOfMassArray):
    try :
        if len(CenterOfMassArray) <4 :
            x=10/0 # to go for except means not ready to produce cuz no 4 frames are ready 
        else :
            length=len(CenterOfMassArray)
            resultX =CenterOfMassArray[length-1][0]-CenterOfMassArray[length-4][0] # last point - before it by 4 frames-- if +ve then its app if -ve then going away 
            resultY =CenterOfMassArray[length-1][1]-CenterOfMassArray[length-4][1]
            return  (resultX,resultY)
         
    except :  
        return 'not 4 frames yet center of mass'

def SpeedInY4Frames(CenterOfMassArray):
    try :
        if len(CenterOfMassArray) <4 :
            x=10/0 # to go for except means not ready to produce cuz no 4 frames are ready 
        else :
            length=len(CenterOfMassArray)
            SpeedX=(CenterOfMassArray[length-1][0]-CenterOfMassArray[length-4][0])/4
            SpeedY=(CenterOfMassArray[length-1][1]-CenterOfMassArray[length-4][1])/4 # 4 means 4 frames 
             # we are using the time as 4 frames
            return  (SpeedX,SpeedY)
    except :  
        print('not 4 frames yet speed')
        return (0,0)

def DoubleCheckApproaching(CenterOfMass):
    x,y=CenterOfMass[0],CenterOfMass[1]
    if y>=0 :
        return '+veY'
    elif y<0 and abs(x) <10: # not moving diagonal yet so we can assume that its maybe approaching
        return '-veY'
    else :
        return 'X' # means that the object moving diagonal


def TripleCheckApproaching(CenterOfMassArray): # to trace the chnage in X
    length=len(CenterOfMassArray)
    resultX =CenterOfMassArray[length-1][0]-CenterOfMassArray[length-4][0] # last point - before it by 4 frames-- if +ve then its app if -ve then going away 
    resultY =CenterOfMassArray[length-1][1]-CenterOfMassArray[length-4][1]
    if (abs(resultX)>=10 ): # means that its crossing or moving diagonal 
        return 'crossing'
    else :
        return 'not crossing'


def FindTheNewState(All_Previous_States) :
    Last_State=All_Previous_States[-1] # using 3 levels
    # if (Last_State=='White'):
    #     return 'Gray'
    # elif (Last_State=='Gray'):
    #     return 'Black'
    # elif (Last_State=='Black'):
    #     return 'White'
    if (Last_State==0): # turned into black 5las
        return 256 
    else :
        return Last_State-16  


def RateOfChangeTracer(Old_Rate_Array):
    try :
        if len(Old_Rate_Array) <4 :
            x=10/0 # to go for except
        else :
            Sum=0 
            for i in range((len(Old_Rate_Array)-4),(len(Old_Rate_Array))):
                Sum+=Old_Rate_Array[i]
             
            return (Sum//4) #div by 4 cuz am sure that old_rate_array is of size 4 or more so am taking last 4 elemnts
      
    except :
        print('except Rate')
        # print((Old_Weighted_Avg_Array))
        Sum=0 
        for i in range(len(Old_Rate_Array)): # less than 4 frames are here so compare with 
             Sum+=Old_Rate_Array[i]
        return (Sum//len(Old_Rate_Array)) 


def SlopeTracer(Old_CenterOfMass_Array,frame,frame_height):
    try :
        if len(Old_CenterOfMass_Array) <10 :
            x=10/0 # to go for except
        else :
            print('OFFFFFFFFFFFF SIZE 4 OR GREATER')
            Last_Dimensions= Old_CenterOfMass_Array[len(Old_CenterOfMass_Array)-1] # last element in the array
            Previous_Dimensions= Old_CenterOfMass_Array[len(Old_CenterOfMass_Array)-10] # last element in the array

            x2,y2=Last_Dimensions[0],Last_Dimensions[1]
            x1,y1=Previous_Dimensions[0],Previous_Dimensions[1]
            print('CHNAGE IN X='+str(x2-x1))
            print('CHNAGE IN y='+str(y2-y1))
            cv.line(frame, Previous_Dimensions, Last_Dimensions, (255,255,255), 2) # line  starts from the 0.25 width of screen
            EquationOfLine(Last_Dimensions,Previous_Dimensions,frame_height,frame)
            print('am not heeeeeeeeere')

            
      
    except :
       print('SAAAAAAAAAAAAAAAAAD NOT OF SIZE 4 OR GREATER')


def EquationOfLine(Last_Dimensions,Previous_Dimensions,frame_height,frame):
    x2,y2=Last_Dimensions[0],Last_Dimensions[1]
    x1,y1=Previous_Dimensions[0],Previous_Dimensions[1]
    slope =(y2-y1) / (x2-x1)
    y_inter=y1-(slope*x1)
    newX_inter=(frame_height-y_inter)/slope
    cv.line(frame, Previous_Dimensions, (int(newX_inter),frame_height), (0,255,0), 2) # line  starts from the 0.25 width of screen


def EquationOfLineParallel(Firstpt_Left,Secpt_left,Firstpt_Right,Secpt_Right,Center_LowerBox):
    x2_left,y2_left=Firstpt_Left[0],Firstpt_Left[1]
    x1_left,y1_left=Secpt_left[0],Secpt_left[1]
    slope_left =(y2_left-y1_left) / (x2_left-x1_left)
    y_inter_left=y1_left-(slope_left*x1_left)
    #my left line equation is as follows y= slopeleft*X +y_inter_left

    x2_right,y2_right=Firstpt_Right[0],Firstpt_Right[1]
    x1_right,y1_right=Secpt_Right[0],Secpt_Right[1]
    slope_right =(y2_right-y1_right) / (x2_right-x1_right)
    y_inter_right=y1_right-(slope_right*x1_right)
    #my right line equation is as follows y= sloperight*X +y_inter_right
    X_bbox =Center_LowerBox[0]
    y_bbox=Center_LowerBox[1]
    Ynew_left = (slope_left * X_bbox) +y_inter_left 
    Ynew_right = (slope_right * X_bbox) +y_inter_right
    print('ynew_left=='+str(Ynew_left))
    print('ynew_right=='+str(Ynew_right))
    print('y_bbox=='+str(y_bbox))
    if( (Ynew_left < y_bbox  and Ynew_left >=0 )or (Ynew_right < y_bbox  and Ynew_right>=0)):
        print('yesssssssssssss you are inside me111111') 
        return True
    elif (Ynew_left<=0 and Ynew_right<=0) :
        print('yesssssssssssss you are inside me222222') 
        return True
    else :
        print('am ouuttttttttttttttttttt')
        return False







def RateOfChangeShow(frame,old_area,new_area,x2,y2):
    Rate=abs(new_area-old_area)
    infoshowRate(frame,Rate,x2,y2)


# def RateOfChangeAvgShow(frame,old_area,new_area,x2,y2):
#     Rate=abs(new_area-old_area)
#     infoshowRateAvg(frame,Rate,x2,y2)



    
def RateOfChangeCal(old_area,new_area):
    Rate=abs(new_area-old_area)
    return Rate
   




def SortMeSpeed(e):
    return e['SpeedY']    

def SortMeCenterOfMass(e):
    return e['CenterOfMassY']    


def myFunc(e):
    return e['area']


def myFunc2(e):
    return e['good_match']     




TrackVideoTEST()
# TrackImageSorted()


# Feature_Det()


# 











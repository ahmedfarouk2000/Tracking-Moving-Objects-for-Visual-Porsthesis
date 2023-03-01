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
        # if area=='percantage':
        #     finalresult="Percantage:"+str(percantage)+"%"
        #     outcolor=(0,255,255)
        #     location=(x,y)
        # elif area=='counter':
        #     finalresult="Counter:"+str(percantage)
        #     outcolor=(0,255,255)
        #     location=(x,y)
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
def TrackImage():

    cv.namedWindow("Control" , cv.WINDOW_NORMAL)
    cv.resizeWindow("Control", 1920, 290)
    cv.createTrackbar('Width','Control',500,1000,nothing)
    cv.createTrackbar('Height','Control',500,1000,nothing)
    cv.createTrackbar('Percentage','Control',7,10,nothing)
  
    cv.createTrackbar('Next','Control',0,1,nextimage)
    cv.createTrackbar('Previous','Control',0,1,nextimage)
    cv.createTrackbar('Switch','Control',0,1,nothing)
    cv.createTrackbar('Counter','Control',5,50,nothing)


    net = cv.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    img=cv.imread('photo/alotdepth2.jpg')
    allImages=os.listdir('/photo/trackppl')
    totalImages=len(allImages)
    counter=0 
    switchNext=0 
    switchPrev=0 


    while True :
        nextImg =cv.getTrackbarPos('Next','Control')
        prevImg =cv.getTrackbarPos('Previous','Control')
        
        if   nextImg==1 and switchNext==1 :
            counter+=1
            img =cv.imread('photo/trackppl/'+allImages[counter%totalImages]) 
            # capture =cv_img[counter%totalImages]
            switchNext=0 
            
        elif nextImg==0  and switchNext==0 :
            counter+=1
            img =cv.imread('photo/trackppl/'+allImages[counter%totalImages]) 
            switchNext=1
            
        elif prevImg==0 and  switchPrev==0 : 
                counter-=1
                img =cv.imread('photo/trackppl/'+allImages[counter%totalImages]) 
                switchPrev=1
            
        elif prevImg==1 and  switchPrev==1 :
                counter-=1
                img =cv.imread('photo/trackppl/'+allImages[counter%totalImages]) 
                switchPrev=0


        widthPhos = cv.getTrackbarPos('Width','Control')
        heightPhos = cv.getTrackbarPos('Height','Control')
        dimPhos = (widthPhos, heightPhos)
        img_resized=img.copy()
        img_resized = cv.resize(img_resized, dimPhos, interpolation = cv.INTER_AREA)  
        img_bbox=img_resized.copy()    

        height,width,_= img_resized.shape
        # print(height,width)
        blank=np.zeros((height,width),np.uint8)

      


        blob =cv.dnn.blobFromImage(img_resized,swapRB=True)
        net.setInput(blob)
        # print('working man')
        boxes,masks =net.forward(['detection_out_final',"detection_masks"])
        detection_count=boxes.shape[2]
        # print(detection_count)
        Percentage =cv.getTrackbarPos('Percentage','Control')
       

      
        Counter =cv.getTrackbarPos('Counter','Control')
        for i in range (Counter):
            #box[1]-->classtypeId, box[2]-->confidance
            box=boxes[0,0,i]
            # print(box[0])
            #class_id =0 are the humans
            class_id=box[1]
            #  if class_id==0 : #this means its a human
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
           
            WindowArea=width*height
            if (BboxArea/WindowArea)*100 >Percentage : 
                print(BboxArea)
                cv.rectangle(img_bbox, (x, y), (x2, y2), (255, 0, 0), 3) 
                # print(x2-x,y2-y)  
                # print(rec.shape[0])  
                
                # cv.imshow('ex',img)
                # cv.waitKey(0)

                roi =blank[y:y2 , x:x2] # this is the small object of interset we have execlusded from the image one by one
                roi_height,roi_wdith= roi.shape

                # #to get the mask we needed 
                mask =masks[i,int(class_id)] ## very very small its just 15x15 pixel size so we must scale it back to be bigger or the same original size
                
                mask=cv.resize(mask,(roi_wdith,roi_height))
                # cv.imshow('ahm',mask)
                # cv.waitKey(0)
                _,mask=cv.threshold(mask,0.1,255,cv.THRESH_BINARY)    
                
                contours,_=cv.findContours(np.array(mask, np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # cv.drawContours(image=blank, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=2)

                for i in contours:    
                    cv.fillPoly(roi, [i], (255,255,255))
                    # area_px=cv.contourArea(i)
                    # print(contours)
                    # print(area_px)



        masked=cv.bitwise_and(img_resized,img_resized,mask=blank) #bitwise must apply on image of shape (height,width) dont have 3rd att which is the channels
        Switch =cv.getTrackbarPos('Switch','Control')
        if Switch==0 :
            cv.imshow("Image", img_bbox)
        else :
            cv.imshow("Image",masked)

        # cv.imshow("real", img_resized)    
        # cv.imshow("Image", blank)
        # cv.imshow("Black image", blank)
        cv.waitKey(0) 



#didnt make yet the Sorted algo on the video functionnnnnnn
def TrackVideo():
    cv.namedWindow("Control" , cv.WINDOW_NORMAL)
    cv.resizeWindow("Control", 1920, 290)
    cv.createTrackbar('Percentage','Control',10,10,nothing)
    cv.createTrackbar('Switch','Control',0,1,nothing)

    capture =cv.VideoCapture('video/Newpeoplefull10fps.mp4') 
    net = cv.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    frame_size = (frame_width,frame_height)
    # print(frame_size)
    output = cv.VideoWriter('video/trackresult/output_video_from_file.avi', cv.VideoWriter_fourcc('M','J','P','G'), 15, frame_size)
    while True :
  
        # img=cv.imread('photo/humans.jpg')
        _,img=capture.read() #success is a flag that return true if can read from cam and false if not
        img_bbox=img.copy()    
     
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
        for i in range (detection_count):
            #box[1]-->classtypeId, box[2]-->confidance
            box=boxes[0,0,i]
            # print(box[0])
            #class_id =0 are the humans
            #first box[0] is a flag that define 0 not under any class of objects otherwise 1 under the class he  detected for
            class_id=box[1]
            if class_id==0 : #this means its a human
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
                
                cv.rectangle(img_bbox, (x, y), (x2, y2), (255, 0, 0), 3) 
                # cropped_img=img_bbox[y:y2,x:x2]
                # scale_percent = 10 # percent of original size
                # cropped_width,cropped_height,_=cropped_img.shape
                # width = int(cropped_width * scale_percent )
                # height = int(cropped_height * scale_percent )
                # dim = (width, height)
               
                # resized = cv.resize(cropped_img, dim, interpolation = cv.INTER_AREA)
                # cv.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3) 
                # print(x2-x,y2-y)  
                # print(rec.shape[0])  
                
                # cv.imshow('ex',img)
                # cv.waitKey(0)

                roi =blank[y:y2 , x:x2] # this is the small object of interset we have execlusded from the image one by one
                roi_height,roi_wdith= roi.shape

                # #to get the mask we needed 
                mask =masks[i,int(class_id)] ## very very small its just 15x15 pixel size so we must scale it back to be bigger or the same original size
                
                mask=cv.resize(mask,(roi_wdith,roi_height))
                # cv.imshow('ahm',mask)
                # cv.waitKey(0)
                _,mask=cv.threshold(mask,0.1,255,cv.THRESH_BINARY)    
                
                contours,_=cv.findContours(np.array(mask, np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # cv.drawContours(image=img, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=2)

                for i in contours:    
                    cv.fillPoly(roi, [i], (255,255,255))
                    #     area_px=cv.contourArea(i)
                        # print(contours)
                        # print(area_px)
        masked=cv.bitwise_and(img,img,mask=blank) #bitwise must apply on image of shape (h
        Switch =cv.getTrackbarPos('Switch','Control')
        if Switch==0 :
            cv.imshow("Image", img_bbox)
        else :
            cv.imshow("Image",masked)
        # output.write(img_bbox)
        # print(mask.shape)
        # print(img.shape)
        # cv.imshow('asdasd',mask)            
        if cv.waitKey(1)=='a':
                    cv.destroyAllWindows() 
                    break
        
    # cv.imshow("Image", mask)
    # # cv.imshow("Image", blank)
    # # cv.imshow("Black image", blank)
    # cv.waitKey(0)    
    # 


Old_Path = glob.glob("C:/Users/AHMED FAROUK/Documents/python computer vision/old frame objects/*")
New_Path = glob.glob("C:/Users/AHMED FAROUK/Documents/python computer vision/new frame objects/*")
Old_Objetcs = []
for img in Old_Path:
    n = cv.imread(img)
    Old_Objetcs.append(n)

New_Objetcs = []
for img in New_Path:
    n = cv.imread(img)
    New_Objetcs.append(n) 



def TrackVideoTEST():
    cv.namedWindow("Control" , cv.WINDOW_NORMAL)
    cv.resizeWindow("Control", 1920, 290)
    cv.createTrackbar('Percentage','Control',10,10,nothing)
    cv.createTrackbar('Switch','Control',0,1,nothing)

    # capture =cv.VideoCapture('video/london720full.mp4') 
    capture =cv.VideoCapture('video/car1.mp4') 
    net = cv.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    frame_size = (frame_width,frame_height)
  
    # print(frame_size)
    output = cv.VideoWriter('video/trackresult/output_video_from_file.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, frame_size)
    First_Frame=0 
    Counter=0
    # Alldim=[]
    while True :
       
  
        # img=cv.imread('photo/humans.jpg')
        _,img=capture.read() #success is a flag that return true if can read from cam and false if not
        img_bbox=img.copy()   

        Upper_Point = (int(0.33*frame_width),int(0)) 
        Lower_Point = (int(0.1*frame_width),int(frame_height))
        cv.line(img_bbox, Upper_Point, Lower_Point, (0,255,255), 2) # line  starts from the 0.25 width of screen

        
        Upper_Point2 = (int(0.66*frame_width),int(0)) 
        Lower_Point2 = (int(0.9*frame_width),int(frame_height))
        cv.line(img_bbox, Upper_Point2, Lower_Point2, (0,255,255), 2) # line  starts from the 0.75 width of screen
     
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
        for i in range (detection_count):
            #box[1]-->classtypeId, box[2]-->confidance
            box=boxes[0,0,i]
            # print(box[0])
            #class_id =0 are the humans
            #first box[0] is a flag that define 0 not under any class of objects otherwise 1 under the class he  detected for
            class_id=box[1]
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
            # current_percantage=int((BboxArea/WindowArea)*100)
        
            # cv.rectangle(img_bbox, (x, y), (x2, y2), (255, 255, 255), 3) 
            dimensions={'x':x,'x2':x2,'y':y,'y2':y2}

            
        
           


            if(BboxWidth>100): #wont fail to detect 
                # print('if')
                cropped_img=img_bbox[y:y2,x:x2]
                if First_Frame==0 :
                        # cv.imwrite('C:/Users/AHMED FAROUK/Documents/python computer vision/old frame objects/old'+str(Object_Counter)+'.jpg',cropped_img)
                        # Current_insert ={'image':cropped_img ,'scale_percent':1}
                        # Current_dimensions={'x':x,'y':y,'x2':x2,'y2':y2}
                        Current_insert=(cropped_img,1,dimensions,class_id)
                        
                        Old_Objetcs.append(Current_insert)
                        # Object_Counter+=1
                else :
                        # cv.imwrite('C:/Users/AHMED FAROUK/Documents/python computer vision/new frame objects/new'+str(Object_Counter)+'.jpg',cropped_img)
                        # Object_Counter+=1
                        # Current_dimensions={'x':x,'y':y,'x2':x2,'y2':y2}
                        # Current_insert ={'image':cropped_img ,'scale_percent':1}
                        Current_insert=(cropped_img,1,dimensions,class_id)
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
                    Current_insert=(resized,scale_percent,dimensions,class_id)
                    Old_Objetcs.append(Current_insert)
                    # Object_Counter+=1
                else :
                    # cv.imwrite('C:/Users/AHMED FAROUK/Documents/python computer vision/new frame objects/new'+str(Object_Counter)+'.jpg',cropped_img)
                    # Object_Counter+=1
                    # Current_dimensions={'x':x,'y':y,'x2':x2,'y2':y2}
                    Current_insert=(resized,scale_percent,dimensions,class_id)
                    New_Objetcs.append(Current_insert)
                    # Alldim.append(dimensions)
                    
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
    All=[]
    Best_matches=[]
    global Fail_counter
    global current_frame
    

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
        # print(New_Dimensions)
        
        for j in range(len(Old_Objetcs)):
        
            Old_Object=Old_Objetcs[j][0]
            scale_percent_old=Old_Objetcs[j][1]
            Old_area=Old_Object.shape[0]*Old_Object.shape[1]
            Old_Dimensions=Old_Objetcs[j][2]
            Old_Class=Old_Objetcs[j][3]
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
                ,'New_Class':New_Class,'Old_Class':Old_Class,'Avg_Rate':0}
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
                    print(Current_New_Rate)
                 
                    New_Dimensions=All[k]['New_Dimensions']
                    x_new,x2_new,y_new,y2_new=New_Dimensions['x'],New_Dimensions['x2'],New_Dimensions['y'],New_Dimensions['y2']
                    BboxWidth_New=x2_new-x_new
                    BboxHeight_New=y2_new-y_new
                    Current_CenteOfMass=((x_new+(BboxWidth_New//2)),(y_new+(BboxHeight_New//2)))

                    # Middle_Point2=((x_old+(BboxWidth_Old//2)),(y_old+(BboxHeight_Old//2)))
                    # Weighted_Area_new_tuple=(Weighted_Area_new,)
                    try :
                        old_index=All[k]['Old_index']
                        # print('i am here beforeeeee')
                        All_Old_Weighted_Avg=Old_Objetcs[old_index][4] # will be found as list of previous weighted
                        All_Old_Rate_Avg=Old_Objetcs[old_index][5] # will be found as list of previous Rate
                        All_CenterOfMass=Old_Objetcs[old_index][6] # will be found as list of previous Cnter of masses
                        # All_Old_Weighted_Avg_tuple=(All_Old_Weighted_Avg,)
                        # print('i am here ')
                        # print(All_Old_Weighted_Avg_tuple)
                        try:
                            All_together= [*All_Old_Weighted_Avg,Weighted_Area_new]
                            All_together_Rate= [*All_Old_Rate_Avg,Current_New_Rate]
                            All_together_CenterOfMass= [*All_CenterOfMass,Current_CenteOfMass]
                            print('try')
                        except:
                            print('catch1')
                            All_together= [All_Old_Weighted_Avg,Weighted_Area_new]
                            All_together_Rate= [All_Old_Rate_Avg,Current_New_Rate]
                            All_together_CenterOfMass= [All_CenterOfMass,Current_CenteOfMass]
                            print('catch2')
                        
                        print(All_together)
                        print(All_together_Rate)
                        print(All_together_CenterOfMass)
                        

                    except :
                        print('no old till now wait')
                        All_together=[]
                        All_together_Rate=[]
                        All_together_CenterOfMass=[]
                       
              
                    Current_insert= New_Objetcs[new_index]+(All_together,All_together_Rate,All_together_CenterOfMass)
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
                ,'New_Dimensions':New_Dimensions,'New_Object':True,'New_Class':New_Class,'Avg_Rate':0}
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
                    RateOfChangeAvg=RateOfChangeTracer(New_Objetcs[new][5])
                    # print(Best_matches[c])
                    Best_matches[c]['Avg_Rate']=RateOfChangeAvg # i just stored the avg here less go
                    # print('heere babbbbbbbbbbbbbbaaaaaaaaaaa')
                    # print(Best_matches[c])
                  
            except: # this means that new object just became old so no old weighted for him 
                    print("I CANNNNNTtttttttttttttttttttttttt")

    print('BEFORRRRRRRE')
    print(Best_matches)
    Best_matches.sort( reverse=True,key=SortMe)  # this will sort them in ascending order so that i can just use them to know who's faster
    print('AFTEEEEEEER')
    print(Best_matches)

    


    Greater_to_Smaller=0
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
         
        Current_CenteOfMass=((x_new+(BboxWidth_New//2)),(y_new+(BboxHeight_New//2))) ## check if the center of mass located in first half or sec half of screen
        cv.line(frame, (0,int(height//2)), (int(width),int(height//2)), (0,255,255), 2)
        Lower_center_dim= ((x2_new -(BboxWidth_New//2)),(y2_new -(BboxHeight_New//2)))
        cv.circle(frame,Lower_center_dim, 2, (0,0,255), -1)
        # Lower_center= 
       
     
        draw_or_not=EquationOfLineParallel((0.33*width,0),(0.1*width,height),(0.66*width,0),(0.9*width,height),Lower_center_dim)
        if (y2_new >height/2 and draw_or_not==True): # means in down part of screen any object above this line wont be tracked
            # or will be tracked but with diff brightness
            Greater_to_Smaller+=1
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
                    ApprochingOrNotVar=ApprouchingOrNot(New_Objetcs[new][4])
                    # print(New_Objetcs[new][4])
                    # print('app or nooooooooot======'+str(ApprochingOrNotVar))
                    RateOfChangeAvg=RateOfChangeTracer(New_Objetcs[new][5])
                
                    print('heere MANA')
                    print(New_Objetcs[new][5])
                    # print('crurent_ength==='+str(len(New_Objetcs[new][5])))
                    print('CURRENT_AVG_RATE='+str(RateOfChangeAvg))
                
                    infoshowRateAvg(frame,RateOfChangeAvg,len(New_Objetcs[new][5]),x2_new,y2_new)
                    # print(New_Objetcs[new][4])
                    # print(ApprochingOrNotVar)

                    #placed in last place till i trace the entire avg weighted
                    # print("old wieghttttttttttt="+str(Old_Weighted_Avg))
                    # print("new wieghttttttttttt="+str(New_Weighted_Avg))
                    JoinedFlag=True
                except: # this means that new object just became old so no old weighted for him 
                    print("I CANNNNNT no previous Weight found")
            
                if JoinedFlag is False :
                    Old_Weighted_Avg=0  # so that it wont cause i problem when comparing both the old_weight and new_weight
                    ApprochingOrNotVar=1


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
            

                if (ApprochingOrNotVar==1 and IOU_percantage>0) :
                
                    cv.rectangle(frame, (x_new, y_new), (x2_new, y2_new), (0, IOU_percantage*255, 0), 3) 
                
                elif (ApprochingOrNotVar==0 and IOU_percantage>0):
                    cv.rectangle(frame,  (x_new, y_new), (x2_new, y2_new), (255, 0, 0), 3) 
                
                else :
                    #this means a miss match happens and must take further steps 
                    Fail_counter+=1
                    cv.line(frame, Middle_Point2, Middle_Point1, (255,255,255), 2)
                    cv.rectangle(frame, (x_new, y_new), (x2_new, y2_new), (0, 0, 255), 3) 
                    # print('third If')
                    
                    # time.sleep(10)
                    
                    print('failed match='+str(best))
                    # cv.imshow('old',Old_Objetcs[old][0])      
                    # cv.imshow('new',New_Objetcs[new][0])    
                    # cv.waitKey(0)
            else : # means new object 
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
                # cv.imshow('new object',New_Objetcs[new][0])    
                # cv.waitKey(0)

    fail(frame,Fail_counter) 
    Conflict(frame,Compare_String) 





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
   



def TrackVideoSorted():
    cv.namedWindow("Control" , cv.WINDOW_NORMAL)
    cv.resizeWindow("Control", 1920, 290)
    cv.createTrackbar('Percentage','Control',5,10,nothing)
    cv.createTrackbar('Counter','Control',5,50,nothing)
    cv.createTrackbar('Switch','Control',0,1,nothing)

    capture =cv.VideoCapture('video/motorcycle.mp4') 
    net = cv.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    frame_size = (frame_width,frame_height)
    # print(frame_size)
    output = cv.VideoWriter('video/trackresult/output_video_from_file.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, frame_size)
    while True :
  
        # img=cv.imread('photo/humans.jpg')
        _,img=capture.read() #success is a flag that return true if can read from cam and false if not
        img_bbox=img.copy()    
     
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
        AllObjects=[]
        for i in range(detection_count):
            box=boxes[0,0,i]
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

            Current_bbox ={'index':i ,'area':BboxArea,'x':x,'y':y}
            AllObjects.append(Current_bbox)

        AllObjects.sort( reverse=True,key=myFunc) 
        Counter =cv.getTrackbarPos('Counter','Control') # the nearest n obejct that i want to display
        InfoShow(img_bbox,'percantage',Percentage,0,20)
        InfoShow(img_bbox,'counter',Counter,0,40) 
        if len(AllObjects) <Counter :
            Counter=len(AllObjects)
       
        for i in range (Counter):
            current_index=AllObjects[i]['index']
            box=boxes[0,0,current_index]
            # print(box[0])
            #class_id =0 are the humans
            #first box[0] is a flag that define 0 not under any class of objects otherwise 1 under the class he  detected for
            class_id=box[1]
            if class_id==0 : #this means its a human
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

                current_area=AllObjects[i]['area']
                current_x=AllObjects[i]['x']
                current_y=AllObjects[i]['y']
                current_percantage=int((BboxArea/WindowArea)*100)
                InfoShow(img_bbox,current_area,current_percantage,current_x,current_y)
              
                if int((BboxArea/WindowArea)*100) >Percentage : 
                    cv.rectangle(img_bbox, (x, y), (x2, y2), (255, 0, 0), 3) 
                    # cv.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3) 
                    # print(x2-x,y2-y)  
                    # print(rec.shape[0])  
                    
                    # cv.imshow('ex',img)
                    # cv.waitKey(0)

                    roi =blank[y:y2 , x:x2] # this is the small object of interset we have execlusded from the image one by one
                    roi_height,roi_wdith= roi.shape

                    # #to get the mask we needed 
                    mask =masks[i,int(class_id)] ## very very small its just 15x15 pixel size so we must scale it back to be bigger or the same original size
                    
                    mask=cv.resize(mask,(roi_wdith,roi_height))
                    # cv.imshow('ahm',mask)
                    # cv.waitKey(0)
                    _,mask=cv.threshold(mask,0.1,255,cv.THRESH_BINARY)    
                    
                    contours,_=cv.findContours(np.array(mask, np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    # cv.drawContours(image=img, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=2)

                    for i in contours:    
                        cv.fillPoly(roi, [i], (255,255,255))
                        #     area_px=cv.contourArea(i)
                            # print(contours)
                            # print(area_px)
        masked=cv.bitwise_and(img,img,mask=blank) #bitwise must apply on image of shape (h
        Switch =cv.getTrackbarPos('Switch','Control')
        if Switch==0 :
            cv.imshow("Image", img_bbox)
        else :
            cv.imshow("Image",masked)
        output.write(img_bbox)
        # print(mask.shape)
        # print(img.shape)
        # cv.imshow('asdasd',mask)            
        if cv.waitKey(1)=='a':
                    cv.destroyAllWindows() 
                    break    
 
    #                
def TrackImageSorted():

    cv.namedWindow("Control" , cv.WINDOW_NORMAL)
    cv.resizeWindow("Control", 1920, 290)
    cv.createTrackbar('Width','Control',500,1000,nothing)
    cv.createTrackbar('Height','Control',500,1000,nothing)
    cv.createTrackbar('Percentage','Control',7,10,nothing)
  
    cv.createTrackbar('Next','Control',0,1,nextimage)
    cv.createTrackbar('Previous','Control',0,1,nextimage)
    cv.createTrackbar('Switch','Control',0,1,nothing)
    cv.createTrackbar('Counter','Control',5,50,nothing)

    cv.createTrackbar('upper','Control',125,255,nothing)
    cv.createTrackbar('lower','Control',175,255,nothing)
    cv.createTrackbar('blur','Control',5,47,nothing)


    net = cv.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    img=cv.imread('photo/alotdepth2.jpg')
    allImages=os.listdir('photo/trackppl')
    totalImages=len(allImages)
    counter=0 
    switchNext=0 
    switchPrev=0 


    while True :
        nextImg =cv.getTrackbarPos('Next','Control')
        prevImg =cv.getTrackbarPos('Previous','Control')

        upper =cv.getTrackbarPos('upper','Control')
        lower =cv.getTrackbarPos('lower','Control')
        blur =cv.getTrackbarPos('blur','Control')
        if (blur %2==0):
            blur+=1
        
        if   nextImg==1 and switchNext==1 :
            counter+=1
            img =cv.imread('photo/trackppl/'+allImages[counter%totalImages]) 
            # capture =cv_img[counter%totalImages]
            switchNext=0 
            
        elif nextImg==0  and switchNext==0 :
            counter+=1
            img =cv.imread('photo/trackppl/'+allImages[counter%totalImages]) 
            switchNext=1
            
        elif prevImg==0 and  switchPrev==0 : 
                counter-=1
                img =cv.imread('photo/trackppl/'+allImages[counter%totalImages]) 
                switchPrev=1
            
        elif prevImg==1 and  switchPrev==1 :
                counter-=1
                img =cv.imread('photo/trackppl/'+allImages[counter%totalImages]) 
                switchPrev=0


        widthPhos = cv.getTrackbarPos('Width','Control')
        heightPhos = cv.getTrackbarPos('Height','Control')
        dimPhos = (widthPhos, heightPhos)
        img_resized=img.copy()
        img_resized = cv.resize(img_resized, dimPhos, interpolation = cv.INTER_AREA)  
        img_bbox=img_resized.copy()    

        height,width,_= img_resized.shape
        # print(height,width)
        blank=np.zeros((height,width),np.uint8)

      


        blob =cv.dnn.blobFromImage(img_resized,swapRB=True)
        net.setInput(blob)
        # print('working man')
        boxes,masks =net.forward(['detection_out_final',"detection_masks"])
        detection_count=boxes.shape[2]
        # print(detection_count)
        Percentage =cv.getTrackbarPos('Percentage','Control')
       

        AllObjects=[]
        for i in range(detection_count):
            box=boxes[0,0,i]
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

            Current_bbox ={'index':i ,'area':BboxArea,'x':x,'y':y}
            AllObjects.append(Current_bbox)

        AllObjects.sort( reverse=True,key=myFunc) 
        Counter =cv.getTrackbarPos('Counter','Control') # the nearest n obejct that i want to display
        # InfoShow(img_bbox,'percantage',Percentage,0,20)
        # InfoShow(img_bbox,'counter',Counter,0,40) 
        if len(AllObjects) <Counter :
            Counter=len(AllObjects)
        for i in range (Counter):
            #box[1]-->classtypeId, box[2]-->confidance
            
            current_index=AllObjects[i]['index']
          
            box=boxes[0,0,current_index]
            # print(box[0])
            #class_id =0 are the humans
            class_id=box[1]
            #  if class_id==0 : #this means its a human
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
            WindowArea=width*height
            current_area=AllObjects[i]['area']
            current_x=AllObjects[i]['x']
            current_y=AllObjects[i]['y']
            current_percantage=int((BboxArea/WindowArea)*100)
            # InfoShow(img_bbox,current_area,current_percantage,current_x,current_y)
          
           
            if int((BboxArea/WindowArea)*100) >Percentage : 
             
                cv.rectangle(img_bbox, (x, y), (x2, y2), (255, 0, 0), 3) 
              

                roi =blank[y:y2 , x:x2] # this is the small object of interset we have execlusded from the image one by one
                roi_height,roi_wdith= roi.shape
             

             
                mask =masks[i,int(class_id)] ## very very small its just 15x15 pixel size so we must scale it back to be bigger or the same original size
                
                mask=cv.resize(mask,(roi_wdith,roi_height))
             
                _,mask=cv.threshold(mask,0.1,255,cv.THRESH_BINARY)  
              
                
                contours,_=cv.findContours(np.array(mask, np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                  

                for i in contours:  
                    print('wowwwwwwwww')  
                    cv.fillPoly(roi, [i], (255,255,255))

                cv.imshow('roooooi',roi)
                cv.imshow('mask',mask)
        cv.imshow('blank',blank)
        MedianBlur = cv.medianBlur(blank, ksize=blur)
        CannyFliter =cv.Canny(MedianBlur , 125,175)
        cv.imshow('canny',CannyFliter)

        cv.imshow('median',MedianBlur)
        masked_temp=cv.bitwise_and(img_resized,img_resized,mask=MedianBlur) #bitwise must apply on image of shape (height,width) dont have 3rd att which is the channels
        print('img_resized==',img_resized.shape)
        print('mask==',MedianBlur.shape)
        cv.imshow('medinablur',masked_temp)
        cv.imshow('img_resized',img_resized)
        #so the the one will be used is masked_temp

        masked=cv.bitwise_and(img_resized,img_resized,mask=blank) #bitwise must apply on image of shape (height,width) dont have 3rd att which is the channels
        cv.imshow('blank',blank)

        Switch =cv.getTrackbarPos('Switch','Control')
        if Switch==0 :
            cv.imshow("Image", img_bbox)
            # cv.imwrite('C:/Users/AHMED FAROUK/Desktop/becholer research/img rep 1/outout.png',img_bbox)
        else :
            cv.imshow("Image",masked)

        # cv.imshow("real", img_resized)    
        # cv.imshow("Image", blank)
        # cv.imshow("Black image", blank)
        cv.waitKey(0)    

def myFunc(e):
  return e['area']

    
def myFunc2(e):
    return e['good_match']     

def SortMe(e):
    return e['Avg_Rate']    



# TrackVideoTEST()
TrackImageSorted()


# Feature_Det()


# 









